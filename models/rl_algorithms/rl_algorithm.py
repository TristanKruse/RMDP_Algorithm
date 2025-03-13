# models/rl_postponement.py
from typing import Set, Dict, Optional, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import logging
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PostponementValueNetwork(nn.Module):
    """Neural network for value function approximation in the postponement decision."""
    
    def __init__(self, input_size, hidden_size=64):
        super(PostponementValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)  # Single output: estimated value of state-action
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling agent experiences.
    """
    
    def __init__(self, capacity=10000):
        """
        Initialize replay buffer with fixed capacity.
        """
        self.buffer = deque(maxlen=capacity)
        
    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to the buffer.
        """
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        """
        Randomly sample a batch of experiences from the buffer.
        """
        return random.sample(self.buffer, min(len(self.buffer), batch_size))
    
    def __len__(self):
        """
        Get current number of experiences in buffer.
        """
        return len(self.buffer)

class RLPostponementDecision:
    """
    RL-based component for making postponement decisions using VFA with LNS.
    This class replaces the evaluate_postponement method in the original PostponementHandler.
    """
    
    def __init__(
        self,
        max_postponements: int,
        max_postpone_time: float,
        learning_rate: float = 0.001,
        discount_factor: float = 0.95,
        exploration_rate: float = 0.3,
        exploration_decay: float = 0.995,
        min_exploration_rate: float = 0.05,
        batch_size: int = 32,
        training_mode: bool = True,
        state_size: int = 10,  # Number of features in state representation
        lns_sample_size: int = 5  # Number of neighborhood samples to evaluate
    ):
        # Core parameters
        self.max_postponements = max_postponements
        self.max_postpone_time = max_postpone_time
        self.training_mode = training_mode
        self.state_size = state_size
        self.lns_sample_size = lns_sample_size
        
        # RL parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.batch_size = batch_size
        
        # Initialize neural network and optimizer
        self.value_network = PostponementValueNetwork(state_size + 1)  # +1 for action
        self.optimizer = optim.Adam(self.value_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer()
        
        # Tracking for experience collection
        self.postponed_orders = {}  # order_id -> (first_postpone_time, count)
        self.current_episode_states = {}  # order_id -> state tensor
        self.current_episode_actions = {}  # order_id -> action (0 or 1)
        self.step_count = 0
        
        # Initialize metrics
        self.episode_rewards = []
        self.total_training_steps = 0
        self.batch_losses = []
    
    def extract_state_features(self, order_id: int, postponed: Set[int], route_plan: dict, current_time: float, state: dict) -> torch.Tensor:
        """
        Extract relevant features from the state for the RL model.
        """
        features = []
        
        # 1. System-level features
        
        # Time of day (normalized to [0,1])
        time_of_day = (current_time / 60) % 24 / 24
        features.append(time_of_day)
        
        # System utilization
        total_vehicles = len(route_plan)
        busy_vehicles = sum(1 for r in route_plan.values() if r.sequence)
        system_utilization = busy_vehicles / max(1, total_vehicles)
        features.append(system_utilization)
        
        # Number of postponed orders (normalized)
        postponed_ratio = len(postponed) / max(1, self.max_postponements * 3)  # Scale factor
        features.append(postponed_ratio)
        
        # Count active orders and assignments
        active_orders = len(state.get("orders", [])) 
        total_assigned = sum(sum(len(p) + len(d) for _, p, d in route.sequence) for route in route_plan.values() if route.sequence)
        unassigned_ratio = (active_orders - total_assigned) / max(1, active_orders)
        features.append(unassigned_ratio)
        
        # 2. Order-specific features
        order_info = state["unassigned_orders"].get(order_id)
        if order_info:
            # Time since order creation (normalized)
            time_since_order = (current_time - order_info["request_time"]) / max(1, self.max_postpone_time * 2)
            features.append(min(1.0, time_since_order))  # Cap at 1.0
            
            # Previous postponements for this order
            prev_postponements = 0
            if order_id in self.postponed_orders:
                _, count = self.postponed_orders[order_id]
                prev_postponements = count / max(1, self.max_postponements)
            features.append(prev_postponements)
            
            # Is this order's restaurant already in a route?
            pickup_node_id = order_info["pickup_node_id"].id
            restaurant_in_routes = 0
            for route in route_plan.values():
                if route.sequence:
                    for node_id, _, _ in route.sequence:
                        if node_id == pickup_node_id:
                            restaurant_in_routes = 1
                            break
            features.append(restaurant_in_routes)
        else:
            # If order info not found, add placeholder values
            features.extend([0.0, 0.0, 0.0])
        
        # 3. Spatial features - simplified for now
        # We'll add placeholder values for potential proximity features
        features.extend([0.0, 0.0, 0.0])
        
        # Ensure we have the right number of features
        if len(features) < self.state_size:
            features.extend([0.0] * (self.state_size - len(features)))
        elif len(features) > self.state_size:
            features = features[:self.state_size]
        
        return torch.tensor(features, dtype=torch.float32)
    
    def estimate_value(self, state_tensor: torch.Tensor, action: int) -> float:
        """
        Estimate the value of a state-action pair using the value network.
        
        Args:
            state_tensor: Tensor of state features
            action: The action (0 = don't postpone, 1 = postpone)
            
        Returns:
            Estimated value of the state-action pair
        """
        # Concatenate state features with action
        action_tensor = torch.tensor([float(action)], dtype=torch.float32)
        state_action = torch.cat([state_tensor, action_tensor])
        
        # Pass through value network
        with torch.no_grad():
            value = self.value_network(state_action.unsqueeze(0)).item()
        
        return value
    
    def evaluate_postponement(
        self,
        postponed: Set[int],
        route_plan: dict,
        order_id: int,
        current_time: float,
        state: dict,
    ) -> bool:
        """
        Decide whether to postpone an order using VFA with LNS.
        """
        # Check basic constraints first
        if not self._can_postpone(order_id, postponed, route_plan, current_time, state):
            return False
        
        # Extract state features
        state_tensor = self.extract_state_features(order_id, postponed, route_plan, current_time, state)
        
        # Store state for experience collection if in training mode
        if self.training_mode:
            self.current_episode_states[order_id] = state_tensor
        
        # Exploration phase - with probability ε, explore randomly
        if self.training_mode and random.random() < self.exploration_rate:
            action = random.choice([0, 1])
            should_postpone = action == 1
        else:
            # Exploitation with Large Neighborhood Search
            should_postpone = self._large_neighborhood_search(state_tensor, order_id, postponed, route_plan, current_time, state)
        
        # Store action for experience collection if in training mode
        if self.training_mode:
            self.current_episode_actions[order_id] = 1 if should_postpone else 0
        
        # Track postponement if decided to postpone
        if should_postpone:
            self._track_postponement(order_id, current_time)
        
        return should_postpone
    
    def _large_neighborhood_search(
        self, 
        state_tensor: torch.Tensor, 
        order_id: int,
        postponed: Set[int],
        route_plan: dict,
        current_time: float,
        state: dict
    ) -> bool:
        """
        Perform large neighborhood search to find the best postponement decision.
        
        In this simplified version:
        1. We evaluate both options (postpone/don't postpone)
        2. We could add more complex neighborhood generation in a future version
        
        Returns:
            Boolean indicating whether to postpone
        """
        # Evaluate both options (postpone vs. don't postpone)
        postpone_value = self.estimate_value(state_tensor, 1)
        dont_postpone_value = self.estimate_value(state_tensor, 0)
        
        # For debugging
        logger.debug(f"Order {order_id}: postpone value = {postpone_value:.4f}, don't postpone value = {dont_postpone_value:.4f}")
        
        # Return action with higher estimated value
        return postpone_value > dont_postpone_value
    
    def _can_postpone(self, order_id: int, postponed: Set[int], route_plan: dict, current_time: float, state: dict) -> bool:
        """
        Check if order meets basic requirements for postponement.
        """
        # Check if already postponed too many times
        if order_id in self.postponed_orders:
            first_time, count = self.postponed_orders[order_id]
            if count >= self.max_postponements:
                return False
            if current_time - first_time >= self.max_postpone_time:
                return False
        
        # Check if order's restaurant is next stop for any vehicle
        order_info = state["unassigned_orders"].get(order_id)
        if not order_info:
            return False
            
        pickup_node_id = order_info["pickup_node_id"].id
        
        # Check each vehicle's next stop
        for route in route_plan.values():
            if route.sequence:  # If route has any stops
                next_stop = route.sequence[0]  # (node_id, pickups, deliveries)
                if next_stop[0] == pickup_node_id:  # If next stop is this restaurant
                    return False
        
        return True
    
    def _track_postponement(self, order_id: int, current_time: float) -> None:
        """Track when an order is postponed."""
        if order_id in self.postponed_orders:
            first_time, count = self.postponed_orders[order_id]
            self.postponed_orders[order_id] = (first_time, count + 1)
        else:
            self.postponed_orders[order_id] = (current_time, 1)
    
    def update_from_rewards(self, rewards: Dict[int, float]) -> float:
        """
        Update model using collected experiences and rewards.
        """
        if not self.training_mode or not self.current_episode_states:
            return 0.0
        
        # Add experiences to replay buffer
        for order_id in self.current_episode_states.keys():
            if order_id in self.current_episode_actions and order_id in rewards:
                state_tensor = self.current_episode_states[order_id]
                action = self.current_episode_actions[order_id]
                reward = rewards[order_id]
                
                # Concatenate state features with action for storage
                action_tensor = torch.tensor([float(action)], dtype=torch.float32)
                state_action = torch.cat([state_tensor, action_tensor]).unsqueeze(0)
                
                # For simplicity, we use the same state as next_state and set done=True
                self.replay_buffer.add(state_action, action, reward, state_action, True)
        
        # Update model if we have enough samples
        loss = self._update_model()
        
        # Clear episode tracking
        self.current_episode_states = {}
        self.current_episode_actions = {}
        
        # Decay exploration rate
        self._decay_exploration_rate()
        
        return loss
    
    def _update_model(self) -> float:
        """
        Update neural network model from replay buffer.
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        states, _, rewards, _, _ = zip(*batch)
        
        # Convert to tensors
        states = torch.cat(states)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        
        # Calculate current values
        current_values = self.value_network(states).squeeze()
        
        # Calculate loss (MSE between predicted values and actual rewards)
        loss = self.criterion(current_values, rewards)
        
        # Update model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Track training progress
        self.total_training_steps += 1
        self.batch_losses.append(loss.item())
        
        return loss.item()
    
    def _decay_exploration_rate(self) -> None:
        """Decay exploration rate according to schedule."""
        self.exploration_rate = max(self.min_exploration_rate, 
                                   self.exploration_rate * self.exploration_decay)
    
    def calculate_reward_from_delay(self, previous_delay: float, current_delay: float) -> float:
        """
        Calculate reward based on change in total delay.
        """
        return -(current_delay - previous_delay)
    
    def save_model(self, path: str) -> None:
        """Save model checkpoint to disk."""
        checkpoint = {
            'model_state_dict': self.value_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'exploration_rate': self.exploration_rate,
            'total_training_steps': self.total_training_steps
        }
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load model checkpoint from disk."""
        checkpoint = torch.load(path)
        self.value_network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.exploration_rate = checkpoint['exploration_rate']
        self.total_training_steps = checkpoint['total_training_steps']
        logger.info(f"Model loaded from {path}")
        
    def set_training_mode(self, training: bool) -> None:
        """Switch between training and evaluation mode."""
        self.training_mode = training
        if not training:
            # Clear experience collection tracking when switching to evaluation
            self.current_episode_states = {}
            self.current_episode_actions = {}