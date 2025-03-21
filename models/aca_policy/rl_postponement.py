# models/rl_postponement.py
from typing import Set, Dict
import torch
import torch.nn as nn
import torch.optim as optim
import random
import logging
from collections import deque
import os

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
        # When calling self.value_network(x), this method is automatically executed by PyTorch
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
        learning_rate: float = 0.001,
        discount_factor: float = 0.95,
        exploration_rate: float = 0.3,
        exploration_decay: float = 0.995,
        min_exploration_rate: float = 0.05,
        batch_size: int = 32,
        training_mode: bool = True,
        state_size: int = 1,  # Number of features in state representation
        lns_sample_size: int = 5  # Number of neighborhood samples to evaluate
    ):
        # Core parameters
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
    
    def extract_state_features(self, order_id: int, route_plan: dict, current_time: float, state: dict) -> torch.Tensor:
        """
        Extract relevant features from the state for the RL model.
        """
        features = []
        
        # I. System-level features
        # 1. Feature: Time of day (normalized to [0,1])
        # Captures temporal patterns in demand
        time_of_day = (current_time / 60) % 24 / 24
        features.append(time_of_day)
        
        # 2. Feature: System utilization
        # When utilization is high, postponing might be beneficial
        total_vehicles = len(route_plan)
        busy_vehicles = sum(1 for r in route_plan.values() if r.sequence)
        system_utilization = busy_vehicles / max(1, total_vehicles)
        features.append(system_utilization)

        # 3. Feature: Unassigned Ratio
        # Important signal for deciding whether to clear backlog or postpone more
        active_orders = len(state.get("orders", [])) 
        total_assigned = sum(sum(len(p) + len(d) for _, p, d in route.sequence) for route in route_plan.values() if route.sequence)
        unassigned_ratio = (active_orders - total_assigned) / max(1, active_orders)
        features.append(unassigned_ratio)

        # II. Order-specific features
        order_info = state["unassigned_orders"].get(order_id)
        if order_info:
            # 4. Feature: Order Urgency
            # Critical for decision-making, how much time has passed relative to the delivery window
            total_window = order_info["deadline"] - order_info["request_time"]
            time_elapsed = current_time - order_info["request_time"]  # Time passed since order creation
            urgency = time_elapsed / total_window  # How much of the window has been used

            features.append(min(1.0, urgency))  # Cap at 1.0 for orders past deadline

            # 5. Feature: Bundling Potential
            # Calculate how many other unassigned orders are from the same restaurant
            bundling_potential = 0
            if order_info:  # Make sure order_info exists
                pickup_node_id = order_info["pickup_node_id"].id  # Get pickup_node_id from order_info
                
                # Count other unassigned orders from the same restaurant
                same_restaurant_orders = sum(
                    1 for o_id, o_info in state["unassigned_orders"].items() 
                    if o_id != order_id and o_info["pickup_node_id"].id == pickup_node_id
                )
                # Normalize to [0,1] - assuming rarely more than 5 orders from same restaurant
                bundling_potential = min(1.0, same_restaurant_orders / 5.0)

            features.append(bundling_potential)

            # 6. Feature: Restaurant Congestion
            # How many vehicles are already heading to or at this restaurant?
            restaurant_congestion = 0
            if order_info:  # Make sure order_info exists
                pickup_node_id = order_info["pickup_node_id"].id  # Get pickup_node_id from order_info
                
                # For this restaurant, how many orders are already assigned vs. how many vehicles are heading there?
                orders_assigned_to_restaurant = 0
                vehicles_heading_to_restaurant = 0

                for route in route_plan.values():
                    if not route.sequence:  # Skip empty routes
                        continue
                        
                    for node_id, pickups, _ in route.sequence:
                        if node_id == pickup_node_id:
                            orders_assigned_to_restaurant += len(pickups)
                            vehicles_heading_to_restaurant += 1
                            break

                # This ratio tells us how efficiently the restaurant's orders are being served
                if vehicles_heading_to_restaurant > 0:
                    orders_per_vehicle = orders_assigned_to_restaurant / vehicles_heading_to_restaurant
                    # Normalize to [0,1] - assuming 3 orders per vehicle is optimal
                    restaurant_congestion = min(1.0, orders_per_vehicle / 3.0)
                else:
                    restaurant_congestion = 0.0

            features.append(restaurant_congestion)
        else:
            # If order info not found, add placeholder values
            features.extend([0.0, 0.0, 0.0])
        
        # III. Spatial features - simplified for now
        # We'll add placeholder values for potential proximity features
        # 7. Feature: ....
        # features.extend([0.0, 0.0, 0.0])
        
        # Ensure we have the right number of features
        if len(features) < self.state_size:
            features.extend([0.0] * (self.state_size - len(features)))
        elif len(features) > self.state_size:
            features = features[:self.state_size]

        logger.debug(f"Order {order_id} features: {features}")
        
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
        # Extract state features
        state_tensor = self.extract_state_features(order_id, route_plan, current_time, state)
        
        # Store state for experience collection if in training mode
        if self.training_mode:
            self.current_episode_states[order_id] = state_tensor
        
        # Biased exploration - 90% chance to not postpone during exploration
        if self.training_mode and random.random() < self.exploration_rate:
            action = 0 if random.random() < 0.9 else 1  # 90% bias toward not postponing
            should_postpone = action == 1
        else:
            # Exploitation with LNS
            should_postpone = self._large_neighborhood_search(state_tensor)

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
    ) -> bool:

        # Evaluate both options (postpone vs. don't postpone)
        postpone_value = self.estimate_value(state_tensor, 1)
        dont_postpone_value = self.estimate_value(state_tensor, 0)
        
        # For debugging
        # logger.debug(f"Order {order_id}: postpone value = {postpone_value:.4f}, don't postpone value = {dont_postpone_value:.4f}")
        
        # Return action with higher estimated value
        return postpone_value > dont_postpone_value
    
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

    def save_model(self, path: str) -> None:
        """Save model checkpoint to disk."""
        if path is None:
            logger.warning("No save path provided, model not saved")
            return
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
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
        # logger.info(f"Model loaded from {path}")
        
    def set_training_mode(self, training: bool) -> None:
        """Switch between training and evaluation mode."""
        self.training_mode = training
        if not training:
            # Clear experience collection tracking when switching to evaluation
            self.current_episode_states = {}
            self.current_episode_actions = {}

