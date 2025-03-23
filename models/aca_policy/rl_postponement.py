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


# Double DQN
class PostponementValueNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super().__init__()
        self.fc_shared = nn.Linear(input_size, hidden_size)
        self.fc_value = nn.Linear(hidden_size, 1)
        self.fc_advantage = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = torch.relu(self.fc_shared(x))
        value = self.fc_value(x)
        advantage = self.fc_advantage(x)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

class RLPostponementDecision:
    """
    RL-based component for making postponement decisions using VFA with LNS.
    This class replaces the evaluate_postponement method in the original PostponementHandler.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        discount_factor: float = 0.95,
        exploration_rate: float = 0.9,
        exploration_decay: float = 0.995,
        min_exploration_rate: float = 0.05,
        batch_size: int = 32,
        training_mode: bool = True,
        state_size: int = 6,  # Number of features in state representation
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
        self.value_network = PostponementValueNetwork(state_size)  # No +1, action not in input
        self.optimizer = optim.Adam(self.value_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer()
        
        # Tracking for experience collection
        self.current_episode_states = {}  # order_id -> state tensor
        self.current_episode_actions = {}  # order_id -> action (0 or 1)
        self.step_count = 0
        
        # Initialize metrics
        self.episode_rewards = []
        self.total_training_steps = 0
        self.batch_losses = []
        # Add tracking for next states
        self.current_episode_next_states = {}  # order_id -> next_state tensor after action
    
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
        unassigned_orders = len(state.get("unassigned_orders", []))  # Orders not yet assigned
        total_assigned = sum(sum(len(p) + len(d) for _, p, d in route.sequence) for route in route_plan.values() if route.sequence)
        total_orders = total_assigned + unassigned_orders
        unassigned_ratio = unassigned_orders / max(1, total_orders)  # Ratio of unassigned to total
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

        # Round all features to 4 decimal places
        features = [round(f, 4) for f in features]

        return torch.tensor(features, dtype=torch.float32)
    
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
            action = 1 if random.random() < 0.5 else 0  # 50% postpone, 50% assign
            should_postpone = action == 1
        else:
            # Exploitation with LNS
            should_postpone = self._greedy_action_selection(state_tensor)

        # Store action for experience collection if in training mode
        if self.training_mode:
            self.current_episode_actions[order_id] = 1 if should_postpone else 0
        
        return should_postpone
        
    def estimate_value(self, state_tensor: torch.Tensor, action: int) -> float:
        with torch.no_grad():
            q_values = self.value_network(state_tensor.unsqueeze(0))  # [1, 2]
            return q_values[0, action].item()

    def _greedy_action_selection(self, state_tensor: torch.Tensor) -> bool:
        with torch.no_grad():
            q_values = self.value_network(state_tensor.unsqueeze(0))  # [1, 2]
            return q_values.argmax().item() == 1  # 1 if postpone has higher Q-value

    def _update_model(self) -> float:
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.cat(states)  # [batch_size, state_size]
        actions = torch.tensor(actions, dtype=torch.long)  # [batch_size]
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.cat(next_states)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Current Q-values for taken actions
        q_values = self.value_network(states)  # [batch_size, 2]
        current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # [batch_size]

        # Target Q-values
        with torch.no_grad():
            next_q_values = self.value_network(next_states)  # [batch_size, 2]
            max_next_q = next_q_values.max(dim=1)[0]  # [batch_size]
            targets = rewards + self.discount_factor * max_next_q * (1 - dones)

        loss = self.criterion(current_q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.total_training_steps += 1
        self.batch_losses.append(loss.item())
        return loss.item

    def update_from_rewards(self, rewards: Dict[int, float], next_state_dict: dict) -> float:
        if not self.training_mode or not self.current_episode_states:
            return 0.0

        for order_id in self.current_episode_states.keys():
            if order_id in self.current_episode_actions and order_id in rewards:
                state_tensor = self.current_episode_states[order_id]  # [6]
                action = self.current_episode_actions[order_id]
                reward = rewards[order_id]

                if order_id in next_state_dict["unassigned_orders"]:
                    next_state_tensor = self.extract_state_features(
                        order_id, next_state_dict["route_plan"], next_state_dict["time"], next_state_dict
                    )
                    done = False
                else:
                    next_state_tensor = torch.zeros_like(state_tensor)
                    done = True

                self.replay_buffer.add(state_tensor.unsqueeze(0), action, reward, next_state_tensor.unsqueeze(0), done)

        loss = self._update_model()
        self.current_episode_states = {}
        self.current_episode_actions = {}
        self._decay_exploration_rate()
        return loss

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

    def load_model(self, path: str) -> None:
        """Load model checkpoint from disk."""
        checkpoint = torch.load(path)
        self.value_network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.exploration_rate = checkpoint['exploration_rate']
        self.total_training_steps = checkpoint['total_training_steps']
        # logger.info(f"Model loaded from {path}")
        
