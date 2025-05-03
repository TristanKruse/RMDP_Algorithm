# # models/rl_postponement.py
# from typing import Set, Dict
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import random
# import logging
# from collections import deque
# import os

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# class ReplayBuffer:
#     def __init__(self, capacity=50000):  # Increased capacity
#         self.buffer = deque(maxlen=capacity)
        
#     def add(self, state, action, reward, delta_t):
#         self.buffer.append((state, action, reward, delta_t))
        
#     def sample(self, batch_size):
#         return random.sample(self.buffer, min(len(self.buffer), batch_size))
    
#     def __len__(self):
#         return len(self.buffer)

# class PostponementValueNetwork(nn.Module):
#     def __init__(self, input_size, hidden_size=128):
#         super().__init__()
#         self.fc_shared = nn.Linear(input_size, hidden_size)
#         self.fc_value = nn.Linear(hidden_size, 1)
#         self.fc_advantage = nn.Linear(hidden_size, 2)

#     def forward(self, x):
#         x = torch.relu(self.fc_shared(x))
#         value = self.fc_value(x)
#         advantage = self.fc_advantage(x)
#         return value + (advantage - advantage.mean(dim=1, keepdim=True))

# class RLPostponementDecision:
#     def __init__(
#         self,
#         learning_rate: float = 0.0005,
#         discount_factor: float = 0.95,
#         exploration_rate: float = 0.9,
#         exploration_decay: float = 0.99999,
#         min_exploration_rate: float = 0.2,
#         batch_size: int = 64,
#         training_mode: bool = True,
#         state_size: int = 6,
#         lns_sample_size: int = 5,
#         max_order_age: int = 300,
#         timeout_reward: float = -160.0,
#         target_update_frequency: int = 50,
#         replay_buffer_capacity: int = 50000,
#         bundling_reward: float = 0.05,
#         postponement_penalty: float = -0.005,
#         on_time_reward: float = 0.2
#     ):
#         self.training_mode = training_mode
#         self.state_size = state_size
#         self.lns_sample_size = lns_sample_size
        
#         self.learning_rate = learning_rate
#         self.discount_factor = discount_factor
#         self.exploration_rate = exploration_rate
#         self.exploration_decay = exploration_decay
#         self.min_exploration_rate = min_exploration_rate
#         self.batch_size = batch_size
#         self.target_update_frequency = target_update_frequency
#         self.bundling_reward = bundling_reward
#         self.postponement_penalty = postponement_penalty
#         self.on_time_reward = on_time_reward
        
#         self.value_network = PostponementValueNetwork(state_size)
#         self.target_network = PostponementValueNetwork(state_size)
#         self.target_network.load_state_dict(self.value_network.state_dict())
#         self.target_network.eval()
#         self.optimizer = optim.Adam(self.value_network.parameters(), lr=learning_rate)
#         self.criterion = nn.MSELoss()
        
#         self.replay_buffer = ReplayBuffer(capacity=replay_buffer_capacity)
#         self.current_episode_states = {}
#         self.current_episode_actions = {}
#         self.step_count = 0
        
#         self.episode_rewards = []
#         self.total_training_steps = 0
#         self.batch_losses = []
#         self.current_episode_next_states = {}
        
#         self.order_tracker = {}
#         self.completed_orders = []
#         self.max_order_age = max_order_age
#         self.timeout_reward = timeout_reward
#         self.order_postponement_steps = {}


#     def extract_state_features(self, order_id: int, route_plan: dict, current_time: float, state: dict) -> torch.Tensor:
#         features = []
#         time_of_day = (current_time / 60) % 24 / 24
#         features.append(time_of_day)
        
#         total_vehicles = len(route_plan)
#         busy_vehicles = sum(1 for r in route_plan.values() if r.sequence)
#         system_utilization = busy_vehicles / max(1, total_vehicles)
#         features.append(system_utilization)

#         unassigned_orders = len(state.get("unassigned_orders", []))
#         total_assigned = sum(sum(len(p) + len(d) for _, p, d in route.sequence) for route in route_plan.values() if route.sequence)
#         total_orders = total_assigned + unassigned_orders
#         unassigned_ratio = unassigned_orders / max(1, total_orders)
#         features.append(unassigned_ratio)

#         order_info = state["unassigned_orders"].get(order_id)
#         if order_info:
#             total_window = order_info["deadline"] - order_info["request_time"]
#             time_elapsed = current_time - order_info["request_time"]
#             urgency = time_elapsed / total_window
#             features.append(min(1.0, urgency))

#             bundling_potential = 0
#             pickup_node_id = order_info["pickup_node_id"].id
#             same_restaurant_orders = sum(
#                 1 for o_id, o_info in state["unassigned_orders"].items() 
#                 if o_id != order_id and o_info["pickup_node_id"].id == pickup_node_id
#             )
#             bundling_potential = min(1.0, same_restaurant_orders / 5.0)
#             features.append(bundling_potential)

#             restaurant_congestion = 0
#             pickup_node_id = order_info["pickup_node_id"].id
#             orders_assigned_to_restaurant = 0
#             vehicles_heading_to_restaurant = 0
#             for route in route_plan.values():
#                 if not route.sequence:
#                     continue
#                 for node_id, pickups, _ in route.sequence:
#                     if node_id == pickup_node_id:
#                         orders_assigned_to_restaurant += len(pickups)
#                         vehicles_heading_to_restaurant += 1
#                         break
#             if vehicles_heading_to_restaurant > 0:
#                 orders_per_vehicle = orders_assigned_to_restaurant / vehicles_heading_to_restaurant
#                 restaurant_congestion = min(1.0, orders_per_vehicle / 5.0)
#             features.append(restaurant_congestion)
#         else:
#             features.extend([0.0, 0.0, 0.0])
        
#         if len(features) < self.state_size:
#             features.extend([0.0] * (self.state_size - len(features)))
#         elif len(features) > self.state_size:
#             features = features[:self.state_size]

#         features = [round(f, 4) for f in features]
#         return torch.tensor(features, dtype=torch.float32)

#     def evaluate_postponement(self, postponed: Set[int], route_plan: dict, order_id: int, current_time: float, state: dict, exploration_rate=None) -> bool:
#         if exploration_rate is not None:
#             current_exploration = exploration_rate
#         else:
#             current_exploration = self.exploration_rate

#         state_tensor = self.extract_state_features(order_id, route_plan, current_time, state)
        
#         if self.training_mode:
#             self.current_episode_states[order_id] = state_tensor
        
#         if self.training_mode and random.random() < current_exploration:
#             action = 1 if random.random() < 0.5 else 0
#             should_postpone = action == 1
#         else:
#             should_postpone = self._greedy_action_selection(state_tensor)

#         if self.training_mode:
#             if order_id not in self.order_tracker:
#                 self.order_tracker[order_id] = {
#                     'actions': [],
#                     'first_seen': current_time,
#                     'delivered': False,
#                     'final_delay': None
#                 }
#                 self.order_postponement_steps[order_id] = 0  # Initialize postponement steps
#             self.order_tracker[order_id]['actions'].append((state_tensor, 1 if should_postpone else 0, current_time))
#             self.current_episode_states[order_id] = state_tensor
#             self.current_episode_actions[order_id] = 1 if should_postpone else 0
            
#             # If postponing, increment the postponement steps and add a small penalty
#             if should_postpone:
#                 self.order_postponement_steps[order_id] += 1
#                 # Add an intermediate experience with a postponement penalty
#                 self.replay_buffer.add(
#                     state=state_tensor.unsqueeze(0),
#                     action=1,
#                     reward=self.postponement_penalty,
#                     delta_t=0  # Immediate feedback
#                 )
#                 if len(self.replay_buffer) >= self.batch_size:
#                     self._update_model()

#         return should_postpone

#     def _process_completed_orders(self, current_time=None):
#         if not self.completed_orders:
#             return 0.0
        
#         experiences_added = 0
#         for order_id in self.completed_orders:
#             order_data = self.order_tracker[order_id]
#             if not order_data['delivered']:
#                 continue
                
#             final_delay = order_data['final_delay']
#             was_bundled = order_data.get('was_bundled', False)  # Get bundling info
#             max_delay = self.max_order_age
#             normalized_reward = -(final_delay / max_delay) if final_delay > 0 else 0.2
#             normalized_reward = max(-1.0, normalized_reward)
            
#             # Add bundling reward if the order was bundled
#             if was_bundled:
#                 normalized_reward += self.bundling_reward
#                 # print(f"Order {order_id} was bundled, adding bundling reward: {self.bundling_reward}")
            
#             for i, (state_tensor, action, action_time) in enumerate(order_data['actions']):
#                 delta_t = current_time - action_time
#                 self.replay_buffer.add(
#                     state=state_tensor.unsqueeze(0),
#                     action=action,
#                     reward=normalized_reward,
#                     delta_t=delta_t
#                 )
#                 experiences_added += 1
        
#         if len(self.replay_buffer) >= self.batch_size:
#             loss = self._update_model()
        
#         for order_id in self.completed_orders:
#             if order_id in self.order_tracker:
#                 del self.order_tracker[order_id]
#             if order_id in self.order_postponement_steps:
#                 del self.order_postponement_steps[order_id]
#         self.completed_orders = []
#         self._cleanup_old_orders(current_time)
#         return 0.0

#     def record_order_delivery(self, order_id, final_delay, current_time=None, was_bundled=False):
#         """Record the final delay when an order is delivered and whether it was bundled."""
#         if order_id in self.order_tracker:
#             self.order_tracker[order_id]['delivered'] = True
#             self.order_tracker[order_id]['final_delay'] = final_delay
#             self.order_tracker[order_id]['was_bundled'] = was_bundled  # Store bundling info
#             self.completed_orders.append(order_id)
#             self._process_completed_orders(current_time)

#     def _cleanup_old_orders(self, current_time):
#         if not current_time:
#             return
        
#         timeout_orders = []
#         for order_id, data in self.order_tracker.items():
#             if data['delivered']:
#                 continue
#             if current_time - data['first_seen'] > self.max_order_age:
#                 timeout_orders.append(order_id)
#                 normalized_timeout_reward = -1.0
#                 for i, (state_tensor, action, action_time) in enumerate(data['actions']):
#                     delta_t = current_time - action_time
#                     self.replay_buffer.add(
#                         state=state_tensor.unsqueeze(0),
#                         action=action,
#                         reward=normalized_timeout_reward,
#                         delta_t=delta_t
#                     )
        
#         if timeout_orders:
#             logger.debug(f"Timed out {len(timeout_orders)} orders: {timeout_orders}")
#         for order_id in timeout_orders:
#             del self.order_tracker[order_id]

#     def _greedy_action_selection(self, state_tensor: torch.Tensor) -> bool:
#         with torch.no_grad():
#             q_values = self.value_network(state_tensor.unsqueeze(0))
#             return q_values.argmax().item() == 1

#     def save_model(self, path: str) -> None:
#         if path is None:
#             logger.warning("No save path provided, model not saved")
#             return
#         os.makedirs(os.path.dirname(path), exist_ok=True)
#         checkpoint = {
#             'model_state_dict': self.value_network.state_dict(),
#             'optimizer_state_dict': self.optimizer.state_dict(),
#             'exploration_rate': self.exploration_rate,
#             'total_training_steps': self.total_training_steps,
#             'batch_losses': self.batch_losses
#         }
#         torch.save(checkpoint, path)

#     def load_model(self, path: str) -> None:
#         checkpoint = torch.load(path, weights_only=False)
#         self.value_network.load_state_dict(checkpoint['model_state_dict'])
#         self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         self.exploration_rate = checkpoint['exploration_rate']
#         self.total_training_steps = checkpoint['total_training_steps']
#         self.batch_losses = checkpoint.get('batch_losses', [])

#     def _update_model(self) -> float:
#         if len(self.replay_buffer) < self.batch_size:
#             return 0.0

#         batch = self.replay_buffer.sample(self.batch_size)
#         states, actions, rewards, delta_ts = zip(*batch)

#         states = torch.cat(states)
#         actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
#         rewards = torch.tensor(rewards, dtype=torch.float32)
#         delta_ts = torch.tensor(delta_ts, dtype=torch.float32)

#         current_q_values = self.value_network(states)
#         current_q = current_q_values.gather(1, actions).squeeze(1)

#         with torch.no_grad():
#             discount_factors = torch.pow(self.discount_factor, delta_ts)
#             target_q = rewards * discount_factors
#             logger.debug(f"Batch stats: Reward mean={rewards.mean().item():.2f}, "
#                          f"Discount factor mean={discount_factors.mean().item():.2f}, "
#                          f"Target Q mean={target_q.mean().item():.2f}, "
#                          f"Current Q mean={current_q.mean().item():.2f}")

#         loss = self.criterion(current_q, target_q)
#         self.optimizer.zero_grad()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), max_norm=1.0)
#         self.optimizer.step()

#         self.total_training_steps += 1
#         self.batch_losses.append(loss.item())

#         if self.total_training_steps % self.target_update_frequency == 0:
#             self.target_network.load_state_dict(self.value_network.state_dict())
#             logger.debug("Updated target network")

#         return loss.item()



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
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)
        
    def add(self, state, action, reward, delta_t):
        self.buffer.append((state, action, reward, delta_t))
        
    def sample(self, batch_size):
        return random.sample(self.buffer, min(len(self.buffer), batch_size))
    
    def __len__(self):
        return len(self.buffer)

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
    def __init__(
        self,
        learning_rate: float = 0.0005,
        discount_factor: float = 0.95,
        exploration_rate: float = 0.9,
        exploration_decay: float = 0.99999,
        min_exploration_rate: float = 0.2,
        batch_size: int = 64,
        training_mode: bool = True,
        state_size: int = 6,
        lns_sample_size: int = 5,
        target_update_frequency: int = 50,
        replay_buffer_capacity: int = 50000,
        bundling_reward: float = 0.05,
        postponement_penalty: float = -0.005,
        on_time_reward: float = 0.2
    ):
        self.training_mode = training_mode
        self.state_size = state_size
        self.lns_sample_size = lns_sample_size
        
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.bundling_reward = bundling_reward
        self.postponement_penalty = postponement_penalty
        self.on_time_reward = on_time_reward
        
        self.value_network = PostponementValueNetwork(state_size)
        self.target_network = PostponementValueNetwork(state_size)
        self.target_network.load_state_dict(self.value_network.state_dict())
        self.target_network.eval()
        self.optimizer = optim.Adam(self.value_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        self.replay_buffer = ReplayBuffer(capacity=replay_buffer_capacity)
        self.current_episode_states = {}
        self.current_episode_actions = {}
        self.step_count = 0
        
        self.episode_rewards = []
        self.total_training_steps = 0
        self.batch_losses = []
        self.current_episode_next_states = {}
        
        self.order_tracker = {}
        self.completed_orders = []
        self.order_postponement_steps = {}

    def extract_state_features(self, order_id: int, route_plan: dict, current_time: float, state: dict) -> torch.Tensor:
        features = []
        time_of_day = (current_time / 60) % 24 / 24
        features.append(time_of_day)
        
        total_vehicles = len(route_plan)
        busy_vehicles = sum(1 for r in route_plan.values() if r.sequence)
        system_utilization = busy_vehicles / max(1, total_vehicles)
        features.append(system_utilization)

        unassigned_orders = len(state.get("unassigned_orders", []))
        total_assigned = sum(sum(len(p) + len(d) for _, p, d in route.sequence) for route in route_plan.values() if route.sequence)
        total_orders = total_assigned + unassigned_orders
        unassigned_ratio = unassigned_orders / max(1, total_orders)
        features.append(unassigned_ratio)

        order_info = state["unassigned_orders"].get(order_id)
        if order_info:
            total_window = order_info["deadline"] - order_info["request_time"]
            time_elapsed = current_time - order_info["request_time"]
            urgency = time_elapsed / total_window
            features.append(min(1.0, urgency))

            bundling_potential = 0
            pickup_node_id = order_info["pickup_node_id"].id
            same_restaurant_orders = sum(
                1 for o_id, o_info in state["unassigned_orders"].items() 
                if o_id != order_id and o_info["pickup_node_id"].id == pickup_node_id
            )
            bundling_potential = min(1.0, same_restaurant_orders / 5.0)
            features.append(bundling_potential)

            restaurant_congestion = 0
            pickup_node_id = order_info["pickup_node_id"].id
            orders_assigned_to_restaurant = 0
            vehicles_heading_to_restaurant = 0
            for route in route_plan.values():
                if not route.sequence:
                    continue
                for node_id, pickups, _ in route.sequence:
                    if node_id == pickup_node_id:
                        orders_assigned_to_restaurant += len(pickups)
                        vehicles_heading_to_restaurant += 1
                        break
            if vehicles_heading_to_restaurant > 0:
                orders_per_vehicle = orders_assigned_to_restaurant / vehicles_heading_to_restaurant
                restaurant_congestion = min(1.0, orders_per_vehicle / 5.0)
            features.append(restaurant_congestion)
        else:
            features.extend([0.0, 0.0, 0.0])
        
        if len(features) < self.state_size:
            features.extend([0.0] * (self.state_size - len(features)))
        elif len(features) > self.state_size:
            features = features[:self.state_size]

        features = [round(f, 4) for f in features]
        return torch.tensor(features, dtype=torch.float32)

    def evaluate_postponement(self, postponed: Set[int], route_plan: dict, order_id: int, current_time: float, state: dict, exploration_rate=None) -> bool:
        if exploration_rate is not None:
            current_exploration = exploration_rate
        else:
            current_exploration = self.exploration_rate

        state_tensor = self.extract_state_features(order_id, route_plan, current_time, state)
        
        if self.training_mode:
            self.current_episode_states[order_id] = state_tensor
        
        if self.training_mode and random.random() < current_exploration:
            action = 1 if random.random() < 0.5 else 0
            should_postpone = action == 1
        else:
            should_postpone = self._greedy_action_selection(state_tensor)

        if self.training_mode:
            if order_id not in self.order_tracker:
                self.order_tracker[order_id] = {
                    'actions': [],
                    'first_seen': current_time,
                    'delivered': False,
                    'final_delay': None
                }
                self.order_postponement_steps[order_id] = 0
            self.order_tracker[order_id]['actions'].append((state_tensor, 1 if should_postpone else 0, current_time))
            self.current_episode_states[order_id] = state_tensor
            self.current_episode_actions[order_id] = 1 if should_postpone else 0
            
            if should_postpone:
                self.order_postponement_steps[order_id] += 1
                self.replay_buffer.add(
                    state=state_tensor.unsqueeze(0),
                    action=1,
                    reward=self.postponement_penalty,
                    delta_t=0
                )
                if len(self.replay_buffer) >= self.batch_size:
                    self._update_model()

        return should_postpone

    def _process_completed_orders(self, current_time=None):
        if not self.completed_orders:
            return 0.0
        
        experiences_added = 0
        for order_id in self.completed_orders:
            order_data = self.order_tracker[order_id]
            if not order_data['delivered']:
                continue
                
            final_delay = order_data['final_delay']
            was_bundled = order_data.get('was_bundled', False)
            normalized_reward = -(final_delay / 300) if final_delay > 0 else self.on_time_reward  # Use 300 as max delay for normalization
            normalized_reward = max(-1.0, normalized_reward)
            
            if was_bundled:
                normalized_reward += self.bundling_reward
                logger.debug(f"Order {order_id} was bundled, adding bundling reward: {self.bundling_reward}")
            
            for i, (state_tensor, action, action_time) in enumerate(order_data['actions']):
                delta_t = current_time - action_time if current_time else 0
                self.replay_buffer.add(
                    state=state_tensor.unsqueeze(0),
                    action=action,
                    reward=normalized_reward,
                    delta_t=delta_t
                )
                experiences_added += 1
        
        if len(self.replay_buffer) >= self.batch_size:
            self._update_model()
        
        for order_id in self.completed_orders:
            if order_id in self.order_tracker:
                del self.order_tracker[order_id]
            if order_id in self.order_postponement_steps:
                del self.order_postponement_steps[order_id]
        self.completed_orders = []
        return 0.0

    def record_order_delivery(self, order_id, final_delay, current_time=None, was_bundled=False):
        """Record the final delay when an order is delivered and whether it was bundled."""
        if order_id in self.order_tracker:
            self.order_tracker[order_id]['delivered'] = True
            self.order_tracker[order_id]['final_delay'] = final_delay
            self.order_tracker[order_id]['was_bundled'] = was_bundled
            self.completed_orders.append(order_id)
            self._process_completed_orders(current_time)

    def _greedy_action_selection(self, state_tensor: torch.Tensor) -> bool:
        with torch.no_grad():
            q_values = self.value_network(state_tensor.unsqueeze(0))
            return q_values.argmax().item() == 1

    def save_model(self, path: str) -> None:
        if path is None:
            logger.warning("No save path provided, model not saved")
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            'model_state_dict': self.value_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'exploration_rate': self.exploration_rate,
            'total_training_steps': self.total_training_steps,
            'batch_losses': self.batch_losses
        }
        torch.save(checkpoint, path)

    def load_model(self, path: str) -> None:
        checkpoint = torch.load(path, weights_only=False)
        self.value_network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.exploration_rate = checkpoint['exploration_rate']
        self.total_training_steps = checkpoint['total_training_steps']
        self.batch_losses = checkpoint.get('batch_losses', [])

    def _update_model(self) -> float:
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, delta_ts = zip(*batch)

        states = torch.cat(states)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        delta_ts = torch.tensor(delta_ts, dtype=torch.float32)

        current_q_values = self.value_network(states)
        current_q = current_q_values.gather(1, actions).squeeze(1)

        with torch.no_grad():
            discount_factors = torch.pow(self.discount_factor, delta_ts)
            target_q = rewards * discount_factors
            logger.debug(f"Batch stats: Reward mean={rewards.mean().item():.2f}, "
                         f"Discount factor mean={discount_factors.mean().item():.2f}, "
                         f"Target Q mean={target_q.mean().item():.2f}, "
                         f"Current Q mean={current_q.mean().item():.2f}")

        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.total_training_steps += 1
        self.batch_losses.append(loss.item())

        if self.total_training_steps % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.value_network.state_dict())
            logger.debug("Updated target network")

        return loss.item()