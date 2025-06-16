from models.aca_policy.aca_policy import ACA
from models.fastest_bundling.fastest_bundler import FastestBundler
from models.fastest_vehicle.fastest_vehicle import FastestVehicleSolver

SOLVERS = {
    "aca": lambda movement_per_step, location_manager: ACA(
        location_manager=location_manager,
        # Core algorithm parameters
        buffer=17,
        max_postponements=0,
        max_postpone_time=0,
        # Time & Vehicle parameters
        vehicle_capacity=5,  # test 5
        service_time=3.0,
        mean_prep_time=13.4,
        delivery_window=39.0,
        # Default to heuristic postponement
        postponement_method="heuristic",
    ),
    # Add RL-based ACA
    "rl_aca": lambda movement_per_step, location_manager: ACA(
        location_manager=location_manager,
        # Core algorithm parameters
        buffer=17,
        max_postponements=0,
        max_postpone_time=0,
        # Vehicle parameters
        vehicle_capacity=5,
        # Time parameters
        service_time=3.0,
        mean_prep_time=13.4,
        delivery_window=39.0,
        # Use RL-based postponement
        postponement_method="rl-aca",
        rl_training_mode=True,
        rl_state_size=7,
        rl_learning_rate=0.0005,
        rl_batch_size=64,
        rl_target_update_frequency=50,
        rl_discount_factor=0.95,
        rl_exploration_rate=0.9,
        rl_min_exploration_rate=0.05,
        rl_replay_buffer_capacity=10000,
        rl_bundling_reward=0.00,
        rl_postponement_penalty=0.00,
        rl_on_time_reward=0.0,
    ),
    "bundler": lambda s, loc_manager: FastestBundler(
        movement_per_step=s,
        location_manager=loc_manager,
        max_bundle_size=3,
    ),
    "fastest": lambda s, loc_manager: FastestVehicleSolver(movement_per_step=s, location_manager=loc_manager),
}
