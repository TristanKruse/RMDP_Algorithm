# training/utils/file_io.py
import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def ensure_directory_exists(directory: str) -> None:
    """Ensure that a directory exists, create it if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_results(
    episode_stats: Dict[str, Any],
    solver_name: str,
    seed: Optional[int],
    meituan_config: Optional[Any],
    solver_params: Optional[Dict[str, Any]] = None,
    env_params: Optional[Dict[str, Any]] = None,
) -> None:
    """Save episode results to disk."""
    # Create timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create results directory if it doesn't exist
    results_dir = os.path.join("data", "results")
    ensure_directory_exists(results_dir)

    # Convert sets to lists for JSON serialization
    serializable_stats = {}
    for key, value in episode_stats.items():
        if isinstance(value, set):
            serializable_stats[key] = list(value)
        else:
            serializable_stats[key] = value

    # Prepare the results data
    results_data = {
        "timestamp": timestamp,
        "solver_name": solver_name,
        "seed": seed,
        "episode_stats": serializable_stats,
        "solver_params": solver_params,
        "env_params": env_params,
    }

    # Add Meituan config if available
    if meituan_config is not None:
        results_data["meituan_config"] = meituan_config.to_dict()

    # Save to JSON file
    filename = f"results_{solver_name}_{timestamp}.json"
    filepath = os.path.join(results_dir, filename)

    try:
        with open(filepath, "w") as f:
            json.dump(results_data, f, indent=2)
        logger.info(f"Results saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving results to {filepath}: {str(e)}")


def load_results(filepath: str) -> Dict[str, Any]:
    """Load results from a JSON file."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading results from {filepath}: {str(e)}")
        return {}


def save_visualization(figure, filename: str, directory: str = "data/visualizations") -> None:
    """Save a matplotlib figure to disk."""
    ensure_directory_exists(directory)
    filepath = os.path.join(directory, filename)

    try:
        figure.savefig(filepath)
        logger.info(f"Visualization saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving visualization to {filepath}: {str(e)}")


def save_model(model, filepath: str) -> None:
    """Save a model to disk."""
    directory = os.path.dirname(filepath)
    ensure_directory_exists(directory)

    try:
        model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving model to {filepath}: {str(e)}")


def load_model(filepath: str) -> Any:
    """Load a model from disk."""
    try:
        # This is a placeholder - actual implementation will depend on the model type
        # and the framework being used (e.g., PyTorch, TensorFlow)
        pass
    except Exception as e:
        logger.error(f"Error loading model from {filepath}: {str(e)}")
        return None
