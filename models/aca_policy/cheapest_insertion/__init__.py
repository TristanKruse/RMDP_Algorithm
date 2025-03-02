import os
import sys

# Add the directory containing the .pyd file to Python's path
sys.path.append(os.path.dirname(__file__))

# Now import from the module
from .cheapest_insertion import CheapestInsertion, Location, Stop

__all__ = ['CheapestInsertion', 'Location', 'Stop']