# test_module.py
import sys
print("Python path:", sys.path)

try:
    import cheapest_insertion
    print("Module contents:", dir(cheapest_insertion))
except ImportError as e:
    print("Import error:", e)