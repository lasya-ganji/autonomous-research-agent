import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

print("FINAL ROOT:", ROOT_DIR)
print("sys.path first 3 entries:", sys.path[:3])