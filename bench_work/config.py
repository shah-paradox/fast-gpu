import os

# --- Performance settings ---
# Number of threads for OpenMP
OMP_NUM_THREADS = "1"
# Number of CPUs for the processor and solver
N_CPU = 1

# --- Grid settings ---
GRID_X = 8
GRID_Y = 8

# --- Image paths ---
# The directory where your test images are stored
TEST_DIR = "tests"
# The prefix of the test case (e.g., "test0", "test3")
TEST_PREFIX = "test0"

# These will automatically construct the paths based on the settings above
SRC_PATH = f"{TEST_DIR}/{TEST_PREFIX}_src.png"
MASK_PATH = f"{TEST_DIR}/{TEST_PREFIX}_mask.png"
TGT_PATH = f"{TEST_DIR}/{TEST_PREFIX}_target.png"

# --- Offsets ---
# (y, x) offsets for the mask on source and target
SRC_OFFSET = (0, 0)
TGT_OFFSET = (0, 0)

# Apply environment variables
os.environ["OMP_NUM_THREADS"] = OMP_NUM_THREADS
