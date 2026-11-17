"""All settings in one place."""

# Random seed for reproducibility
SEED = 42

# Range of seeds for evaluating multiple splits
SPLIT_SEEDS = range(20)

# Number of neighbors for KNN
KNN_NEIGHBORS = 5

# Regularization parameter C for SVM
SVM_C = 10

# Number of trees in Random Forest
FOREST_TREES = 200

# Minimum confidence threshold for hand detection
CONFIDENCE_THRESHOLD = 0.6

# Buffer size for prediction history
MEMORY_SIZE = 10

# Consecutive frames required to hold a gesture
HOLD_FRAMES = 15

# Test dataset split fraction
TEST_SIZE = 0.2

# Path to the raw dataset CSV
RAW_DATA_PATH = "data/raw/psl_data.csv"