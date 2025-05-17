import os

# Data paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed") # Used as PROCESSED_DATA_PATH in client.py
GLOBAL_STATS_PATH = os.path.join(BASE_DIR, "data", "global_stats")

# Ensure directories exist
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(GLOBAL_STATS_PATH, exist_ok=True)

# --- Data processing parameters (used by preprocess.py) ---
WINDOW_SIZE = 128
OVERLAP = 64
RANDOM_STATE = 42

TRAIN_RATIO = 0.6
VALIDATION_RATIO_OF_REMAINING = 0.25
TEST_SIZE = 0.3

# Sensor configuration
SENSORS = 8
SENSOR_NAMES_ORDERED = [f'AN{i+3}' for i in range(SENSORS)]

# --- Ground Truth Generation Parameters ---
DEFAULT_HEALTHY_GT_SCORE = 0.05
ALL_SENSORS_DAMAGED_GT_SCORE = 0.95 # For D*.mat files in training
SPECIFIC_PRIMARY_DAMAGED_GT_SCORE_RANGE = (0.85, 0.95)
SPECIFIC_SECONDARY_DAMAGED_GT_SCORE_RANGE = (0.4, 0.6)

SPECIFIC_SENSOR_DAMAGE_PROFILES = {
    
    "mixed.mat": { 
        "damaged_indices": [
            SENSOR_NAMES_ORDERED.index("AN3"), 
            SENSOR_NAMES_ORDERED.index("AN7"), 
            SENSOR_NAMES_ORDERED.index("AN9")
        ],
        "target_healthy_score": DEFAULT_HEALTHY_GT_SCORE,
        "target_damaged_score": SPECIFIC_PRIMARY_DAMAGED_GT_SCORE_RANGE[1],
        "overall_fault_status": 1 # This file represents a damaged state overall
    },
    "test_1.mat": { # Simulates a D*.mat file - all sensors damaged
        "damaged_indices": list(range(SENSORS)), # All sensors
        "target_healthy_score": DEFAULT_HEALTHY_GT_SCORE, # Not really used as all are damaged
        "target_damaged_score": ALL_SENSORS_DAMAGED_GT_SCORE, # Consistent with D*.mat training GT
        "overall_fault_status": 1 # This file represents a damaged state overall
    },
    "test_2.mat": { # Simulates an H*.mat file - all sensors healthy
        "damaged_indices": [], # No sensors damaged
        "target_healthy_score": DEFAULT_HEALTHY_GT_SCORE,
        "target_damaged_score": DEFAULT_HEALTHY_GT_SCORE, # Not really used as no damaged_indices
        "overall_fault_status": 0 # This file represents a healthy state overall
    },
    "test_3.mat": {
        "damaged_indices": [
            SENSOR_NAMES_ORDERED.index("AN3"), 
            SENSOR_NAMES_ORDERED.index("AN5"), 
            SENSOR_NAMES_ORDERED.index("AN7"),
            SENSOR_NAMES_ORDERED.index("AN8"),
            SENSOR_NAMES_ORDERED.index("AN9"),
            SENSOR_NAMES_ORDERED.index("AN10")
        ],
        "target_healthy_score": DEFAULT_HEALTHY_GT_SCORE,
        "target_damaged_score": SPECIFIC_PRIMARY_DAMAGED_GT_SCORE_RANGE[1],
        "overall_fault_status": 1 # This file represents a damaged state overall
    }
}

GENERAL_DAMAGED_FILE_CONTEXT = {
    "D1.mat": "Represents widespread damage (e.g., HS-ST scuffing, HS-SH bearing overheating).",
    # ... other D*.mat contexts ...
}

# Feature Extraction Toggles
APPLY_FFT_FEATURES = False
APPLY_STATISTICAL_FEATURES = False

# --- Training parameters ---
BATCH_SIZE = 32
EPOCHS = 20 
LR = 0.0001 
WEIGHT_DECAY = 0.0001
MOMENTUM = 0.9

# Model architecture parameters
LSTM_HIDDEN_SIZE = 32
NUM_LSTM_LAYERS = 1
DROPOUT_RATE = 0.3

# Loss function parameters
LOCATION_LOSS_WEIGHT = 1.0
FOCAL_GAMMA = 2.0
POS_WEIGHT = 5.0
label_smoothing = 0.05

# Detection threshold parameters
DEFAULT_THRESHOLD = 0.5
MIN_RECALL = 0.5

# Scheduler settings
scheduler_patience = 5 

# Augmentation toggle
use_augmentation = True 

# Gradient clipping
gradient_clip_val = 1.0

# --- Federated Learning Specific Parameters ---
MAX_ROUNDS = 10 # Adjusted from your log
LOCAL_EPOCHS_FL = 2 
CLIENT_DATALOADER_NUM_WORKERS = 0 
LOCAL_FL_EARLY_STOP_PATIENCE = 2
LOCAL_FL_METRIC_BEST_MODEL = 'val_loss' 
LOCAL_FL_SCHEDULER_PATIENCE = 1 

early_stopping_patience = LOCAL_FL_EARLY_STOP_PATIENCE 
metric_for_best_model = LOCAL_FL_METRIC_BEST_MODEL   

try:
    import torch
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
except ImportError:
    print("Warning: PyTorch not found. Defaulting DEVICE to 'cpu'.")
    DEVICE = 'cpu'
