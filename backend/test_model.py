import os
import sys
import numpy as np
import scipy.io
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_squared_error, mean_absolute_error
)
import argparse
import logging

# --- Setup Project Root Path & Logging ---
# Assuming test_model.py is in the project root (e.g., backend/)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT) # Add project root to manage imports

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- Import from Project Modules & Config ---
try:
    from config.config import (
        SENSORS, SENSOR_NAMES_ORDERED, SPECIFIC_SENSOR_DAMAGE_PROFILES,
        DEFAULT_HEALTHY_GT_SCORE, ALL_SENSORS_DAMAGED_GT_SCORE,
        SPECIFIC_PRIMARY_DAMAGED_GT_SCORE_RANGE, # Used as fallback if ALL_SENSORS_DAMAGED_GT_SCORE not in older config
        WINDOW_SIZE as CFG_WINDOW_SIZE, # Renamed to avoid conflict with local WINDOW_SIZE
        OVERLAP as CFG_OVERLAP,
        BASE_DIR as CFG_BASE_DIR, # Base dir as defined in config.py
        GLOBAL_STATS_PATH as CFG_GLOBAL_STATS_PATH,
        LSTM_HIDDEN_SIZE, NUM_LSTM_LAYERS, DROPOUT_RATE # For model instantiation
    )
    # Construct paths relative to CFG_BASE_DIR (project root as seen by config.py)
    UNSEEN_DATA_DIR = os.path.join(CFG_BASE_DIR, "data", "unseen_data")
    GLOBAL_STATS_FILE = os.path.join(CFG_GLOBAL_STATS_PATH, "global_stats.npz")

    from src.models.model import GearboxCNNLSTM
except ImportError as e:
    logger.error(f"Error importing necessary modules or configurations: {e}")
    logger.error("Please ensure config/config.py and src/models/model.py are correctly set up and accessible.")
    logger.error(f"PROJECT_ROOT for this script: {PROJECT_ROOT}")
    sys.exit(1)

# --- Helper Functions ---

def load_global_stats(stats_file_path):
    """Loads global mean and std from the .npz file."""
    if not os.path.exists(stats_file_path):
        logger.error(f"Global stats file not found at {stats_file_path}")
        logger.error("Please run the calculate_global_stats.py script first.")
        sys.exit(1)
    try:
        stats = np.load(stats_file_path)
        global_mean = stats['global_mean']
        global_std = stats['global_std']
        logger.info(f"Loaded global mean (shape: {global_mean.shape}) and std (shape: {global_std.shape}) from {stats_file_path}")
        return global_mean, global_std
    except Exception as e:
        logger.error(f"Error loading global stats: {e}")
        sys.exit(1)

def create_windows_from_data(data, window_size, overlap):
    """Creates overlapping windows from sensor data. Data shape: (num_samples, num_channels)"""
    windows = []
    step = window_size - overlap
    if data.shape[0] < window_size:
        logger.warning(f"Data length {data.shape[0]} is less than window size {window_size}. No windows created.")
        return np.array([])
    for start in range(0, data.shape[0] - window_size + 1, step):
        windows.append(data[start:start + window_size, :])
    return np.array(windows) # Shape: (num_windows, window_size, num_channels)

def preprocess_single_mat_file(mat_filepath, global_mean, global_std,
                               window_size_cfg, overlap_cfg, num_sensors_cfg=SENSORS):
    """Loads a single .mat file, extracts sensor data, windows, and normalizes it using global stats."""
    try:
        mat_data = scipy.io.loadmat(mat_filepath)
        sensor_data_list = []
        for i in range(num_sensors_cfg): # Assumes AN3 to AN(3+num_sensors_cfg-1)
            sensor_key = f'AN{i+3}'
            if sensor_key in mat_data:
                # Ensure channel data is reshaped to be 2D (samples, 1) before flatten
                channel_data = mat_data[sensor_key].astype(np.float32).reshape(-1) 
                sensor_data_list.append(channel_data)
            else:
                logger.error(f"Sensor key {sensor_key} not found in {mat_filepath}.")
                return None
        
        # Ensure all channels have the same length by truncating to the minimum length
        min_len = min(len(ch) for ch in sensor_data_list)
        if min_len == 0:
            logger.error(f"One or more sensor channels are empty in {mat_filepath}.")
            return None
        sensor_data_array = np.array([ch[:min_len] for ch in sensor_data_list]).T # Shape: (min_len, num_sensors_cfg)

        windows = create_windows_from_data(sensor_data_array, window_size_cfg, overlap_cfg)
        if windows.size == 0:
            logger.warning(f"No windows created for {mat_filepath}. Skipping.")
            return None

        # Normalize data: (windows - mean) / std
        # Ensure global_mean and global_std are broadcastable (e.g., shape (1, 1, num_sensors_cfg))
        if global_mean.ndim == 1: global_mean = global_mean.reshape(1, 1, -1)
        if global_std.ndim == 1: global_std = global_std.reshape(1, 1, -1)

        normalized_windows = (windows - global_mean) / (global_std + 1e-8) # Add epsilon for stability
        return normalized_windows
    except Exception as e:
        logger.error(f"Error processing {mat_filepath}: {e}", exc_info=True)
        return None

def get_ground_truth_for_file(filename_simple, num_windows_created, num_sensors_cfg=SENSORS):
    """
    Generates ground truth labels (overall and per-sensor) for a given test file.
    Uses SPECIFIC_SENSOR_DAMAGE_PROFILES from config.py for known files like 'mixed.mat'.
    Simulates 'all damaged' for 'test_1.mat' and 'all healthy' for 'test_2.mat'.
    """
    gt_overall_label_scalar = 0  # Default to healthy
    gt_sensor_locations_pattern = np.full(num_sensors_cfg, DEFAULT_HEALTHY_GT_SCORE, dtype=np.float32)

    if filename_simple == "test_1.mat": # Simulates D*.mat (all sensors damaged)
        gt_overall_label_scalar = 1
        gt_sensor_locations_pattern[:] = ALL_SENSORS_DAMAGED_GT_SCORE
        logger.info(f"  GT for '{filename_simple}': Overall=Damaged, All Sensors GT Score={ALL_SENSORS_DAMAGED_GT_SCORE}")
    elif filename_simple == "test_2.mat": 
        gt_overall_label_scalar = 0
        gt_sensor_locations_pattern[:] = DEFAULT_HEALTHY_GT_SCORE
        logger.info(f"  GT for '{filename_simple}': Overall=Healthy, All Sensors GT Score={DEFAULT_HEALTHY_GT_SCORE}")
    elif SPECIFIC_SENSOR_DAMAGE_PROFILES and filename_simple in SPECIFIC_SENSOR_DAMAGE_PROFILES:
        profile = SPECIFIC_SENSOR_DAMAGE_PROFILES[filename_simple]
        gt_overall_label_scalar = 1 # Assume files in specific profiles are damaged overall
        
        healthy_score_for_profile = profile.get("target_healthy_score", DEFAULT_HEALTHY_GT_SCORE)
        gt_sensor_locations_pattern[:] = healthy_score_for_profile # Initialize all to healthy
        
        damaged_score_for_profile = profile.get("target_damaged_score", SPECIFIC_PRIMARY_DAMAGED_GT_SCORE_RANGE[1])
        damaged_indices_in_profile = profile.get("damaged_indices", [])
        
        for sensor_idx in damaged_indices_in_profile:
            if 0 <= sensor_idx < num_sensors_cfg:
                gt_sensor_locations_pattern[sensor_idx] = damaged_score_for_profile
            else:
                logger.warning(f"    Sensor index {sensor_idx} in profile for '{filename_simple}' is out of bounds.")
        logger.info(f"  GT for '{filename_simple}': Using SPECIFIC_SENSOR_DAMAGE_PROFILE. Overall=Damaged. Damaged sensor indices: {damaged_indices_in_profile} set to {damaged_score_for_profile}, others to {healthy_score_for_profile}.")
    else:
        logger.warning(f"No specific ground truth defined in get_ground_truth_for_file for '{filename_simple}'. Assuming healthy by default.")
        gt_overall_label_scalar = 0
        gt_sensor_locations_pattern[:] = DEFAULT_HEALTHY_GT_SCORE

    gt_overall_labels_array = np.full(num_windows_created, gt_overall_label_scalar, dtype=int)
    gt_sensor_locations_array = np.tile(gt_sensor_locations_pattern, (num_windows_created, 1))
    
    return gt_overall_labels_array, gt_sensor_locations_array

# --- Main Testing Logic ---
def run_tests(model_path_arg):
    logger.info("Starting model testing script...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. Load global statistics
    global_mean, global_std = load_global_stats(GLOBAL_STATS_FILE)

    # 2. Load Model
    # Model parameters should ideally match the trained model's configuration.
    # These are loaded from config.py
    model = GearboxCNNLSTM(
        window_size=CFG_WINDOW_SIZE, # From config, mainly for informational consistency
        lstm_hidden_size=LSTM_HIDDEN_SIZE,
        num_lstm_layers=NUM_LSTM_LAYERS,
        num_sensors=SENSORS,
        dropout_rate=DROPOUT_RATE # Set to 0.0 for eval if dropout was only for training regularization
                                 # Or keep if MC Dropout is intended (but model.eval() disables it by default)
    ).to(device)

    if not os.path.exists(model_path_arg):
        logger.error(f"Model file not found at {model_path_arg}")
        sys.exit(1)
    try:
        model.load_state_dict(torch.load(model_path_arg, map_location=device))
        model.eval() # Set model to evaluation mode
        logger.info(f"Model loaded from {model_path_arg} and set to evaluation mode.")
    except Exception as e:
        logger.error(f"Error loading model state_dict: {e}", exc_info=True)
        sys.exit(1)

    # 3. Define Unseen Files to Test
    # Ensure these files are in UNSEEN_DATA_DIR
    unseen_files_to_test = {
        "test_1.mat": "Simulates 'all sensors damaged' (like H*.mat)",
        "test_2.mat": "Simulates 'all sensors healthy' (like D*.mat)",
        "mixed.mat": "Localized damage: AN3, AN7, AN9 damaged (actual mixed profile)"
        # Add "mixed.mat" or other specific test files if you have them and defined their GT
    }

    all_results_summary = {}

    for filename, description in unseen_files_to_test.items():
        logger.info(f"\n--- Testing file: {filename} ({description}) ---")
        mat_filepath = os.path.join(UNSEEN_DATA_DIR, filename)

        if not os.path.exists(mat_filepath):
            logger.warning(f"File {mat_filepath} not found in {UNSEEN_DATA_DIR}. Skipping.")
            continue

        processed_windows = preprocess_single_mat_file(
            mat_filepath, global_mean, global_std,
            CFG_WINDOW_SIZE, CFG_OVERLAP, SENSORS
        )

        if processed_windows is None or processed_windows.size == 0:
            logger.warning(f"Could not process {filename}. Skipping.")
            continue
        
        num_windows_processed = processed_windows.shape[0]
        logger.info(f"Processed '{filename}' into {num_windows_processed} windows of shape {processed_windows.shape[1:]}")

        # Get Ground Truth for this file
        gt_overall_labels, gt_sensor_locations = get_ground_truth_for_file(filename, num_windows_processed, SENSORS)

        # Convert to PyTorch tensors
        windows_tensor = torch.from_numpy(processed_windows).float().to(device)
        # Model expects (batch, time_steps, features/channels)
        
        # Make Predictions
        pred_overall_probs_list = []
        pred_sensor_probs_list = []
        
        # Process in batches to avoid OOM for very large test files, though often not needed for single files
        batch_size_eval = 128 # Can be adjusted
        with torch.no_grad():
            for i in range(0, len(windows_tensor), batch_size_eval):
                batch_windows = windows_tensor[i:i+batch_size_eval]
                model_outputs = model(batch_windows)
                
                # Overall fault detection: model output is sigmoid if not self.training and not self.mc_dropout
                # If model outputs logits, apply sigmoid here.
                # Assuming model's 'fault_detection' is already probability during eval (due to internal sigmoid)
                pred_overall_raw_output_batch = model_outputs['fault_detection']
                if not (0 <= pred_overall_raw_output_batch.min() and pred_overall_raw_output_batch.max() <= 1):
                    # If output seems to be logits, apply sigmoid
                    logger.debug("Applying sigmoid to 'fault_detection' output as it doesn't appear to be probabilities.")
                    pred_overall_raw_output_batch = torch.sigmoid(pred_overall_raw_output_batch)

                # Per-sensor anomalies: model outputs logits, apply sigmoid
                pred_sensor_logits_batch = model_outputs['sensor_anomalies']
                pred_sensor_probs_batch = torch.sigmoid(pred_sensor_logits_batch)

                pred_overall_probs_list.append(pred_overall_raw_output_batch.cpu().numpy())
                pred_sensor_probs_list.append(pred_sensor_probs_batch.cpu().numpy())
        
        pred_overall_probs_all = np.concatenate(pred_overall_probs_list).squeeze()
        pred_sensor_probs_all = np.concatenate(pred_sensor_probs_list)
        
        # Binarize overall predictions using a threshold (e.g., 0.5 or a calibrated one)
        # For simplicity, using 0.5 here. You might want to use a calibrated threshold.
        overall_detection_threshold = 0.5 # TODO: Consider making this configurable or load from training
        pred_overall_binary = (pred_overall_probs_all > overall_detection_threshold).astype(int)
        
        logger.info(f"  Sample Overall GT Labels (first 5): {gt_overall_labels[:5]}")
        logger.info(f"  Sample Overall Predicted Probs (first 5): {pred_overall_probs_all[:5].round(3)}")
        logger.info(f"  Sample Overall Predicted Binary (first 5, thresh={overall_detection_threshold}): {pred_overall_binary[:5]}")


        # --- Evaluate Overall Damage ---
        logger.info("\n  --- Overall Damage Evaluation ---")
        overall_accuracy = accuracy_score(gt_overall_labels, pred_overall_binary)
        overall_precision = precision_score(gt_overall_labels, pred_overall_binary, zero_division=0)
        overall_recall = recall_score(gt_overall_labels, pred_overall_binary, zero_division=0)
        overall_f1 = f1_score(gt_overall_labels, pred_overall_binary, zero_division=0)
        overall_cm = confusion_matrix(gt_overall_labels, pred_overall_binary)

        logger.info(f"    Accuracy:  {overall_accuracy:.4f}")
        logger.info(f"    Precision: {overall_precision:.4f}")
        logger.info(f"    Recall:    {overall_recall:.4f}")
        logger.info(f"    F1-Score:  {overall_f1:.4f}")
        logger.info(f"    Confusion Matrix:\n{overall_cm}")

        # --- Evaluate Per-Sensor Damage/Location ---
        logger.info("\n  --- Per-Sensor Damage/Location Evaluation ---")
        # Binarize ground truth sensor locations (e.g., scores > 0.5 are considered damaged for GT)
        # The GT scores are already probabilities (0.05 for healthy, higher for damaged)
        # For metrics, we need a binary GT. Let's say GT > 0.5 means damaged.
        gt_sensor_binary = (gt_sensor_locations > 0.5).astype(int)
        
        # Binarize predicted sensor probabilities
        sensor_anomaly_threshold = 0.5 # TODO: Consider making this configurable
        pred_sensor_binary = (pred_sensor_probs_all > sensor_anomaly_threshold).astype(int)

        logger.info(f"  Sample Per-Sensor GT (first window, first 5 sensors): {gt_sensor_locations[0, :5].round(3)}")
        logger.info(f"  Sample Per-Sensor Pred Probs (first window, first 5 sensors): {pred_sensor_probs_all[0, :5].round(3)}")
        logger.info(f"  Sample Per-Sensor Pred Binary (first window, first 5 sensors, thresh={sensor_anomaly_threshold}): {pred_sensor_binary[0, :5]}")

        sensor_metrics = {}
        for i in range(SENSORS): # Use SENSORS from config
            sensor_name = SENSOR_NAMES_ORDERED[i] # Use SENSOR_NAMES_ORDERED from config
            gt_sensor_i = gt_sensor_binary[:, i]
            pred_sensor_i = pred_sensor_binary[:, i]
            
            acc = accuracy_score(gt_sensor_i, pred_sensor_i)
            prec = precision_score(gt_sensor_i, pred_sensor_i, zero_division=0)
            rec = recall_score(gt_sensor_i, pred_sensor_i, zero_division=0)
            f1 = f1_score(gt_sensor_i, pred_sensor_i, zero_division=0)
            
            sensor_metrics[sensor_name] = {"acc": acc, "prec": prec, "rec": rec, "f1": f1}
            logger.info(f"    Sensor {sensor_name}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")

        avg_sensor_acc = np.mean([m["acc"] for m in sensor_metrics.values()])
        avg_sensor_prec = np.mean([m["prec"] for m in sensor_metrics.values()])
        avg_sensor_rec = np.mean([m["rec"] for m in sensor_metrics.values()])
        avg_sensor_f1 = np.mean([m["f1"] for m in sensor_metrics.values()])
        logger.info(f"    Average (Macro) Sensor Metrics: Acc={avg_sensor_acc:.4f}, Prec={avg_sensor_prec:.4f}, Rec={avg_sensor_rec:.4f}, F1={avg_sensor_f1:.4f}")

        # MSE/MAE on raw predicted probabilities vs. raw GT scores
        sensor_mse = mean_squared_error(gt_sensor_locations, pred_sensor_probs_all)
        sensor_mae = mean_absolute_error(gt_sensor_locations, pred_sensor_probs_all)
        logger.info(f"    Sensor Scores MSE (Pred Probs vs GT Scores): {sensor_mse:.4f}")
        logger.info(f"    Sensor Scores MAE (Pred Probs vs GT Scores): {sensor_mae:.4f}")

        all_results_summary[filename] = {
            "overall_accuracy": overall_accuracy, "overall_f1": overall_f1,
            "avg_sensor_f1": avg_sensor_f1, "sensor_mse": sensor_mse
        }

    logger.info("\n\n--- Overall Test Summary ---")
    for filename, metrics in all_results_summary.items():
        logger.info(f"File: {filename}")
        logger.info(f"  Overall Accuracy: {metrics['overall_accuracy']:.4f}, Overall F1: {metrics['overall_f1']:.4f}")
        logger.info(f"  Avg Sensor F1:    {metrics['avg_sensor_f1']:.4f}, Sensor Scores MSE: {metrics['sensor_mse']:.4f}")
    logger.info("--- Testing script finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained Gearbox Fault Detection model on unseen .mat files.")
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help="Path to the trained .pth model file."
    )
    # Add other arguments if needed, e.g., to specify UNSEEN_DATA_DIR or thresholds
    
    args = parser.parse_args()
    
    if not os.path.exists(UNSEEN_DATA_DIR):
        logger.warning(f"UNSEEN_DATA_DIR '{UNSEEN_DATA_DIR}' does not exist. Creating it.")
        os.makedirs(UNSEEN_DATA_DIR, exist_ok=True)

    run_tests(args.model_path)
