import numpy as np
import scipy.io
import os
import sys
from sklearn.model_selection import train_test_split

# Add the project root to Python path to allow importing from config
# This assumes preprocess.py is in src/data_processing/
project_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root_path not in sys.path:
    sys.path.append(project_root_path)

try:
    from config.config import * # Imports all config variables
except ModuleNotFoundError as e:
    print(f"ERROR: Could not import configuration from config.config. Details: {e}")
    print("Please ensure config/config.py exists and project_root_path is correctly set for sys.path.")
    sys.exit(1)

# Define the path to the raw data directory using BASE_DIR from config
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw")


def create_windows(data, window_size, overlap):
    """Creates overlapping windows from sensor data."""
    windows = []
    step = window_size - overlap
    if data.shape[0] < window_size:
        print(f"Warning: Data length {data.shape[0]} is less than window size {window_size}. No windows created.")
        return np.array([])
    for start in range(0, data.shape[0] - window_size + 1, step):
        windows.append(data[start:start + window_size, :])
    return np.array(windows)

def normalize_data(data):
    """Normalize data using local statistics (client-specific)."""
    if data.size == 0:
        print("Warning: Attempting to normalize empty data array. Returning as is with zero mean/std.")
        num_channels = SENSORS
        return data, np.zeros((1,1,num_channels)), np.ones((1,1,num_channels))

    if data.ndim == 2:
        print(f"Warning: Normalizing 2D data (shape {data.shape}), reshaping to 3D assuming single channel.")
        data = data[:, :, np.newaxis]
    elif data.ndim != 3:
        raise ValueError(f"Data for normalization must be 3D (num_windows, window_length, num_channels), but got shape {data.shape}")
    
    local_mean = np.mean(data, axis=(0, 1), keepdims=True)
    local_std = np.std(data, axis=(0, 1), keepdims=True)
    local_std[local_std == 0] = 1e-8 
    
    normalized_data = (data - local_mean) / local_std
    return normalized_data, local_mean, local_std

def extract_fft_features(windows_data, sampling_rate=40000):
    if windows_data.size == 0: return np.array([])
    print("Applying FFT feature extraction (placeholder)...")
    fft_transform = np.fft.fft(windows_data, axis=1)
    fft_features = np.abs(fft_transform)
    fft_features = fft_features[:, :windows_data.shape[1] // 2, :]
    return np.mean(fft_features, axis=1)

def extract_statistical_features(windows_data):
    if windows_data.size == 0: return np.array([])
    print("Applying statistical feature extraction (placeholder)...")
    rms = np.sqrt(np.mean(windows_data**2, axis=1))
    return rms

def process_client(healthy_file, damaged_file, client_name):
    print(f"Processing data for client: {client_name} (Healthy: {healthy_file}, Damaged: {damaged_file})")
    
    healthy_mat_path = os.path.join(RAW_DATA_PATH, healthy_file)
    damaged_mat_path = os.path.join(RAW_DATA_PATH, damaged_file)

    if not os.path.exists(healthy_mat_path):
        print(f"ERROR: Healthy file not found for {client_name} at '{healthy_mat_path}'")
        return None
    if not os.path.exists(damaged_mat_path):
        print(f"ERROR: Damaged file not found for {client_name} at '{damaged_mat_path}'")
        return None
        
    try:
        healthy_mat = scipy.io.loadmat(healthy_mat_path)
        damaged_mat = scipy.io.loadmat(damaged_mat_path)
    except Exception as e:
        print(f"ERROR: Could not load .mat files for {client_name}. Details: {e}")
        return None
    
    try:
        healthy_sensor_data = np.vstack([healthy_mat[f'AN{i+3}'].flatten() for i in range(SENSORS)]).T
        damaged_sensor_data = np.vstack([damaged_mat[f'AN{i+3}'].flatten() for i in range(SENSORS)]).T
    except KeyError as e:
        print(f"ERROR: Sensor key {e} not found in .mat file for {client_name}. Ensure AN3-AN{3+SENSORS-1} exist.")
        return None

    operational_noise_level = np.random.uniform(0.05, 0.15)
    healthy_sensor_data += np.random.normal(0, operational_noise_level, healthy_sensor_data.shape)
    damaged_sensor_data += np.random.normal(0, operational_noise_level, damaged_sensor_data.shape)
    
    healthy_windows = create_windows(healthy_sensor_data, WINDOW_SIZE, OVERLAP)
    damaged_windows = create_windows(damaged_sensor_data, WINDOW_SIZE, OVERLAP)
    
    if healthy_windows.size == 0:
        print(f"ERROR: No healthy windows created for {client_name}. Skipping.")
        return None
    if damaged_windows.size == 0:
        print(f"ERROR: No damaged windows created for {client_name}. Skipping.")
        return None

    binary_healthy_labels = np.zeros(len(healthy_windows), dtype=int)
    binary_damaged_labels = np.ones(len(damaged_windows), dtype=int)

    healthy_locations = np.full((len(healthy_windows), SENSORS), DEFAULT_HEALTHY_GT_SCORE)
    
    damaged_locations = np.full((len(damaged_windows), SENSORS), DEFAULT_HEALTHY_GT_SCORE)
    profile_key = os.path.basename(damaged_file)
    ground_truth_method_applied = "UNKNOWN"
    most_affected_sensor_idx_for_metadata = -1

    # --- Start of Ground Truth Logic for Damaged Files ---
    print(f"  DEBUG: For damaged file '{profile_key}':")
    is_in_specific_profiles = SPECIFIC_SENSOR_DAMAGE_PROFILES and profile_key in SPECIFIC_SENSOR_DAMAGE_PROFILES
    print(f"  DEBUG: Is '{profile_key}' in SPECIFIC_SENSOR_DAMAGE_PROFILES? {is_in_specific_profiles}")

    # Condition for D*.mat files
    is_generic_d_mat_file = profile_key.upper().startswith("D") and profile_key.lower().endswith(".mat")
    print(f"  DEBUG: Is '{profile_key}' a generic D*.mat file? {is_generic_d_mat_file}")
    print(f"    DEBUG: profile_key.upper().startswith('D') = {profile_key.upper().startswith('D')}")
    print(f"    DEBUG: profile_key.lower().endswith('.mat') = {profile_key.lower().endswith('.mat')}")


    if is_in_specific_profiles:
        print(f"  Applying SPECIFIC sensor damage profile for '{profile_key}'...")
        profile = SPECIFIC_SENSOR_DAMAGE_PROFILES[profile_key]
        profile_healthy_score = profile.get("target_healthy_score", DEFAULT_HEALTHY_GT_SCORE)
        profile_damaged_score = profile.get("target_damaged_score", SPECIFIC_PRIMARY_DAMAGED_GT_SCORE_RANGE[1])
        
        damaged_locations[:] = profile_healthy_score
        for sensor_idx in profile["damaged_indices"]:
            if 0 <= sensor_idx < SENSORS:
                damaged_locations[:, sensor_idx] = profile_damaged_score
            else:
                print(f"    Warning: Sensor index {sensor_idx} in profile for '{profile_key}' is out of bounds (0-{SENSORS-1}).")
        if profile.get("damaged_indices"):
             most_affected_sensor_idx_for_metadata = profile["damaged_indices"][0]
        ground_truth_method_applied = "SPECIFIC_PROFILE"
    
    elif is_generic_d_mat_file: # Use the pre-calculated boolean
        print(f"  Applying 'ALL SENSORS DAMAGED' ground truth for generic damaged file: '{profile_key}'...")
        damaged_locations[:] = ALL_SENSORS_DAMAGED_GT_SCORE
        most_affected_sensor_idx_for_metadata = 0 
        ground_truth_method_applied = "ALL_SENSORS_DAMAGED_FOR_D#.MAT"
        
    else: 
        # This print statement was the one appearing in your output
        print(f"  No specific profile for '{profile_key}' and not a generic D*.mat. Using VARIANCE-BASED fallback.")
        damaged_variances = np.var(damaged_sensor_data, axis=0)
        most_affected_sensor_idx = np.argmax(damaged_variances)
        most_affected_sensor_idx_for_metadata = most_affected_sensor_idx

        if 0 <= most_affected_sensor_idx < SENSORS:
            damaged_locations[:, most_affected_sensor_idx] = np.random.uniform(
                SPECIFIC_PRIMARY_DAMAGED_GT_SCORE_RANGE[0], 
                SPECIFIC_PRIMARY_DAMAGED_GT_SCORE_RANGE[1],
                len(damaged_windows)
            )
        
        secondary_options = [i for i in range(SENSORS) if i != most_affected_sensor_idx]
        if secondary_options:
            num_secondary_to_pick = np.random.randint(1, min(3, len(secondary_options) + 1)) 
            variance_secondary_sensors = np.random.choice(
                secondary_options, size=num_secondary_to_pick, replace=False
            )
            for sensor_idx in variance_secondary_sensors:
                 if 0 <= sensor_idx < SENSORS:
                    damaged_locations[:, sensor_idx] = np.random.uniform(
                        SPECIFIC_SECONDARY_DAMAGED_GT_SCORE_RANGE[0],
                        SPECIFIC_SECONDARY_DAMAGED_GT_SCORE_RANGE[1],
                        len(damaged_windows)
                    )
        ground_truth_method_applied = "VARIANCE_BASED_FALLBACK"
    # --- End of Ground Truth Logic ---
    
    current_data_healthy = healthy_windows
    current_data_damaged = damaged_windows

    if APPLY_FFT_FEATURES:
        current_data_healthy = extract_fft_features(current_data_healthy)
        current_data_damaged = extract_fft_features(current_data_damaged)
    if APPLY_STATISTICAL_FEATURES:
        current_data_healthy = extract_statistical_features(current_data_healthy)
        current_data_damaged = extract_statistical_features(current_data_damaged)

    all_data = np.concatenate([current_data_healthy, current_data_damaged])
    all_binary_labels = np.concatenate([binary_healthy_labels, binary_damaged_labels])
    all_locations = np.concatenate([healthy_locations, damaged_locations])

    label_noise_rate = 0.05 
    noise_mask = np.random.rand(len(all_binary_labels)) < label_noise_rate
    all_binary_labels[noise_mask] = 1 - all_binary_labels[noise_mask]
            
    idx = np.random.permutation(len(all_data))
    all_data, all_binary_labels, all_locations = all_data[idx], all_binary_labels[idx], all_locations[idx]
    
    all_data_normalized, local_mean, local_std = normalize_data(all_data)

    X_train, X_temp, y_train_binary, y_temp_binary, loc_train, loc_temp = train_test_split(
        all_data_normalized, all_binary_labels, all_locations,
        test_size=(1.0 - TRAIN_RATIO),
        random_state=RANDOM_STATE,
        stratify=all_binary_labels
    )

    relative_test_size_for_temp = 1.0 - VALIDATION_RATIO_OF_REMAINING
    can_stratify_val_test = len(np.unique(y_temp_binary)) >= 2 if len(y_temp_binary) > 0 else False

    if X_temp.shape[0] == 0:
        print(f"Warning: Temporary dataset for val/test split is empty for client {client_name}. No val/test data generated.")
        X_val, X_test, y_val_binary, y_test_binary, loc_val, loc_test = \
            np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    elif can_stratify_val_test:
        X_val, X_test, y_val_binary, y_test_binary, loc_val, loc_test = train_test_split(
            X_temp, y_temp_binary, loc_temp,
            test_size=relative_test_size_for_temp, 
            random_state=RANDOM_STATE,
            stratify=y_temp_binary
        )
    else:
         X_val, X_test, y_val_binary, y_test_binary, loc_val, loc_test = train_test_split(
            X_temp, y_temp_binary, loc_temp,
            test_size=relative_test_size_for_temp,
            random_state=RANDOM_STATE
        )
         if not can_stratify_val_test and len(y_temp_binary) > 0:
            print(f"  Warning: Not enough class diversity in temporary data for client {client_name} to stratify val/test split. Unique labels: {np.unique(y_temp_binary)}")

    client_dir = os.path.join(OUTPUT_PATH, client_name)
    os.makedirs(client_dir, exist_ok=True)
    
    np.save(os.path.join(client_dir, "train_features.npy"), X_train)
    np.save(os.path.join(client_dir, "train_labels.npy"), y_train_binary)
    np.save(os.path.join(client_dir, "train_locations.npy"), loc_train)

    np.save(os.path.join(client_dir, "val_features.npy"), X_val)
    np.save(os.path.join(client_dir, "val_labels.npy"), y_val_binary)
    np.save(os.path.join(client_dir, "val_locations.npy"), loc_val)

    np.save(os.path.join(client_dir, "test_features.npy"), X_test)
    np.save(os.path.join(client_dir, "test_labels.npy"), y_test_binary)
    np.save(os.path.join(client_dir, "test_locations.npy"), loc_test)
    
    metadata = {
        "client_name": client_name,
        "healthy_file_source": healthy_file,
        "damaged_file_source": profile_key,
        "window_size": WINDOW_SIZE, "overlap": OVERLAP, "random_state_seed": RANDOM_STATE,
        "train_ratio_config": TRAIN_RATIO, "validation_ratio_of_remaining_config": VALIDATION_RATIO_OF_REMAINING,
        "local_mean": local_mean, "local_std": local_std,
        "label_type": "binary_healthy_damaged_with_sensor_locations",
        "ground_truth_method_for_damaged_locations": ground_truth_method_applied,
        "most_affected_sensor_idx_metadata_ref": most_affected_sensor_idx_for_metadata,
        "num_healthy_windows_original_file": len(healthy_windows),
        "num_damaged_windows_original_file": len(damaged_windows),
        "label_noise_rate_applied": label_noise_rate,
        "operational_noise_level_applied": operational_noise_level,
        "train_samples_count": len(X_train), "val_samples_count": len(X_val), "test_samples_count": len(X_test),
        "num_channels_in_features": X_train.shape[-1] if X_train.ndim == 3 and len(X_train) > 0 else (SENSORS if all_data_normalized.ndim ==3 and all_data_normalized.size > 0 else "N/A"),
        "feature_data_shape_example (train)": X_train.shape if len(X_train) > 0 else "N/A",
        "SENSORS_config_value": SENSORS, "SENSOR_NAMES_ORDERED_config_value": SENSOR_NAMES_ORDERED,
        "apply_fft_features_config": APPLY_FFT_FEATURES, "apply_statistical_features_config": APPLY_STATISTICAL_FEATURES
    }
    np.savez(os.path.join(client_dir, "metadata.npz"), **metadata)
    
    print(f"  Finished processing for {client_name}. Data saved to '{client_dir}'")
    return {
        'client_name': client_name, 'train_size': len(X_train), 'val_size': len(X_val), 'test_size': len(X_test),
        'num_input_dimensions': X_train.shape[1:] if len(X_train) > 0 else (all_data_normalized.shape[1:] if all_data_normalized.size > 0 else "N/A"),
        'healthy_ratio_in_original_files': len(healthy_windows) / (len(healthy_windows) + len(damaged_windows)) if (len(healthy_windows) + len(damaged_windows)) > 0 else 0,
    }

def prepare_client_data_wrapper(healthy_file, damaged_file, client_name):
    try:
        print(f"\n--- Starting preprocessing for Client: {client_name} ---")
        print(f"  Healthy data source: {healthy_file}")
        print(f"  Damaged data source: {damaged_file}")
        stats = process_client(healthy_file, damaged_file, client_name)
        if stats is None:
             print(f"ERROR: Preprocessing failed for client {client_name}.")
             return None
        print(f"--- Successfully processed data for Client: {client_name} ---")
        if stats.get('train_size',0) > 0 :
            print(f"  Train samples: {stats['train_size']}, Val samples: {stats['val_size']}, Test samples: {stats['test_size']}")
            print(f"  Feature dimensions (window_length, num_channels): {stats.get('num_input_dimensions', 'N/A')}")
            print(f"  Ratio of healthy windows in original files: {stats.get('healthy_ratio_in_original_files', 0):.2f}")
        return stats
    except Exception as e:
        print(f"CRITICAL ERROR during data preparation for client {client_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def simulate_federated_preprocessing():
    print("Starting Federated Data Preprocessing Simulation...")
    if not os.path.exists(RAW_DATA_PATH):
        print(f"ERROR: Raw data path '{RAW_DATA_PATH}' does not exist. Please check config.py and your data setup.")
        return
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    CLIENT_DATA_SOURCES = {
        "Client_1": ("H1.mat", "D1.mat"),
        "Client_2": ("H2.mat", "D2.mat"),
        "Client_3": ("H3.mat", "D3.mat"),
        "Client_4": ("H4.mat", "D4.mat"),
        "Client_5": ("H5.mat", "D5.mat"),
    }

    all_clients_status_summary = {}
    for client_id_key, (h_filename, d_filename) in CLIENT_DATA_SOURCES.items():
        client_processing_stats = prepare_client_data_wrapper(h_filename, d_filename, client_id_key)
        all_clients_status_summary[client_id_key] = client_processing_stats is not None
    
    print("\n--- Overall Preprocessing Summary ---")
    successful_clients, failed_clients = [], []
    for client_name_key, was_successful in all_clients_status_summary.items():
        status_message = 'Successful' if was_successful else 'Failed'
        print(f"  Client '{client_name_key}': Preprocessing {status_message}")
        if was_successful: successful_clients.append(client_name_key)
        else: failed_clients.append(client_name_key)
    
    if failed_clients: print(f"\nWarning: Preprocessing failed for clients: {', '.join(failed_clients)}")
    if not successful_clients: print("\nERROR: Preprocessing failed for ALL clients.")
    else: print(f"\nPreprocessing completed for {len(successful_clients)} client(s). Processed data in '{OUTPUT_PATH}'.")

if __name__ == "__main__":
    essential_vars_to_check = [
        "BASE_DIR", "WINDOW_SIZE", "OVERLAP", "RANDOM_STATE", "OUTPUT_PATH",
        "SENSORS", "SENSOR_NAMES_ORDERED", "DEFAULT_HEALTHY_GT_SCORE",
        "ALL_SENSORS_DAMAGED_GT_SCORE", "SPECIFIC_SENSOR_DAMAGE_PROFILES",
        "TRAIN_RATIO", "VALIDATION_RATIO_OF_REMAINING"
    ]
    missing_vars = [var for var in essential_vars_to_check if var not in globals()]
    if missing_vars:
        print("\n--- CRITICAL CONFIGURATION ERROR ---")
        print("The following essential config variables are MISSING (not loaded from config.py):")
        for var_name in missing_vars: print(f"  - {var_name}")
        print(f"Attempted to load config from a path relative to: {project_root_path}")
        sys.exit(1)
    
    print("\n--- Preprocessing Script Configuration Info ---")
    print(f"  Base Directory: {BASE_DIR}")
    print(f"  Raw Data Path: {RAW_DATA_PATH}")
    print(f"  Output (Processed Data) Path: {OUTPUT_PATH}")
    print(f"  Window Size: {WINDOW_SIZE}, Overlap: {OVERLAP}")
    print(f"  Healthy Sensor GT Score: {DEFAULT_HEALTHY_GT_SCORE}")
    print(f"  'All Sensors Damaged' GT Score (for D*.mat): {ALL_SENSORS_DAMAGED_GT_SCORE}")
    print(f"  Using SENSOR_NAMES_ORDERED: {SENSOR_NAMES_ORDERED}")
    print("--- End Configuration Info ---\n")

    simulate_federated_preprocessing()
