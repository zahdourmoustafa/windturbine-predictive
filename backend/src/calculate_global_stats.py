import os
import numpy as np
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from config.config import OUTPUT_PATH, GLOBAL_STATS_PATH, SENSORS
except ImportError:
    print("Error: Could not import OUTPUT_PATH, GLOBAL_STATS_PATH, or SENSORS from config.config.")
    print("Please ensure config/config.py is correctly set up and accessible.")
    print(f"Attempted to add {project_root} to sys.path")
    sys.exit(1)

# Ensure CLIENT_NAMES is defined, e.g., from config or directly here
# This should list all clients for whom metadata exists.
# You might want to make this dynamic by listing directories in OUTPUT_PATH
# or define it consistently in config.py
CLIENT_NAMES = ["Client_1", "Client_2", "Client_3", "Client_4", "Client_5"]
# If Client_Seiko was processed by preprocess.py, add it here:
# CLIENT_NAMES.append("Client_Seiko")


def calculate_and_save_global_stats():
    """
    Calculates global mean and standard deviation from all client metadata
    and saves them to a file. Prioritizes original window counts for weighting.
    """
    total_samples_all_clients = 0
    # Initialize with expected shape (1, 1, SENSORS) for broadcasting
    # This helps if the first client processed has issues or 0 samples.
    num_features = SENSORS # From config.py
    
    weighted_sum_of_means = np.zeros((1, 1, num_features))
    weighted_sum_of_squared_means_plus_variances = np.zeros((1, 1, num_features))
    
    found_valid_client_data = False

    print(f"Starting calculation of global statistics using data from: {OUTPUT_PATH}")
    print(f"Looking for clients: {CLIENT_NAMES}")

    for client_name in CLIENT_NAMES:
        metadata_file_path = os.path.join(OUTPUT_PATH, client_name, "metadata.npz")
        print(f"\nProcessing client: {client_name}")

        if not os.path.exists(metadata_file_path):
            print(f"  Warning: Metadata file not found for {client_name} at {metadata_file_path}. Skipping.")
            continue

        try:
            metadata = np.load(metadata_file_path, allow_pickle=True)
            print(f"  Found metadata keys: {list(metadata.keys())}")
            
            local_mean = metadata['local_mean']
            local_std = metadata['local_std']
            
            N_client = 0
            # Primary method: Use original window counts before splitting
            if 'num_healthy_windows_original_file' in metadata and \
               'num_damaged_windows_original_file' in metadata:
                num_healthy = int(metadata['num_healthy_windows_original_file'])
                num_damaged = int(metadata['num_damaged_windows_original_file'])
                N_client = num_healthy + num_damaged
                print(f"  Using original window counts: Healthy={num_healthy}, Damaged={num_damaged}, N_client={N_client}")
            else:
                print(f"  Warning: Keys 'num_healthy_windows_original_file' or 'num_damaged_windows_original_file' not found for {client_name}.")
                # Fallback 1: Slightly older original window count keys
                if 'num_healthy_windows_original' in metadata and \
                   'num_damaged_windows_original' in metadata:
                    num_healthy = int(metadata['num_healthy_windows_original'])
                    num_damaged = int(metadata['num_damaged_windows_original'])
                    N_client = num_healthy + num_damaged
                    print(f"  Using older original window counts: Healthy={num_healthy}, Damaged={num_damaged}, N_client={N_client}")
                else:
                    print(f"  Warning: Older original window count keys also not found for {client_name}.")
                    # Fallback 2: Use total split counts if original counts are entirely missing
                    if 'train_samples_count' in metadata and \
                       'val_samples_count' in metadata and \
                       'test_samples_count' in metadata:
                        train_c = int(metadata['train_samples_count'])
                        val_c = int(metadata['val_samples_count'])
                        test_c = int(metadata['test_samples_count'])
                        N_client = train_c + val_c + test_c
                        print(f"  Approximating N_client with total split counts: Train={train_c}, Val={val_c}, Test={test_c}, N_client={N_client}")
                    else:
                        print(f"  Error: No usable sample count keys found in metadata for {client_name}.")


            if N_client <= 0: # Check for strictly positive N_client
                print(f"  Warning: Client {client_name} has {N_client} effective samples based on metadata. Skipping this client for global stats.")
                continue
            
            # Ensure local_mean and local_std have the correct shape (1, 1, num_features)
            # This was handled by normalize_data in preprocess.py, but good to double check.
            if local_mean.shape[-1] != num_features or local_std.shape[-1] != num_features:
                print(f"  Error: Shape mismatch for local_mean/std for {client_name}. Expected {num_features} features.")
                print(f"         local_mean shape: {local_mean.shape}, local_std shape: {local_std.shape}")
                continue

            print(f"  Client {client_name}: N_client={N_client}. Mean shape: {local_mean.shape}, Std shape: {local_std.shape}")

            # Welford's algorithm component for combining stats:
            # new_mean = old_mean + (delta * N_new) / N_total
            # new_M2 = old_M2 + delta^2 * N_old * N_new / N_total
            # For pooled variance: GlobalVar = sum( (N_i-1)*Var_i + N_i*Mean_i^2 ) / sum(N_i) - GlobalMean^2
            # Or more directly: E[X^2] - (E[X])^2
            # E[X] = sum(N_i * Mean_i) / sum(N_i)
            # E[X^2] = sum(N_i * (Var_i + Mean_i^2)) / sum(N_i)
            
            weighted_sum_of_means += N_client * local_mean
            local_variance = local_std**2
            weighted_sum_of_squared_means_plus_variances += N_client * (local_variance + local_mean**2)
            
            total_samples_all_clients += N_client
            found_valid_client_data = True

        except KeyError as e:
            print(f"  Error processing metadata for {client_name}: Missing key {e}. Skipping.")
            continue
        except Exception as e:
            print(f"  An unexpected error occurred processing {client_name}: {e}. Skipping.")
            continue

    if not found_valid_client_data or total_samples_all_clients == 0:
        print("\nError: No valid client data with samples found. Cannot calculate global statistics.")
        return

    global_mean = weighted_sum_of_means / total_samples_all_clients
    
    global_e_x_squared = weighted_sum_of_squared_means_plus_variances / total_samples_all_clients
    global_variance = global_e_x_squared - (global_mean**2)
    
    # Ensure variance is not negative due to floating point inaccuracies for near-zero variances
    global_variance[global_variance < 0] = 0
    global_std = np.sqrt(global_variance) # Add epsilon inside sqrt if needed, but local_std already had it.
    # To be safe, if any global_std is zero (or very small), add epsilon before division elsewhere.
    global_std[global_std < 1e-8] = 1e-8


    print(f"\n--- Global Statistics Calculated ---")
    print(f"Total original windows processed from valid clients: {total_samples_all_clients}")
    print(f"Global Mean shape: {global_mean.shape}")
    # print(f"Global Mean values (first 3 features): {global_mean[0,0,:3]}") # Example print
    print(f"Global Std shape: {global_std.shape}")
    # print(f"Global Std values (first 3 features): {global_std[0,0,:3]}") # Example print

    os.makedirs(GLOBAL_STATS_PATH, exist_ok=True)
    save_path = os.path.join(GLOBAL_STATS_PATH, "global_stats.npz")
    np.savez(save_path, global_mean=global_mean, global_std=global_std)
    print(f"\nSuccessfully saved global statistics to: {save_path}")

if __name__ == "__main__":
    # Ensure config variables are loaded for SENSORS
    if 'SENSORS' not in globals():
        print("Error: SENSORS not defined. Check config import.")
        sys.exit(1)
    calculate_and_save_global_stats()
