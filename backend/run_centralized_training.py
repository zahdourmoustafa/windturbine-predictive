"""
run_centralized_training.py

This script orchestrates centralized training for the Gearbox Fault Detection model.
It loads preprocessed data for a specified client, initializes the model,
runs the training and validation loops, and saves the best performing model.

Key functionalities:
- Loads configuration from `config.config` and allows overrides via command-line arguments.
- Uses a custom PyTorch Dataset (`GearboxDataset`) to load preprocessed .npy files.
- Instantiates the `GearboxCNNLSTM` model.
- Calls the `train_model` function from `src.training.train` to perform training.
- Optionally evaluates the trained model on a test set.
- Saves the best trained model checkpoint.

Command-line arguments:
  --client_id: Client ID for which to load data and train (default: "Client_1").
  --epochs: Number of training epochs (overrides config).
  --batch_size: Batch size for training (overrides config).
  --lr: Learning rate (overrides config).
  --device: Device to use ('cuda' or 'cpu', overrides config/auto-detect).
  --file_limit: Limit number of samples loaded from .npy files for quick testing.
  --num_workers: Number of workers for DataLoader.
  --run_test_eval: Flag to run evaluation on the test set after training.
  --model_output_dir: Directory to save the trained model (default: output/models/centralized).
  --model_filename_suffix: Optional suffix for the saved model filename.

Example Usage:
  python run_centralized_training.py --client_id Client_Seiko --epochs 50 --lr 0.0005 --run_test_eval
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys
import argparse
import logging
from datetime import datetime

# --- Setup Project Root Path ---
# Assuming this script is in the project root directory.
# If it's moved, this path might need adjustment.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- Import Project Modules ---
try:
    from config import config as cfg
    from src.models.model import GearboxCNNLSTM
    from src.training.train import train as train_model
    from src.training.train import evaluate as evaluate_model
except ModuleNotFoundError as e:
    print(f"ERROR: Could not import project modules. Ensure script is run from project root or PYTHONPATH is set.")
    print(f"Details: {e}")
    sys.exit(1)

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class GearboxDataset(Dataset):
    """
    PyTorch Dataset for loading preprocessed gearbox data for a single client.
    Expects features, binary labels, and per-sensor anomaly location scores.
    """
    def __init__(self, client_data_path: str, dataset_type: str = 'train', file_limit: int = None):
        """
        Args:
            client_data_path (str): Path to the processed data directory for a specific client
                                     (e.g., 'data/processed/Client_1').
            dataset_type (str): Type of dataset to load ('train', 'val', or 'test').
            file_limit (int, optional): Maximum number of samples to load. Defaults to None (load all).
        """
        self.dataset_type = dataset_type
        self.features_path = os.path.join(client_data_path, f'{dataset_type}_features.npy')
        self.labels_path = os.path.join(client_data_path, f'{dataset_type}_labels.npy')
        self.locations_path = os.path.join(client_data_path, f'{dataset_type}_locations.npy')

        if not os.path.exists(self.features_path):
            raise FileNotFoundError(f"Features file not found: {self.features_path}")
        if not os.path.exists(self.labels_path):
            raise FileNotFoundError(f"Labels file not found: {self.labels_path}")
        if not os.path.exists(self.locations_path):
            raise FileNotFoundError(f"Locations file not found: {self.locations_path}")

        logger.info(f"Loading {dataset_type} features from: {self.features_path}")
        self.features = np.load(self.features_path)
        logger.info(f"Loading {dataset_type} labels from: {self.labels_path}")
        self.labels = np.load(self.labels_path)
        logger.info(f"Loading {dataset_type} locations from: {self.locations_path}")
        self.locations = np.load(self.locations_path)

        if file_limit is not None and file_limit > 0:
            logger.info(f"Limiting {dataset_type} dataset to first {file_limit} samples.")
            self.features = self.features[:file_limit]
            self.labels = self.labels[:file_limit]
            self.locations = self.locations[:file_limit]

        if not (len(self.features) == len(self.labels) == len(self.locations)):
            msg = (f"Mismatch in number of samples for {dataset_type} data in {client_data_path}: "
                   f"Features={len(self.features)}, Labels={len(self.labels)}, Locations={len(self.locations)}")
            logger.error(msg)
            raise ValueError(msg)
        
        if len(self.features) == 0:
            logger.warning(f"Loaded empty {dataset_type} dataset for client path: {client_data_path}")


        logger.info(f"Successfully loaded {dataset_type} data for client: {os.path.basename(client_data_path)}")
        logger.info(f"  {dataset_type} Features shape: {self.features.shape}")
        logger.info(f"  {dataset_type} Labels shape: {self.labels.shape}, Unique labels: {np.unique(self.labels)}")
        logger.info(f"  {dataset_type} Locations shape: {self.locations.shape}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        """
        Returns a single sample: features, binary label, and sensor location scores.
        Features are expected to be (window_length, num_channels) by the model.
        preprocess.py saves them as (num_samples, window_length, num_channels).
        """
        features_sample = torch.from_numpy(self.features[idx]).float()
        # Ensure label is float for BCEWithLogitsLoss and has a singleton dimension if needed by loss
        label_sample = torch.tensor(self.labels[idx], dtype=torch.float)
        locations_sample = torch.from_numpy(self.locations[idx]).float()
        
        return features_sample, label_sample, locations_sample


class TrainingConfig:
    """
    Consolidates all configuration parameters required for the training script (`src.training.train.train`).
    Prioritizes command-line arguments, then `config.py` values, then hardcoded defaults.
    """
    def __init__(self, args, base_cfg):
        # Core training parameters
        self.device = args.device if args.device else getattr(base_cfg, 'DEVICE', ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.num_epochs = args.epochs if args.epochs is not None else getattr(base_cfg, 'EPOCHS', 20)
        self.learning_rate = args.lr if args.lr is not None else getattr(base_cfg, 'LR', 0.001)
        self.LR = self.learning_rate # Alias for compatibility if train.py uses LR

        # Parameters from config.py that train.py expects, with fallbacks
        self.weight_decay = getattr(base_cfg, 'WEIGHT_DECAY', 1e-4)
        self.scheduler_patience = getattr(base_cfg, 'scheduler_patience', 5)
        self.use_augmentation = getattr(base_cfg, 'use_augmentation', True)
        self.gradient_clip_val = getattr(base_cfg, 'gradient_clip_val', 1.0)
        
        # Loss function specific parameters
        self.LOCATION_LOSS_WEIGHT = getattr(base_cfg, 'LOCATION_LOSS_WEIGHT', 1.0)
        self.FOCAL_GAMMA = getattr(base_cfg, 'FOCAL_GAMMA', 2.0)
        self.POS_WEIGHT = getattr(base_cfg, 'POS_WEIGHT', 3.0)
        self.label_smoothing = getattr(base_cfg, 'label_smoothing', 0.05)

        # Thresholds and metrics related
        self.DEFAULT_THRESHOLD = getattr(base_cfg, 'DEFAULT_THRESHOLD', 0.5)
        self.MIN_RECALL = getattr(base_cfg, 'MIN_RECALL', 0.5)
        
        # Early stopping
        self.early_stopping_patience = getattr(base_cfg, 'early_stopping_patience', 10)
        self.metric_for_best_model = getattr(base_cfg, 'metric_for_best_model', 'f1_score') # e.g., 'f1_score' or 'val_loss'

        # Model specific (though model is instantiated outside, train.py might use these for context)
        self.SENSORS = getattr(base_cfg, 'SENSORS', 8)


def main(args):
    """
    Main function to orchestrate centralized training.
    """
    start_time = datetime.now()
    logger.info(f"Starting centralized training run for client: {args.client_id} at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # --- Configuration Setup ---
    logger.info("Loading and consolidating configuration...")
    training_config = TrainingConfig(args, cfg) # Pass base_cfg (config.py module)
    
    # Log key configurations
    logger.info(f"  Device: {training_config.device}")
    logger.info(f"  Client ID for training: {args.client_id}")
    logger.info(f"  Number of Epochs: {training_config.num_epochs}")
    logger.info(f"  Batch Size (from args): {args.batch_size}") # Batch size is used directly in DataLoader
    logger.info(f"  Learning Rate: {training_config.learning_rate}")
    logger.info(f"  Model output directory: {args.model_output_dir}")

    # Validate essential paths from config.py
    if not hasattr(cfg, 'OUTPUT_PATH') or not cfg.OUTPUT_PATH:
        logger.error("cfg.OUTPUT_PATH (base for processed data) is not defined in your config/config.py.")
        sys.exit(1)
    base_processed_data_path = cfg.OUTPUT_PATH
    
    client_data_path = os.path.join(base_processed_data_path, args.client_id)
    if not os.path.isdir(client_data_path):
        logger.error(f"Processed data directory not found for client '{args.client_id}' at '{client_data_path}'.")
        logger.error("Please ensure preprocess.py has been run successfully and cfg.OUTPUT_PATH is correct.")
        sys.exit(1)

    # --- Datasets and DataLoaders ---
    logger.info("Setting up datasets and dataloaders...")
    try:
        train_dataset = GearboxDataset(client_data_path, dataset_type='train', file_limit=args.file_limit)
        val_dataset = GearboxDataset(client_data_path, dataset_type='val', file_limit=args.file_limit)
    except FileNotFoundError as e:
        logger.error(f"Error initializing datasets: {e}")
        sys.exit(1)
    except ValueError as e: # Catch mismatch in samples
        logger.error(f"Error initializing datasets: {e}")
        sys.exit(1)

    if len(train_dataset) == 0:
        logger.error(f"Training dataset for client {args.client_id} is empty. Cannot proceed with training.")
        sys.exit(1)
    if len(val_dataset) == 0:
        logger.warning(f"Validation dataset for client {args.client_id} is empty. Training will proceed without validation metrics for scheduler/early stopping if they rely on it.")
        # train_model function should handle empty val_loader gracefully if possible, or this needs to be a hard stop.
        # For now, we'll let it proceed, but train_model should be robust.

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, # Use batch_size from args directly
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=True if training_config.device == 'cuda' else False,
        drop_last=True if len(train_dataset) > args.batch_size else False # Drop last if not a full batch
    )
    # For validation, shuffle is False and drop_last is usually False
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=True if training_config.device == 'cuda' else False
    ) if len(val_dataset) > 0 else None # Handle empty validation set

    # --- Model Initialization ---
    logger.info("Initializing model...")
    # Ensure these parameters are available in cfg from config.py
    window_size_cfg = getattr(cfg, 'WINDOW_SIZE', 128) # Example default
    num_sensors_cfg = getattr(cfg, 'SENSORS', 8)
    lstm_hidden_size_cfg = getattr(cfg, 'LSTM_HIDDEN_SIZE', 32)
    num_lstm_layers_cfg = getattr(cfg, 'NUM_LSTM_LAYERS', 1)
    dropout_rate_cfg = getattr(cfg, 'DROPOUT_RATE', 0.3)

    model = GearboxCNNLSTM(
        window_size=window_size_cfg, # Model uses adaptive pooling, so exact window_size less critical here
        lstm_hidden_size=lstm_hidden_size_cfg,
        num_lstm_layers=num_lstm_layers_cfg,
        num_sensors=num_sensors_cfg,
        dropout_rate=dropout_rate_cfg
    ).to(training_config.device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model '{model.__class__.__name__}' initialized with {total_params:,} trainable parameters.")

    # --- Training ---
    logger.info("Starting model training...")
    trained_model = train_model(model, train_loader, val_loader, training_config) # Pass the config object
    logger.info("Training finished.")

    # --- Save the best model ---
    os.makedirs(args.model_output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"centralized_model_{args.client_id}{args.model_filename_suffix}_{timestamp}.pth"
    model_save_path = os.path.join(args.model_output_dir, model_filename)
    
    try:
        torch.save(trained_model.state_dict(), model_save_path)
        logger.info(f"Best trained model saved to: {model_save_path}")
    except Exception as e:
        logger.error(f"Error saving model to {model_save_path}: {e}")

    # --- Final Evaluation on Test Set (Optional) ---
    if args.run_test_eval:
        logger.info("Starting final evaluation on TEST set...")
        try:
            test_dataset = GearboxDataset(client_data_path, dataset_type='test', file_limit=args.file_limit)
            if len(test_dataset) > 0:
                test_loader = DataLoader(
                    test_dataset, 
                    batch_size=args.batch_size, 
                    shuffle=False, 
                    num_workers=args.num_workers,
                    pin_memory=True if training_config.device == 'cuda' else False
                )
                logger.info("Evaluating trained model on test set...")
                # The evaluate_model function in src.training.train now takes num_sensors_config
                # It should get this from the model or config. Let's ensure it's passed if needed.
                # The evaluate function in train.py uses num_sensors_config from its config object if available.
                # We pass training_config which has SENSORS attribute.
                _ = evaluate_model(trained_model, test_loader, device=training_config.device, num_sensors_config=training_config.SENSORS)
                # Evaluation summary is printed within evaluate_model
            else:
                logger.warning(f"Test dataset for client {args.client_id} is empty. Skipping test set evaluation.")
        except FileNotFoundError:
            logger.warning(f"Test data not found for client {args.client_id} at {client_data_path}. Skipping test set evaluation.")
        except Exception as e:
            logger.error(f"Error during test set evaluation: {e}", exc_info=True)
    
    end_time = datetime.now()
    logger.info(f"Centralized training script finished at {end_time.strftime('%Y%m%d %H:%M:%S')}. Total duration: {end_time - start_time}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run centralized training for Gearbox Fault Detection.")
    
    # Data and Client Arguments
    parser.add_argument('--client_id', type=str, default="Client_1", 
                        help="Client ID to use for training and validation (e.g., Client_1). Default: Client_1")
    parser.add_argument('--file_limit', type=int, default=None, 
                        help="Limit number of samples loaded from .npy files for quick testing. Default: None (load all).")

    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=None, 
                        help="Number of training epochs (overrides config.py).")
    parser.add_argument('--batch_size', type=int, default=getattr(cfg, 'BATCH_SIZE', 32),
                        help=f"Batch size for training. Default from config.py or 32.")
    parser.add_argument('--lr', type=float, default=None, 
                        help="Learning rate (overrides config.py).")

    # System and Execution Arguments
    parser.add_argument('--device', type=str, default=None, 
                        help="Device to use ('cuda' or 'cpu', overrides config.py/auto-detect).")
    parser.add_argument('--num_workers', type=int, default=0, 
                        help="Number of workers for DataLoader. Default: 0 (main process).")
    parser.add_argument('--run_test_eval', action='store_true', 
                        help="Run evaluation on the test set after training.")
    
    # Output Arguments
    default_model_dir = os.path.join(PROJECT_ROOT, "output", "models", "centralized")
    parser.add_argument('--model_output_dir', type=str, default=default_model_dir,
                        help=f"Directory to save the trained model. Default: {default_model_dir}")
    parser.add_argument('--model_filename_suffix', type=str, default="",
                        help="Optional suffix for the saved model filename (e.g., '_run1'). Default: empty string.")
    
    args = parser.parse_args()

    # Ensure model output directory exists
    try:
        os.makedirs(args.model_output_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Could not create model output directory: {args.model_output_dir}. Error: {e}")
        sys.exit(1)
        
    # Add a file handler for logging to a file in the model_output_dir
    log_file_path = os.path.join(args.model_output_dir, f"training_log_{args.client_id}{args.model_filename_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)

    logger.info("Parsed command-line arguments:")
    for arg, value in sorted(vars(args).items()):
        logger.info(f"  {arg}: {value}")

    main(args)
