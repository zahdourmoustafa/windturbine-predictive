import torch
import torch.optim as optim
import numpy as np
import os
import sys
import logging
from torch.utils.data import DataLoader, TensorDataset

# --- Setup Project Root Path ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- Import Project Modules & Config ---
try:
    from config.config import (
        OUTPUT_PATH as PROCESSED_DATA_PATH,
        SENSORS as CFG_SENSORS, WINDOW_SIZE as CFG_WINDOW_SIZE,
        LSTM_HIDDEN_SIZE, NUM_LSTM_LAYERS, DROPOUT_RATE,
        BATCH_SIZE as DEFAULT_BATCH_SIZE, # Used for client's DataLoader
        LR as DEFAULT_LR, # Default LR for local training if not overridden
        LOCAL_EPOCHS_FL, # Specific for FL local rounds
        WEIGHT_DECAY as DEFAULT_WEIGHT_DECAY,
        LOCATION_LOSS_WEIGHT, FOCAL_GAMMA, POS_WEIGHT, label_smoothing,
        gradient_clip_val, use_augmentation,
        # Params for LocalFLTrainingConfig, to be passed to src.training.train.train
        LOCAL_FL_SCHEDULER_PATIENCE,
        LOCAL_FL_METRIC_BEST_MODEL,
        LOCAL_FL_EARLY_STOP_PATIENCE,
        MIN_RECALL, DEFAULT_THRESHOLD, CLIENT_DATALOADER_NUM_WORKERS
    )
    from src.models.model import GearboxCNNLSTM
    from src.training.train import train as train_local_epochs_fn
    from src.training.train import evaluate as evaluate_model_on_dataset, FaultLocalizationLoss
except ModuleNotFoundError as e:
    print(f"ERROR: Could not import project modules in client.py. Details: {e}")
    sys.exit(1)
except ImportError as e:
    print(f"ERROR: Could not import specific names in client.py. Details: {e}")
    sys.exit(1)

logger = logging.getLogger(__name__)

class FederatedClient:
    def __init__(self, client_id: str, device: torch.device):
        self.client_id = client_id
        self.device = device
        self.model = GearboxCNNLSTM(
            window_size=CFG_WINDOW_SIZE, lstm_hidden_size=LSTM_HIDDEN_SIZE,
            num_lstm_layers=NUM_LSTM_LAYERS, num_sensors=CFG_SENSORS,
            dropout_rate=DROPOUT_RATE
        ).to(self.device)
        
        self.train_loader = None
        self.val_loader = None
        self.num_train_samples = 0
        self.num_val_samples = 0
        self.data_quality_metric = np.random.uniform(0.8, 1.0)

        logger.info(f"FederatedClient '{self.client_id}' initialized on device '{self.device}'.")
        self._load_local_data()

    def _load_local_data(self):
        client_data_path = os.path.join(PROCESSED_DATA_PATH, self.client_id)
        logger.info(f"Client '{self.client_id}': Loading data from {client_data_path}")
        try:
            X_train = np.load(os.path.join(client_data_path, "train_features.npy"))
            y_train_binary = np.load(os.path.join(client_data_path, "train_labels.npy"))
            loc_train_gt = np.load(os.path.join(client_data_path, "train_locations.npy"))

            X_val = np.load(os.path.join(client_data_path, "val_features.npy"))
            y_val_binary = np.load(os.path.join(client_data_path, "val_labels.npy"))
            loc_val_gt = np.load(os.path.join(client_data_path, "val_locations.npy"))

            self.num_train_samples = len(X_train)
            self.num_val_samples = len(X_val)

            if self.num_train_samples == 0:
                logger.warning(f"Client '{self.client_id}': No training samples found. Local training will be skipped.")
                return

            train_dataset = TensorDataset(
                torch.from_numpy(X_train).float(), torch.from_numpy(y_train_binary).float(), torch.from_numpy(loc_train_gt).float())
            self.train_loader = DataLoader(train_dataset, batch_size=DEFAULT_BATCH_SIZE, shuffle=True, num_workers=CLIENT_DATALOADER_NUM_WORKERS)

            if self.num_val_samples > 0:
                val_dataset = TensorDataset(
                    torch.from_numpy(X_val).float(), torch.from_numpy(y_val_binary).float(), torch.from_numpy(loc_val_gt).float())
                self.val_loader = DataLoader(val_dataset, batch_size=DEFAULT_BATCH_SIZE, shuffle=False, num_workers=CLIENT_DATALOADER_NUM_WORKERS)
            else:
                logger.warning(f"Client '{self.client_id}': No validation samples found. Local validation metrics will be limited.")
            logger.info(f"Client '{self.client_id}': Loaded {self.num_train_samples} train and {self.num_val_samples} val samples.")
        except FileNotFoundError as e:
            logger.error(f"Client '{self.client_id}': Data file not found: {e}. Ensure preprocess.py was run for this client ID.")
            self.num_train_samples, self.num_val_samples = 0, 0
        except Exception as e:
            logger.error(f"Client '{self.client_id}': Critical error loading data: {e}", exc_info=True)
            self.num_train_samples, self.num_val_samples = 0, 0

    def receive_model_update(self, global_model_state_dict):
        try:
            self.model.load_state_dict(global_model_state_dict)
            logger.debug(f"Client '{self.client_id}': Local model updated with global weights.")
        except RuntimeError as e:
            logger.error(f"Client '{self.client_id}': Error loading state_dict. Model architecture mismatch? Details: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Client '{self.client_id}': Unexpected error receiving model update: {e}", exc_info=True)

    def train_local_model(self, local_epochs_override=None, local_lr_override=None):
        if self.train_loader is None or self.num_train_samples == 0:
            logger.warning(f"Client '{self.client_id}': No training data. Skipping local training.")
            # Return structure expected by server even if no training
            return None, {'val_loss': float('inf'), 'f1_score': 0.0, 'sensor_anomaly_mse': float('inf'), 'detection_auc': 0.0, 'data_quality': 0.0}, 0

        current_local_epochs = local_epochs_override if local_epochs_override is not None else LOCAL_EPOCHS_FL
        current_local_lr = local_lr_override if local_lr_override is not None else DEFAULT_LR
        
        logger.info(f"Client '{self.client_id}': Starting local training for {current_local_epochs} epochs with LR={current_local_lr}.")

        class LocalFLTrainingConfig:
            def __init__(self, client_device):
                self.device = client_device
                self.num_epochs = current_local_epochs
                self.learning_rate = current_local_lr
                self.LR = current_local_lr 
                self.weight_decay = DEFAULT_WEIGHT_DECAY
                self.scheduler_patience = LOCAL_FL_SCHEDULER_PATIENCE
                self.use_augmentation = use_augmentation
                self.gradient_clip_val = gradient_clip_val
                self.LOCATION_LOSS_WEIGHT = LOCATION_LOSS_WEIGHT
                self.FOCAL_GAMMA = FOCAL_GAMMA
                self.POS_WEIGHT = POS_WEIGHT
                self.label_smoothing = label_smoothing
                self.DEFAULT_THRESHOLD = DEFAULT_THRESHOLD
                self.MIN_RECALL = MIN_RECALL
                self.early_stopping_patience = LOCAL_FL_EARLY_STOP_PATIENCE
                self.metric_for_best_model = LOCAL_FL_METRIC_BEST_MODEL
                self.SENSORS = CFG_SENSORS
        
        local_config = LocalFLTrainingConfig(self.device)

        try:
            self.model = train_local_epochs_fn(
                model=self.model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                config_obj=local_config
            )
            
            final_local_val_metrics = {}
            if self.val_loader:
                logger.debug(f"Client '{self.client_id}': Performing final local validation on the best locally trained model.")
                default_criterion = FaultLocalizationLoss(
                    alpha=LOCATION_LOSS_WEIGHT, focal_gamma=FOCAL_GAMMA, pos_weight_val=POS_WEIGHT, label_smoothing=label_smoothing
                )
                # Ensure SENSOR_NAMES_ORDERED is available for print_evaluation_summary if called within evaluate_model_on_dataset
                final_local_val_metrics = evaluate_model_on_dataset(
                    model=self.model, val_loader=self.val_loader, criterion=default_criterion,
                    device=self.device, num_sensors_config=CFG_SENSORS
                )
            else:
                logger.warning(f"Client '{self.client_id}': No validation data for final local metrics. Reporting basic structure.")
                final_local_val_metrics = {'val_loss': float(0), 'f1_score': 0.0, 'sensor_anomaly_mse': float(0), 'detection_auc': 0.0}

            logger.info(f"Client '{self.client_id}': Local training complete. Best Local Val F1: {final_local_val_metrics.get('f1_score',0.0):.4f}")
            final_local_val_metrics['data_quality'] = self.data_quality_metric
            return self.model.state_dict(), final_local_val_metrics, self.num_train_samples
            
        except Exception as e:
            logger.error(f"Client '{self.client_id}': Error during local model training: {e}", exc_info=True)
            return None, {'val_loss': float('inf'), 'f1_score': 0.0, 'sensor_anomaly_mse': float('inf'), 'detection_auc': 0.0, 'data_quality': 0.0}, 0
