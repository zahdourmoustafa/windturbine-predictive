import torch
import torch.nn as nn
import numpy as np
import os
import sys
import logging
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset

# --- Setup Project Root Path ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- Import Project Modules & Config ---
try:
    from config.config import (
        BASE_DIR, DEVICE as CFG_DEVICE, SENSORS as CFG_SENSORS,
        WINDOW_SIZE as CFG_WINDOW_SIZE, LSTM_HIDDEN_SIZE, NUM_LSTM_LAYERS, DROPOUT_RATE,
        LOCATION_LOSS_WEIGHT, FOCAL_GAMMA, POS_WEIGHT, label_smoothing, # For default criterion
        BATCH_SIZE as DEFAULT_EVAL_BATCH_SIZE, SENSOR_NAMES_ORDERED # For evaluation
    )
    from src.models.model import GearboxCNNLSTM
    from src.api.client import FederatedClient
    from src.training.train import evaluate as evaluate_model_on_dataset
    from src.training.train import FaultLocalizationLoss
except ModuleNotFoundError as e:
    print(f"ERROR: Could not import project modules in server.py. Details: {e}")
    sys.exit(1)
except ImportError as e:
    print(f"ERROR: Could not import specific names in server.py. Details: {e}")
    sys.exit(1)

logger = logging.getLogger(__name__)

class FederatedServer:
    def __init__(self, client_ids: list):
        self.device = torch.device(CFG_DEVICE)
        self.client_ids_all_potential = client_ids 
        self.clients = [] 
        
        self.global_model_save_dir = os.path.join(BASE_DIR, "models", "federated")
        os.makedirs(self.global_model_save_dir, exist_ok=True)
        self.initial_global_model_path = os.path.join(self.global_model_save_dir, "initial_global_model.pth")
        self.best_global_model_path = os.path.join(self.global_model_save_dir, "best_global_fl_model.pth")

        logger.info(f"Federated Server initializing with device: {self.device}")
        logger.info(f"Attempting to initialize for {len(self.client_ids_all_potential)} potential clients: {self.client_ids_all_potential}")

        self.global_model = GearboxCNNLSTM(
            window_size=CFG_WINDOW_SIZE, lstm_hidden_size=LSTM_HIDDEN_SIZE,
            num_lstm_layers=NUM_LSTM_LAYERS, num_sensors=CFG_SENSORS,
            dropout_rate=DROPOUT_RATE
        ).to(self.device)
        
        torch.save(self.global_model.state_dict(), self.initial_global_model_path)
        logger.info(f"Initial global model state (random weights) saved to: {self.initial_global_model_path}")
        
        for client_id_str in self.client_ids_all_potential:
            try:
                client = FederatedClient(client_id=client_id_str, device=self.device)
                if client.num_train_samples > 0: 
                    self.clients.append(client)
                    logger.info(f"Successfully initialized and added FederatedClient for ID: {client_id_str} ({client.num_train_samples} train samples)")
                else:
                    logger.warning(f"Client '{client_id_str}' initialized but has no training data. It will not participate in FL rounds.")
            except Exception as e:
                logger.error(f"Failed to initialize or load data for client {client_id_str}: {e}", exc_info=True)
        
        if not self.clients:
            logger.error("CRITICAL: No clients were successfully initialized with data. Federated learning cannot proceed.")
            sys.exit(1)
            
        self.num_participating_clients = len(self.clients)
        logger.info(f"Federated Server ready with {self.num_participating_clients} participating client instances.")
    
    def aggregate_models(self, client_updates: list):
        if not client_updates:
            logger.warning("No client model updates received for aggregation. Global model remains unchanged.")
            return self.global_model.state_dict()

        logger.info(f"Aggregating {len(client_updates)} client model updates.")

        client_model_state_dicts = [upd['state_dict'] for upd in client_updates]
        client_data_sizes = [upd['data_size'] for upd in client_updates]
        
        total_data_size = sum(client_data_sizes)
        if total_data_size == 0:
            logger.warning("Total data size from clients is 0. Using equal weights for aggregation.")
            normalized_weights = [1.0 / len(client_model_state_dicts)] * len(client_model_state_dicts)
        else:
            normalized_weights = [size / total_data_size for size in client_data_sizes]
        
        logger.info(f"Client data sizes for aggregation: {client_data_sizes}")
        logger.info(f"Normalized aggregation weights: {[f'{w:.3f}' for w in normalized_weights]}")
        
        global_model_template_state_dict = self.global_model.state_dict()
        aggregated_state_dict = {}

        for key in global_model_template_state_dict.keys():
            if global_model_template_state_dict[key].is_floating_point():
                acc_tensor = torch.zeros_like(global_model_template_state_dict[key], dtype=torch.float32).to(self.device)
                for i, client_state_dict in enumerate(client_model_state_dicts):
                    if key in client_state_dict:
                        acc_tensor += client_state_dict[key].to(self.device).float() * normalized_weights[i]
                    else:
                        client_identifier = self.clients[i].client_id if i < len(self.clients) else f'client_index_{i}'
                        logger.warning(f"Key '{key}' not found in state_dict from {client_identifier}. Skipping for this client in aggregation.")
                aggregated_state_dict[key] = acc_tensor
            else:
                aggregated_state_dict[key] = global_model_template_state_dict[key].clone().to(self.device)
        
        self.global_model.load_state_dict(aggregated_state_dict)
        logger.info("Global model updated with aggregated weights.")
        return self.global_model.state_dict()

    def evaluate_global_model_on_clients_val_data(self):
        self.global_model.eval()
        
        # Initialize sums for metrics we want to average, and counts of valid samples for each
        # This ensures that if a metric is missing or invalid from a client, it doesn't skew the average.
        metrics_to_aggregate = [
            'val_loss', 'f1_score', 'sensor_anomaly_mse', 'detection_auc',
            'accuracy', 'precision', 'recall'
            # Add 'avg_high_anomaly_sensors' if it's consistently returned and numeric
        ]
        sum_weighted_metrics = {key: 0.0 for key in metrics_to_aggregate}
        total_samples_for_metric = {key: 0 for key in metrics_to_aggregate}
        
        logger.info("Evaluating current global model on all participating clients' validation data...")
        default_criterion = FaultLocalizationLoss(
            alpha=LOCATION_LOSS_WEIGHT, focal_gamma=FOCAL_GAMMA, 
            pos_weight_val=POS_WEIGHT, label_smoothing=label_smoothing
        )

        for client in self.clients:
            if client.val_loader is None or client.num_val_samples == 0 :
                logger.warning(f"  Client {client.client_id} has no validation data. Skipping its evaluation.")
                continue

            # The evaluate_model_on_dataset is the 'evaluate' function from src.training.train
            client_val_metrics = evaluate_model_on_dataset(
                model=self.global_model, val_loader=client.val_loader,
                criterion=default_criterion, device=self.device,
                num_sensors_config=CFG_SENSORS,
                print_debug_samples=False # No need for detailed debug prints during server evaluation
            )
            
            if client_val_metrics and client.num_val_samples > 0:
                logger.info(f"  Metrics for {client.client_id} (on global model): "
                            f"Val Loss={client_val_metrics.get('val_loss', float('nan')):.4f}, "
                            f"F1={client_val_metrics.get('f1_score', float('nan')):.4f}, "
                            f"SensorMSE={client_val_metrics.get('sensor_anomaly_mse', float('nan')):.4f}, "
                            f"AUC={client_val_metrics.get('detection_auc', float('nan')):.4f}")
                
                for key in metrics_to_aggregate:
                    value = client_val_metrics.get(key)
                    # More robust check for valid numeric types, including numpy floats
                    if isinstance(value, (int, float, np.number)) and not np.isnan(value) and not np.isinf(value):
                        sum_weighted_metrics[key] += value * client.num_val_samples
                        total_samples_for_metric[key] += client.num_val_samples
                    else:
                        logger.debug(f"  Metric '{key}' for client {client.client_id} is invalid or not found. Value: {value}, Type: {type(value)}. Not included in average for this metric.")
            else:
                logger.warning(f"  No valid metrics returned from evaluation for client {client.client_id}")
        
        aggregated_metrics = {}
        if sum(total_samples_for_metric.values()) == 0 : # Check if any client contributed any valid metrics
             logger.warning("No validation metrics collected from any client (or no val samples). Global model evaluation results will be default/zero.")
             # Fallback to default values if no valid metrics were collected at all
             for key in metrics_to_aggregate:
                aggregated_metrics[f"avg_{key}"] = float('inf') if 'loss' in key or 'mse' in key else 0.0
        else:
            for key in metrics_to_aggregate:
                if total_samples_for_metric[key] > 0:
                    aggregated_metrics[f"avg_{key}"] = sum_weighted_metrics[key] / total_samples_for_metric[key]
                else: # If no client reported this specific metric validly
                    logger.warning(f"No valid samples found for metric '{key}' across all clients. Setting avg to default.")
                    aggregated_metrics[f"avg_{key}"] = float('inf') if 'loss' in key or 'mse' in key else 0.0
        
        # Add 'avg_avg_high_anomaly_sensors' if it was part of client_val_metrics
        # This metric is calculated within evaluate_model_on_dataset and should be present
        if 'avg_high_anomaly_sensors' in client_val_metrics: # Check if key exists from last client (example)
            sum_avg_high_sensors = 0
            count_avg_high_sensors_samples = 0
            for client in self.clients: # Re-iterate to get this specific metric if needed
                # This assumes evaluate_model_on_dataset consistently returns 'avg_high_anomaly_sensors'
                # For simplicity, this part can be integrated into the main loop above if 'avg_high_anomaly_sensors'
                # is added to metrics_to_aggregate.
                # Let's assume it's already handled if 'avg_high_anomaly_sensors' is in metrics_to_aggregate.
                pass # This logic is now covered by the generic loop if key is added to metrics_to_aggregate

        logger.info(f"Aggregated Global Model Evaluation (weighted by client val samples): {aggregated_metrics}")
        return aggregated_metrics

    def save_global_model(self, model_path, is_best=False):
        try:
            torch.save(self.global_model.state_dict(), model_path)
            log_msg = f"Best global FL model state saved to: {model_path}" if is_best else f"Global FL model state saved to: {model_path}"
            logger.info(log_msg)
        except Exception as e:
            logger.error(f"Error saving global FL model to {model_path}: {e}", exc_info=True)

    def train_federated_rounds(self, num_rounds=10):
        logger.info(f"Starting federated training process for {num_rounds} rounds with {self.num_participating_clients} clients...")
        best_global_f1_score = -1.0 

        for current_round in range(1, num_rounds + 1):
            round_start_time = datetime.now()
            logger.info(f"\n===== Federated Round {current_round}/{num_rounds} =====")
            
            global_model_state_dict = self.global_model.state_dict()
            logger.info("Broadcasting global model to participating clients...")
            active_clients_this_round = [] 
            for client in self.clients: 
                if client.num_train_samples > 0: 
                    client.receive_model_update(global_model_state_dict)
                    active_clients_this_round.append(client)
                else:
                    logger.warning(f"Client {client.client_id} has no training data, skipping its participation in round {current_round}.")
            
            if not active_clients_this_round:
                logger.error(f"No clients have training data for round {current_round}. Stopping FL.")
                break

            client_updates_this_round = []
            logger.info(f"Starting local training on {len(active_clients_this_round)} active clients...")
            for client in active_clients_this_round:
                logger.info(f"  Training client: {client.client_id} (Round {current_round})...")
                try:
                    updated_state, local_val_metrics, samples_trained_on = client.train_local_model()
                    if updated_state and local_val_metrics and samples_trained_on > 0:
                        client_updates_this_round.append({
                            "state_dict": updated_state, "metrics": local_val_metrics, "data_size": samples_trained_on})
                        logger.info(f"  Client {client.client_id} finished local training. "
                                    f"Local Val F1: {local_val_metrics.get('f1_score', float('nan')):.4f}, "
                                    f"Local Val Loss: {local_val_metrics.get('val_loss', float('nan')):.4f}")
                    elif samples_trained_on == 0:
                         logger.warning(f"  Client {client.client_id} trained on 0 samples. No update collected.")
                    else:
                        logger.warning(f"  Client {client.client_id} did not return valid updates. Skipping for aggregation.")
                except Exception as e:
                    logger.error(f"  Error during local training for client {client.client_id}: {e}", exc_info=True)
            
            if not client_updates_this_round:
                logger.warning(f"No valid client updates received in Round {current_round}. Global model will not be updated.")
                continue

            logger.info(f"Aggregating {len(client_updates_this_round)} client model updates for Round {current_round}...")
            self.aggregate_models(client_updates_this_round)

            logger.info(f"Evaluating updated global model after Round {current_round}...")
            global_performance_metrics = self.evaluate_global_model_on_clients_val_data()
            
            current_global_f1 = global_performance_metrics.get("avg_f1_score", 0.0) # Use the avg_f1_score key
            logger.info(f"Global Model - End of Round {current_round}: "
                        f"Avg Val Loss={global_performance_metrics.get('avg_val_loss', float('nan')):.4f}, "
                        f"Avg F1={current_global_f1:.4f}, "
                        f"Avg SensorMSE={global_performance_metrics.get('avg_sensor_anomaly_mse', float('nan')):.4f}, "
                        f"Avg AUC={global_performance_metrics.get('avg_detection_auc', float('nan')):.4f}")
            
            if isinstance(current_global_f1, float) and not np.isnan(current_global_f1) and not np.isinf(current_global_f1):
                if current_global_f1 > best_global_f1_score:
                    best_global_f1_score = current_global_f1
                    self.save_global_model(self.best_global_model_path, is_best=True)
            else:
                logger.warning(f"Current global F1 score ({current_global_f1}) is not valid for comparison. Best F1 remains {best_global_f1_score:.4f}")
            
            if current_round % 5 == 0 or current_round == num_rounds:
                round_checkpoint_filename = f"global_model_round_{current_round}_f1_{current_global_f1:.4f}.pth"
                round_checkpoint_path = os.path.join(self.global_model_save_dir, round_checkpoint_filename)
                self.save_global_model(round_checkpoint_path)
            
            logger.info(f"===== Round {current_round} completed in {datetime.now() - round_start_time} =====")

        logger.info(f"Federated training process completed. Best global F1-score achieved across client validations: {best_global_f1_score:.4f}")
