import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, precision_recall_curve, accuracy_score
import sys
import os

# Add project root for config import if train.py is called directly or from other locations
project_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root_path not in sys.path:
    sys.path.append(project_root_path)

try:
    from config.config import (
        DEFAULT_THRESHOLD, MIN_RECALL, LOCATION_LOSS_WEIGHT,
        FOCAL_GAMMA, POS_WEIGHT, SENSORS, SENSOR_NAMES_ORDERED
    )
except ImportError:
    print("Warning: Could not import some parameters from config.config in train.py. Using script defaults.")
    DEFAULT_THRESHOLD = 0.5
    MIN_RECALL = 0.5
    LOCATION_LOSS_WEIGHT = 1.0
    FOCAL_GAMMA = 2.0
    POS_WEIGHT = 3.0
    SENSORS = 8
    SENSOR_NAMES_ORDERED = [f'AN{i+3}' for i in range(SENSORS)] # Fallback

class FaultLocalizationLoss(nn.Module):
    def __init__(self, alpha=None, label_smoothing=0.05,
                 focal_gamma=None, pos_weight_val=None):
        super().__init__()
        self.alpha = alpha if alpha is not None else LOCATION_LOSS_WEIGHT
        _gamma = focal_gamma if focal_gamma is not None else FOCAL_GAMMA
        _pos_w = pos_weight_val if pos_weight_val is not None else POS_WEIGHT
        
        self.label_smoothing = label_smoothing
        self.focal_gamma = _gamma
        
        self.bce_detection_raw = nn.BCEWithLogitsLoss(reduction='none')
        self.pos_weight_detection = torch.tensor(_pos_w, dtype=torch.float) # Ensure float

        self.bce_anomaly_sensors = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, outputs, targets):
        fault_detection_logits = outputs['fault_detection'].squeeze(-1)
        sensor_anomalies_pred = outputs['sensor_anomalies']
        
        fault_label_binary = targets['fault_label'].float()
        sensor_locations_gt = targets['fault_location'].float()
        
        smoothed_targets_binary_detection = fault_label_binary * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        bce_loss_detection_elements = self.bce_detection_raw(fault_detection_logits, smoothed_targets_binary_detection)
        
        pt_detection = torch.exp(-bce_loss_detection_elements)
        focal_weight_detection = (1 - pt_detection) ** self.focal_gamma
        
        self.pos_weight_detection = self.pos_weight_detection.to(bce_loss_detection_elements.device)
        class_weights_detection = torch.where(fault_label_binary > 0.5, self.pos_weight_detection, torch.ones_like(fault_label_binary))
        
        weighted_detection_loss_elements = focal_weight_detection * class_weights_detection * bce_loss_detection_elements
        detection_loss = weighted_detection_loss_elements.mean()
        
        anomaly_loss = self.bce_anomaly_sensors(sensor_anomalies_pred, sensor_locations_gt)
        
        total_loss = detection_loss + self.alpha * anomaly_loss
        
        return {
            'total_loss': total_loss,
            'detection_loss': detection_loss,
            'anomaly_loss': anomaly_loss
        }

def create_multi_sensor_damage_patterns(batch_data, batch_labels_binary, batch_locations_gt, num_sensors=SENSORS):
    augmented_data_list = [batch_data]
    augmented_labels_list = [batch_labels_binary]
    augmented_locations_list = [batch_locations_gt]
    
    damage_patterns_indices = [
        [0, 1], [3, 6], [5, 6], [2, 7], [0, 1, 3], [3, 4, 5, 6]
    ]
    
    batch_size = batch_data.size(0)
    time_steps = batch_data.size(1)
    
    for i in range(batch_size):
        if batch_labels_binary[i] > 0.5:
            num_augmentations_per_sample = np.random.randint(1, 3)
            for _ in range(num_augmentations_per_sample):
                pattern_idx_choice = torch.randint(0, len(damage_patterns_indices), (1,)).item()
                chosen_sensor_indices_in_pattern = damage_patterns_indices[pattern_idx_choice]
                
                new_sample_data = batch_data[i].clone()
                new_sample_locations = batch_locations_gt[i].clone()
                base_noise_amplitude = np.random.uniform(0.1, 0.4)
                
                for sensor_idx_to_augment in chosen_sensor_indices_in_pattern:
                    if 0 <= sensor_idx_to_augment < num_sensors:
                        sensor_specific_noise = (torch.randn(time_steps) * base_noise_amplitude).to(new_sample_data.device)
                        new_sample_data[:, sensor_idx_to_augment] += sensor_specific_noise
                        current_gt_anomaly_score = new_sample_locations[sensor_idx_to_augment].item()
                        augmentation_strength_for_gt = np.random.uniform(0.3, 0.7)
                        new_gt_anomaly_score = min(current_gt_anomaly_score + augmentation_strength_for_gt, 1.0)
                        new_sample_locations[sensor_idx_to_augment] = new_gt_anomaly_score
                    else:
                        print(f"Warning: Augmentation sensor index {sensor_idx_to_augment} out of bounds for {num_sensors} sensors.")
                
                augmented_data_list.append(new_sample_data.unsqueeze(0))
                augmented_labels_list.append(batch_labels_binary[i].unsqueeze(0))
                augmented_locations_list.append(new_sample_locations.unsqueeze(0))
    
    final_data = torch.cat(augmented_data_list, dim=0)
    final_labels = torch.cat(augmented_labels_list, dim=0)
    final_locations = torch.cat(augmented_locations_list, dim=0)
    
    return final_data, final_labels, final_locations

def train_epoch(model, train_loader, optimizer, criterion=None, device=None,
                use_augmentation=True, num_sensors_config=SENSORS, gradient_clip_val=1.0,
                current_epoch_num=0, total_epochs=1):
    if criterion is None:
        criterion = FaultLocalizationLoss(alpha=LOCATION_LOSS_WEIGHT, focal_gamma=FOCAL_GAMMA, pos_weight_val=POS_WEIGHT)
    if device is None:
        device = next(model.parameters()).device
    
    model.train()
    total_loss_sum, detection_loss_sum, anomaly_loss_sum = 0.0, 0.0, 0.0
    all_fault_predictions_logits_epoch, all_fault_true_labels_epoch = [], []
    
    num_batches = len(train_loader)
    for batch_idx, (data, labels_binary_gt, locations_gt) in enumerate(train_loader):
        data, labels_binary_gt, locations_gt = data.to(device), labels_binary_gt.to(device), locations_gt.to(device)
        
        if use_augmentation:
            data, labels_binary_gt, locations_gt = create_multi_sensor_damage_patterns(data, labels_binary_gt, locations_gt, num_sensors=num_sensors_config)
        
        targets_dict = {'fault_label': labels_binary_gt, 'fault_location': locations_gt}
        optimizer.zero_grad()
        model_outputs = model(data)
        loss_components = criterion(model_outputs, targets_dict)
        batch_total_loss = loss_components['total_loss']
        batch_total_loss.backward()
        
        if gradient_clip_val is not None and gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_val)
        optimizer.step()
        
        total_loss_sum += batch_total_loss.item()
        detection_loss_sum += loss_components['detection_loss'].item()
        anomaly_loss_sum += loss_components['anomaly_loss'].item()
        
        all_fault_predictions_logits_epoch.append(model_outputs['fault_detection'].detach().cpu())
        all_fault_true_labels_epoch.append(labels_binary_gt.detach().cpu())

        if (batch_idx + 1) % 100 == 0: # Log less frequently to reduce clutter
             print(f"  Epoch {current_epoch_num}/{total_epochs}, Batch {batch_idx+1}/{num_batches} - Batch Loss: {batch_total_loss.item():.4f}")
    
    all_fault_predictions_logits_epoch = torch.cat(all_fault_predictions_logits_epoch).squeeze().numpy()
    all_fault_true_labels_epoch = torch.cat(all_fault_true_labels_epoch).numpy()
    
    try:
        epoch_detection_auc = roc_auc_score(all_fault_true_labels_epoch, all_fault_predictions_logits_epoch)
    except ValueError:
        epoch_detection_auc = 0.5
        print("Warning: ROC AUC calculation failed in train_epoch. Setting AUC to 0.5.")
    
    all_fault_probs_epoch = 1 / (1 + np.exp(-all_fault_predictions_logits_epoch))
    chosen_threshold_for_metrics = DEFAULT_THRESHOLD
    try:
        precision_values, recall_values, threshold_options_pr = precision_recall_curve(all_fault_true_labels_epoch, all_fault_probs_epoch)
        valid_indices_for_recall = np.where(recall_values >= MIN_RECALL)[0]
        if len(valid_indices_for_recall) > 0:
            f1_scores_at_valid_recall = (2 * precision_values[valid_indices_for_recall] * recall_values[valid_indices_for_recall]) / \
                                      (precision_values[valid_indices_for_recall] + recall_values[valid_indices_for_recall] + 1e-8)
            best_f1_idx_within_valid_recall = np.argmax(f1_scores_at_valid_recall)
            actual_best_threshold_idx = valid_indices_for_recall[best_f1_idx_within_valid_recall]
            if actual_best_threshold_idx < len(threshold_options_pr):
                 chosen_threshold_for_metrics = threshold_options_pr[actual_best_threshold_idx]
            elif len(threshold_options_pr) > 0:
                 chosen_threshold_for_metrics = threshold_options_pr[-1]
        elif len(threshold_options_pr) > 0 and len(precision_values) > 1 : # If no threshold meets min_recall, pick best F1 overall
            f1_scores_all = (2 * precision_values[:-1] * recall_values[:-1]) / (precision_values[:-1] + recall_values[:-1] + 1e-8)
            if len(f1_scores_all) > 0:
                chosen_threshold_for_metrics = threshold_options_pr[np.argmax(f1_scores_all)]
    except Exception as e:
        print(f"Warning: Error during dynamic threshold calculation in train_epoch: {e}. Using default: {DEFAULT_THRESHOLD}.")
    
    binary_preds_for_metrics = (all_fault_probs_epoch >= chosen_threshold_for_metrics).astype(int)
    precision_metric, recall_metric, f1_metric, _ = precision_recall_fscore_support(
        all_fault_true_labels_epoch, binary_preds_for_metrics, average='binary', zero_division=0)
    
    epoch_metrics = {
        'total_loss': total_loss_sum / num_batches if num_batches > 0 else 0,
        'detection_loss': detection_loss_sum / num_batches if num_batches > 0 else 0,
        'anomaly_loss': anomaly_loss_sum / num_batches if num_batches > 0 else 0,
        'detection_auc': epoch_detection_auc,
        'precision_at_chosen_thresh': precision_metric,
        'recall_at_chosen_thresh': recall_metric,
        'f1_at_chosen_thresh': f1_metric,
        'chosen_threshold_for_metrics': chosen_threshold_for_metrics
    }
    return epoch_metrics

def evaluate(model, val_loader, criterion=None, device=None, num_sensors_config=SENSORS, print_debug_samples=False):
    if criterion is None:
        criterion = FaultLocalizationLoss(alpha=LOCATION_LOSS_WEIGHT, focal_gamma=FOCAL_GAMMA, pos_weight_val=POS_WEIGHT)
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    total_loss_sum_val, detection_loss_sum_val, anomaly_loss_sum_val = 0.0, 0.0, 0.0
    all_fault_logits_val_list, all_fault_true_labels_val_list = [], []
    all_sensor_anomalies_pred_logits_val_list, all_sensor_locations_gt_val_list = [], []
    
    num_batches_val = len(val_loader)
    if num_batches_val == 0:
        print("Warning: Validation loader is empty in evaluate function. Returning zero metrics.")
        # Ensure all expected keys are present, even if with default/zero values
        current_sensor_names = SENSOR_NAMES_ORDERED[:num_sensors_config]
        return {
            'total_loss': 0, 'detection_loss': 0, 'anomaly_loss': 0, 'detection_auc': 0.5,
            'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0,
            'chosen_threshold': DEFAULT_THRESHOLD, 'sensor_anomaly_mse': float('inf'), 
            'per_sensor_mse': {name: float('inf') for name in current_sensor_names},
            'val_loss': float('inf'),
            'avg_high_anomaly_sensors': 0
        }

    with torch.no_grad():
        for data, labels_binary_gt, locations_gt in val_loader:
            data, labels_binary_gt, locations_gt = data.to(device), labels_binary_gt.to(device), locations_gt.to(device)
            targets_dict = {'fault_label': labels_binary_gt, 'fault_location': locations_gt}
            model_outputs = model(data)
            loss_components = criterion(model_outputs, targets_dict)
            
            total_loss_sum_val += loss_components['total_loss'].item()
            detection_loss_sum_val += loss_components['detection_loss'].item()
            anomaly_loss_sum_val += loss_components['anomaly_loss'].item()
            
            all_fault_logits_val_list.append(model_outputs['fault_detection'].cpu())
            all_fault_true_labels_val_list.append(labels_binary_gt.cpu())
            all_sensor_anomalies_pred_logits_val_list.append(model_outputs['sensor_anomalies'].cpu())
            all_sensor_locations_gt_val_list.append(locations_gt.cpu())
    
    all_fault_logits_val = torch.cat(all_fault_logits_val_list).squeeze().numpy()
    all_fault_true_labels_val = torch.cat(all_fault_true_labels_val_list).numpy()
    all_sensor_anomalies_pred_logits_val = torch.cat(all_sensor_anomalies_pred_logits_val_list).numpy()
    all_sensor_locations_gt_val = torch.cat(all_sensor_locations_gt_val_list).numpy()

    try:
        val_detection_auc = roc_auc_score(all_fault_true_labels_val, all_fault_logits_val)
    except ValueError:
        val_detection_auc = 0.5
        print("Warning: ROC AUC calculation failed in evaluate. Setting AUC to 0.5.")
    
    all_fault_probs_val = 1 / (1 + np.exp(-all_fault_logits_val))
    chosen_threshold_for_val_metrics = DEFAULT_THRESHOLD
    try:
        precision_values_val, recall_values_val, threshold_options_pr_val = precision_recall_curve(all_fault_true_labels_val, all_fault_probs_val)
        valid_indices_for_recall_val = np.where(recall_values_val >= MIN_RECALL)[0]
        if len(valid_indices_for_recall_val) > 0:
            f1_scores_at_valid_recall_val = (2 * precision_values_val[valid_indices_for_recall_val] * recall_values_val[valid_indices_for_recall_val]) / \
                                          (precision_values_val[valid_indices_for_recall_val] + recall_values_val[valid_indices_for_recall_val] + 1e-8)
            best_f1_idx_val = np.argmax(f1_scores_at_valid_recall_val)
            actual_best_threshold_idx_val = valid_indices_for_recall_val[best_f1_idx_val]
            if actual_best_threshold_idx_val < len(threshold_options_pr_val):
                 chosen_threshold_for_val_metrics = threshold_options_pr_val[actual_best_threshold_idx_val]
            elif len(threshold_options_pr_val) > 0:
                 chosen_threshold_for_val_metrics = threshold_options_pr_val[-1]
        elif len(threshold_options_pr_val) > 0 and len(precision_values_val) > 1:
            f1_scores_all_val = (2 * precision_values_val[:-1] * recall_values_val[:-1]) / (precision_values_val[:-1] + recall_values_val[:-1] + 1e-8)
            if len(f1_scores_all_val) > 0:
                chosen_threshold_for_val_metrics = threshold_options_pr_val[np.argmax(f1_scores_all_val)]
    except Exception as e:
        print(f"Warning: Error during dynamic threshold calculation in evaluate: {e}. Using default: {DEFAULT_THRESHOLD}.")

    binary_preds_for_val_metrics = (all_fault_probs_val >= chosen_threshold_for_val_metrics).astype(int)
    precision_val_metric, recall_val_metric, f1_val_metric, _ = precision_recall_fscore_support(
        all_fault_true_labels_val, binary_preds_for_val_metrics, average='binary', zero_division=0)
    accuracy_val_metric = accuracy_score(all_fault_true_labels_val, binary_preds_for_val_metrics)

    all_sensor_anomalies_pred_probs_val = 1 / (1 + np.exp(-all_sensor_anomalies_pred_logits_val))
    sensor_anomaly_mse_val = np.mean((all_sensor_anomalies_pred_probs_val - all_sensor_locations_gt_val)**2)
    
    per_sensor_mse_val = {}
    current_sensor_names = SENSOR_NAMES_ORDERED[:num_sensors_config]
    if all_sensor_anomalies_pred_probs_val.shape[1] == num_sensors_config:
        per_sensor_mse_values = np.mean((all_sensor_anomalies_pred_probs_val - all_sensor_locations_gt_val)**2, axis=0)
        per_sensor_mse_val = {current_sensor_names[i]: mse_val for i, mse_val in enumerate(per_sensor_mse_values)}
    else:
        print(f"Warning: Shape mismatch for per-sensor MSE. Pred_shape: {all_sensor_anomalies_pred_probs_val.shape}, Num_sensors_config: {num_sensors_config}")

    avg_high_anomaly_sensors = 0
    faulty_sample_indices = np.where(all_fault_true_labels_val == 1)[0]
    if len(faulty_sample_indices) > 0:
        sensor_preds_for_faulty_samples = all_sensor_anomalies_pred_probs_val[faulty_sample_indices]
        avg_high_anomaly_sensors = np.mean(np.sum(sensor_preds_for_faulty_samples > 0.5, axis=1))

    if print_debug_samples and len(all_sensor_locations_gt_val) > 0:
        print("\n--- MSE DEBUG: Sample Sensor Predictions vs GT (First few validation samples) ---")
        num_samples_to_print = min(3, len(all_sensor_locations_gt_val)) # Print fewer samples to reduce log size
        for i in range(num_samples_to_print):
            overall_gt_fault_label = all_fault_true_labels_val[i]
            print(f"Sample Index {i} (Overall GT Fault Label: {overall_gt_fault_label:.0f}):")
            for sensor_j in range(num_sensors_config):
                gt_score = all_sensor_locations_gt_val[i, sensor_j]
                pred_prob = all_sensor_anomalies_pred_probs_val[i, sensor_j]
                squared_error = (pred_prob - gt_score)**2
                print(f"  {current_sensor_names[sensor_j]}: GT={gt_score:.3f}, PredProb={pred_prob:.3f}, SqError={squared_error:.3f}")
        
        # Select one clear healthy and one clear damaged sample if possible for detailed print
        healthy_indices = np.where(all_fault_true_labels_val == 0)[0]
        damaged_indices = np.where(all_fault_true_labels_val == 1)[0]

        if len(healthy_indices) > 0:
            # Pick a random healthy sample for more variety if printed multiple times
            idx = np.random.choice(healthy_indices) 
            print(f"\nExample HEALTHY Sample (Index {idx}, Overall GT Fault: {all_fault_true_labels_val[idx]:.0f}):")
            for sensor_j in range(num_sensors_config):
                 print(f"  {current_sensor_names[sensor_j]}: GT={all_sensor_locations_gt_val[idx, sensor_j]:.3f}, PredProb={all_sensor_anomalies_pred_probs_val[idx, sensor_j]:.3f}")
        else:
            print("\nNo healthy samples found in this validation batch for debug print.")
        
        if len(damaged_indices) > 0:
            idx = np.random.choice(damaged_indices)
            print(f"\nExample DAMAGED Sample (Index {idx}, Overall GT Fault: {all_fault_true_labels_val[idx]:.0f}):")
            for sensor_j in range(num_sensors_config):
                 print(f"  {current_sensor_names[sensor_j]}: GT={all_sensor_locations_gt_val[idx, sensor_j]:.3f}, PredProb={all_sensor_anomalies_pred_probs_val[idx, sensor_j]:.3f}")
        else:
            print("\nNo damaged samples found in this validation batch for debug print.")
        print("--- End MSE DEBUG ---")

    val_metrics_dict = {
        'total_loss': total_loss_sum_val / num_batches_val if num_batches_val > 0 else float('inf'),
        'detection_loss': detection_loss_sum_val / num_batches_val if num_batches_val > 0 else float('inf'),
        'anomaly_loss': anomaly_loss_sum_val / num_batches_val if num_batches_val > 0 else float('inf'),
        'detection_auc': val_detection_auc,
        'accuracy': accuracy_val_metric,
        'precision': precision_val_metric,
        'recall': recall_val_metric,
        'f1_score': f1_val_metric,
        'chosen_threshold': chosen_threshold_for_val_metrics,
        'sensor_anomaly_mse': sensor_anomaly_mse_val,
        'per_sensor_mse': per_sensor_mse_val,
        'val_loss': total_loss_sum_val / num_batches_val if num_batches_val > 0 else float('inf'),
        'avg_high_anomaly_sensors': avg_high_anomaly_sensors
    }
    
    print_evaluation_summary(val_metrics_dict, "Validation")
    return val_metrics_dict

def print_evaluation_summary(metrics, evaluation_set_name="Evaluation"):
    print(f"\n--- {evaluation_set_name} Summary ---")
    print(f"Overall Loss (val_loss): {metrics.get('val_loss', metrics.get('total_loss', 0)):.4f}")
    print(f"  Detection Loss part: {metrics.get('detection_loss', 0):.4f}")
    print(f"  Sensor Anomaly Loss part: {metrics.get('anomaly_loss', 0):.4f}")
    print(f"Fault Detection AUC: {metrics.get('detection_auc', 0):.4f}")
    print(f"Fault Detection Metrics (Threshold: {metrics.get('chosen_threshold', 0):.3f}):")
    print(f"  Accuracy:  {metrics.get('accuracy',0):.4f}")
    print(f"  Precision: {metrics.get('precision',0):.4f}")
    print(f"  Recall:    {metrics.get('recall',0):.4f}")
    print(f"  F1-score:  {metrics.get('f1_score',0):.4f} (Often used for best model selection)")
    print(f"Sensor Anomaly Prediction MSE (vs GT scores): {metrics.get('sensor_anomaly_mse', 0):.4f}")
    if metrics.get('per_sensor_mse') and isinstance(metrics['per_sensor_mse'], dict):
        print("  Per-Sensor MSEs:")
        for sensor, mse in metrics['per_sensor_mse'].items():
            print(f"    {sensor}: {mse:.4f}")
    if 'avg_high_anomaly_sensors' in metrics:
        print(f"Avg high-anomaly sensors in faulty samples (pred > 0.5): {metrics['avg_high_anomaly_sensors']:.2f}")
    print("----------------------------")

def train(model, train_loader, val_loader, config_obj):
    device = torch.device(config_obj.device if hasattr(config_obj, 'device') else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    num_sensors_model = model.num_sensors if hasattr(model, 'num_sensors') else SENSORS
    
    num_epochs = getattr(config_obj, 'num_epochs', 20)
    learning_rate = getattr(config_obj, 'learning_rate', 0.001)
    weight_decay = getattr(config_obj, 'weight_decay', 1e-4)
    use_aug = getattr(config_obj, 'use_augmentation', True)
    grad_clip = getattr(config_obj, 'gradient_clip_val', 1.0)
    
    loc_loss_weight = getattr(config_obj, 'LOCATION_LOSS_WEIGHT', LOCATION_LOSS_WEIGHT)
    f_gamma = getattr(config_obj, 'FOCAL_GAMMA', FOCAL_GAMMA)
    p_weight = getattr(config_obj, 'POS_WEIGHT', POS_WEIGHT)
    lbl_smoothing = getattr(config_obj, 'label_smoothing', 0.05)

    criterion = FaultLocalizationLoss(alpha=loc_loss_weight, focal_gamma=f_gamma, pos_weight_val=p_weight, label_smoothing=lbl_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler_pat = getattr(config_obj, 'scheduler_patience', 5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=scheduler_pat, verbose=True)
    
    early_stop_patience = getattr(config_obj, 'early_stopping_patience', 10)
    metric_to_monitor_early_stop = getattr(config_obj, 'metric_for_best_model', 'f1_score')
    
    if metric_to_monitor_early_stop in ['val_loss', 'total_loss', 'sensor_anomaly_mse']:
        monitor_mode_early_stop = 'min'
        best_metric_val_early_stop = float('inf')
    else:
        monitor_mode_early_stop = 'max'
        best_metric_val_early_stop = float('-inf')
        
    epochs_no_improve = 0
    best_model_state_dict = model.state_dict().copy()

    print(f"\n--- Starting Training ---")
    print(f"Device: {device}")
    print(f"Total Epochs: {num_epochs}, LR: {learning_rate}, Weight Decay: {weight_decay}")
    print(f"Monitoring '{metric_to_monitor_early_stop}' ({monitor_mode_early_stop} mode) for early stopping (patience: {early_stop_patience}).")
    print(f"Loss Params: Alpha={criterion.alpha:.2f}, FocalGamma={criterion.focal_gamma:.2f}, PosWeightDetection={criterion.pos_weight_detection.item():.2f}, LabelSmoothing={criterion.label_smoothing:.2f}")
    print(f"Augmentation: {use_aug}, Grad Clip: {grad_clip}")

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device,
            use_augmentation=use_aug, num_sensors_config=num_sensors_model,
            gradient_clip_val=grad_clip, current_epoch_num=epoch+1, total_epochs=num_epochs)
        
        print(f"Train Metrics: Avg Total Loss={train_metrics['total_loss']:.4f}, Detection AUC={train_metrics['detection_auc']:.4f}, F1(chosen_thresh)={train_metrics['f1_at_chosen_thresh']:.4f}")
        print(f"  Chosen Threshold (train): {train_metrics['chosen_threshold_for_metrics']:.3f}, Anomaly Loss (train): {train_metrics['anomaly_loss']:.4f}")

        debug_this_epoch = (epoch < 2) 
        
        val_metrics = {} # Initialize val_metrics
        if val_loader is not None and len(val_loader) > 0:
            val_metrics = evaluate(model, val_loader, criterion, device, num_sensors_config=num_sensors_model, print_debug_samples=debug_this_epoch)
            current_val_loss_for_scheduler = val_metrics.get('val_loss', float('inf')) # Use .get for safety
            if current_val_loss_for_scheduler != float('inf'):
                 scheduler.step(current_val_loss_for_scheduler)
        else:
            print("Warning: Validation loader is empty or None. Skipping validation and scheduler step for this epoch.")
            current_val_loss_for_scheduler = float('inf') # Cannot step scheduler
            # Populate val_metrics with defaults if val_loader is None to avoid KeyErrors later
            current_sensor_names = SENSOR_NAMES_ORDERED[:num_sensors_model]
            val_metrics = {
                'total_loss': float('inf'), 'detection_loss': float('inf'), 'anomaly_loss': float('inf'), 
                'detection_auc': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0,
                'chosen_threshold': DEFAULT_THRESHOLD, 'sensor_anomaly_mse': float('inf'),
                'per_sensor_mse': {name: float('inf') for name in current_sensor_names},
                'val_loss': float('inf'), 'avg_high_anomaly_sensors': 0
            }


        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr:.6f}")
        
        current_metric_for_early_stop = val_metrics.get(metric_to_monitor_early_stop)
        if current_metric_for_early_stop is None or current_metric_for_early_stop == float('inf'):
            print(f"Warning: Metric '{metric_to_monitor_early_stop}' not found or invalid in validation. Defaulting to 'val_loss'.")
            metric_to_monitor_early_stop = 'val_loss'
            monitor_mode_early_stop = 'min'
            # Ensure best_metric_val_early_stop is correctly initialized or re-initialized if metric changes
            if monitor_mode_early_stop == 'min' and (best_metric_val_early_stop == float('-inf') or current_metric_for_early_stop == float('inf')):
                best_metric_val_early_stop = float('inf')
            elif monitor_mode_early_stop == 'max' and (best_metric_val_early_stop == float('inf') or current_metric_for_early_stop == float('-inf')):
                 best_metric_val_early_stop = float('-inf')
            current_metric_for_early_stop = val_metrics.get('val_loss', float('inf'))


        improved = False
        if current_metric_for_early_stop != float('inf') and current_metric_for_early_stop != float('-inf'): # Only compare valid metrics
            if monitor_mode_early_stop == 'max':
                if current_metric_for_early_stop > best_metric_val_early_stop:
                    best_metric_val_early_stop = current_metric_for_early_stop
                    improved = True
            else: # min mode
                if current_metric_for_early_stop < best_metric_val_early_stop:
                    best_metric_val_early_stop = current_metric_for_early_stop
                    improved = True
                
        if improved:
            print(f"Epoch {epoch+1}: Validation {metric_to_monitor_early_stop} improved to {best_metric_val_early_stop:.4f}. Saving model state...")
            best_model_state_dict = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            # Avoid printing if current_metric_for_early_stop is inf
            if current_metric_for_early_stop != float('inf') and current_metric_for_early_stop != float('-inf'):
                print(f"Epoch {epoch+1}: Val {metric_to_monitor_early_stop} ({current_metric_for_early_stop:.4f}) did not improve from best ({best_metric_val_early_stop:.4f}). Patience: {epochs_no_improve}/{early_stop_patience}")
            else:
                print(f"Epoch {epoch+1}: Val {metric_to_monitor_early_stop} is invalid. Not updating best model. Patience: {epochs_no_improve}/{early_stop_patience}")


        if epochs_no_improve >= early_stop_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs. Best val {metric_to_monitor_early_stop}: {best_metric_val_early_stop:.4f}")
            break
            
    if best_model_state_dict:
        # Check if best_metric_val_early_stop is still at its initial unachievable value
        is_initial_best_val = (monitor_mode_early_stop == 'min' and best_metric_val_early_stop == float('inf')) or \
                              (monitor_mode_early_stop == 'max' and best_metric_val_early_stop == float('-inf'))
        if is_initial_best_val :
             print("\nWarning: No improvement was seen during training. Loading initial model state or last epoch's model state.")
             # Decide if you want to load the initial model or the last one if no improvement
             # model.load_state_dict(initial_model_state_dict) # if you saved initial state
        else:
            print(f"\nLoading best model state with validation {metric_to_monitor_early_stop}: {best_metric_val_early_stop:.4f}")
            model.load_state_dict(best_model_state_dict)
    else: # Should not happen if best_model_state_dict is initialized
        print("\nWarning: No best model state was explicitly saved. Returning model from the last epoch.")
        
    return model

