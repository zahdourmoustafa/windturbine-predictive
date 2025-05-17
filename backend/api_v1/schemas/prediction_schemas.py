from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class SensorDiagnosis(BaseModel):
    sensor: str
    predicted_state: str
    failure_mode: str
    associated_component: str

class Metrics(BaseModel):
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    confusion_matrix: Optional[List[List[int]]] = None

class LossMetrics(BaseModel):
    total_loss: Optional[float] = None
    detection_loss: Optional[float] = None
    anomaly_loss: Optional[float] = None

class PredictionResult(BaseModel):
    filename: str
    error: Optional[str] = None
    file_level_prediction: Optional[str] = None
    confidence_for_decision: Optional[float] = None # Generic confidence related to the prediction
    avg_overall_fault_probability_for_file: Optional[float] = None
    was_healthy_override_applied: Optional[bool] = None
    
    per_sensor_diagnoses: Optional[List[SensorDiagnosis]] = None
    
    # Metrics - will only be populated if GT is available
    overall_detection_metrics: Optional[Metrics] = None
    average_per_sensor_f1: Optional[float] = None
    per_sensor_f1_scores: Optional[Dict[str, float]] = None # e.g. {"AN3": 0.9369, ...}
    loss_metrics_vs_gt: Optional[LossMetrics] = None

    # Raw data for potential chart on frontend, keep it small
    raw_data_preview_segment: Optional[List[float]] = None
    
    # Avg sensor anomaly probabilities for bar chart
    avg_sensor_probs_across_all_windows: Optional[Dict[str, float]] = None # e.g. {"AN3": 0.88, ...}

    # MC Dropout related - if used
    mc_dropout_used: Optional[bool] = None
    avg_overall_fault_uncertainty: Optional[float] = None
    avg_sensor_anomaly_uncertainty_per_sensor: Optional[Dict[str, float]] = None

class PredictionResponse(BaseModel):
    data: Optional[PredictionResult] = None
    message: Optional[str] = None
    success: bool 