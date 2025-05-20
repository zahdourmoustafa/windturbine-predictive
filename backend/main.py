import os
import torch
import sys
import logging
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# --- Setup Project Root Path ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) # Assumes main.py is in project root (e.g., backend/)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- Import Project Modules ---
try:
    from config.config import BASE_DIR, MAX_ROUNDS, OUTPUT_PATH as cfg_OUTPUT_PATH
    from src.api.server import FederatedServer
    # CLIENT_DATA_SOURCES will be imported from preprocess.py by the server/client logic if needed there,
    # or main.py can define the list of clients to initialize for the server.
    # For clarity, let's get it from preprocess.py here to define which clients participate.
    from src.data_processing.preprocess import CLIENT_DATA_SOURCES # Defined in your latest preprocess.py
    from api_v1 import prediction_endpoints  # Import the prediction endpoints router
except ModuleNotFoundError as e:
    print(f"ERROR: Could not import project modules in main.py. Ensure script is run from project root or PYTHONPATH is set.")
    print(f"Details: {e}")
    sys.exit(1)
except ImportError as e:
    print(f"ERROR: Could not import specific names from modules in main.py. Details: {e}")
    # Fallback if CLIENT_DATA_SOURCES cannot be imported (e.g., if preprocess.py is refactored)
    # This means you'd have to manually ensure client IDs here match processed data.
    # Adjust this fallback based on the clients you expect to have preprocessed data for.
    CLIENT_DATA_SOURCES = {f"Client_{i+1}":None for i in range(5)} # Default to 5 clients if import fails
    print(f"Warning: Could not import CLIENT_DATA_SOURCES from preprocess.py. Using fallback client list: {list(CLIENT_DATA_SOURCES.keys())}")


# --- Configure Logging ---
# Ensure the log directory exists
LOG_DIR = os.path.join(BASE_DIR, "output", "logs", "federated") # Log directory within BASE_DIR
os.makedirs(LOG_DIR, exist_ok=True)
log_file_path = os.path.join(LOG_DIR, f"fl_training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__) # Get logger for this module


def main_fl_training():
    """
    Main function to initialize and run the Federated Learning training process.
    """
    logger.info("=================================================")
    logger.info("=== STARTING FEDERATED LEARNING TRAINING RUN ===")
    logger.info("=================================================")

    # Determine the list of client IDs that will participate in FL
    # These IDs must have corresponding preprocessed data in data/processed/
    client_ids_for_fl = list(CLIENT_DATA_SOURCES.keys()) # Get client IDs from preprocess.py
    num_fl_clients = len(client_ids_for_fl)

    if num_fl_clients == 0:
        logger.error("No clients defined in CLIENT_DATA_SOURCES (from preprocess.py or fallback). Cannot start FL training.")
        return

    logger.info(f"Initializing Federated Server for {num_fl_clients} clients: {client_ids_for_fl}")
    
    # Correctly initialize FederatedServer with the 'client_ids' argument
    server = FederatedServer(client_ids=client_ids_for_fl) 
    
    # Number of federated rounds from config.py
    num_rounds = MAX_ROUNDS 
    logger.info(f"Starting federated training for {num_rounds} rounds...")
    
    # Assuming the server's main training method is train_federated_rounds
    server.train_federated_rounds(num_rounds=num_rounds) 
    
    # Define the save directory for the final FL model
    fl_model_save_dir = os.path.join(BASE_DIR, "models", "federated")
    os.makedirs(fl_model_save_dir, exist_ok=True)
    
    final_fl_model_filename = f"final_global_fl_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
    final_fl_model_path = os.path.join(fl_model_save_dir, final_fl_model_filename)
    
    try:
        torch.save(server.global_model.state_dict(), final_fl_model_path)
        logger.info(f"Final Global FL Model saved to: {final_fl_model_path}")
    except Exception as e:
        logger.error(f"Error saving final global FL model: {e}", exc_info=True)

    logger.info("================================================")
    logger.info("=== FEDERATED LEARNING TRAINING RUN COMPLETE ===")
    logger.info("================================================")

if __name__ == "__main__":
    # Ensure config variables like BASE_DIR are loaded
    if 'BASE_DIR' not in globals() or not BASE_DIR: # Check if BASE_DIR from config was loaded
        print("CRITICAL ERROR: BASE_DIR not loaded from config.py. Cannot proceed.")
        sys.exit(1)
    main_fl_training()

app = FastAPI(
    title="Gearbox Fault Detection API",
    description="API for predicting gearbox faults from .mat sensor data files.",
    version="1.0.0"
)

# CORS (Cross-Origin Resource Sharing) settings
# Allows requests from your Next.js frontend (running on a different port during development)
origins = [
    "http://localhost:3000",  # Default Next.js dev port
    # Add any other origins if needed (e.g., your deployed frontend URL)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the prediction API router
app.include_router(prediction_endpoints.router, prefix="/api/v1", tags=["Prediction"]) # Corrected router name

@app.get("/health", tags=["Health Check"])
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    # Host 0.0.0.0 makes it accessible on the network, not just localhost
    # Reload=True is good for development, disable for production
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)