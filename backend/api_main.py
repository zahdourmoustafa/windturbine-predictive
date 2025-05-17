from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Assuming your router is in api_v1.endpoints.prediction_endpoints
from api_v1.endpoints import prediction_endpoints

app = FastAPI(
    title="Gearbox Fault Detection API",
    description="API for predicting gearbox faults from .mat sensor data files.",
    version="1.0.0"
)

# CORS (Cross-Origin Resource Sharing) settings
origins = [
    "http://localhost:3000",  # Default Next.js dev port
    # Add any other origins if needed (e.g., your deployed frontend URL)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Corrected
    allow_headers=["*"], # Corrected
)

# Include the prediction API router
app.include_router(prediction_endpoints.router, prefix="/api/v1", tags=["Prediction"])

@app.get("/api/v1/health", tags=["Health Check"])
async def health_check():
    return {"status": "healthy", "message": "API is running"}

if __name__ == "__main__":
    import uvicorn
    print("Starting Uvicorn server for Gearbox Fault Detection API...")
    # Host 0.0.0.0 makes it accessible on the network, not just localhost
    # Reload=True is good for development, disable for production
    uvicorn.run("api_main:app", host="0.0.0.0", port=8000, reload=True) 