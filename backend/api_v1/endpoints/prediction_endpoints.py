from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse

from api_v1.schemas.prediction_schemas import PredictionResponse, PredictionResult
from api_v1.services.prediction_service import get_prediction_results

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
async def predict_gearbox_fault(file: UploadFile = File(...) ):
    """
    Receives a .mat file, processes it using the pre-trained model, 
    and returns the fault diagnosis and sensor anomaly analysis.
    """
    if not file.filename.endswith('.mat'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .mat file.")

    try:
        file_content = await file.read()
        if not file_content:
            raise HTTPException(status_code=400, detail="File is empty.")

        # Call the service to process the file content
        raw_results = await get_prediction_results(file_content=file_content, filename=file.filename)
        
        if raw_results.get("error"):
            # If the service layer caught a processing error
            return PredictionResponse(
                success=False,
                message=f"Error processing file {file.filename}: {raw_results.get('error')}"
            )
        
        # Adapt raw_results to PredictionResult schema before wrapping in PredictionResponse
        # The schema will handle type conversions and validation for optional fields.
        prediction_data = PredictionResult(**raw_results)

        return PredictionResponse(
            success=True,
            data=prediction_data,
            message=f"Successfully processed file {file.filename}"
        )

    except HTTPException as http_exc: # Re-raise HTTPExceptions directly
        raise http_exc 
    except Exception as e:
        # Catch-all for unexpected errors during file handling or service call initiation
        # More specific error handling can be added as needed.
        # Log the error e for debugging
        print(f"Unhandled exception: {e}") # Replace with proper logging
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}") 