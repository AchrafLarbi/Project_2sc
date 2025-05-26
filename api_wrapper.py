from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import asyncio
import tempfile
import os
import shutil
from typing import List, Optional
from pydantic import BaseModel
import json
import sys
import argparse
import logging
import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the current directory to Python path to import speed_plate_detector
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from speed_plate_detector import ViolationRecord, main as process_video, parse_arguments

# Create necessary directories
def ensure_directories():
    """Ensure all necessary directories exist"""
    directories = [
        "speed_violations",
        "speed_violations/violations",
        "speed_violations/output"
    ]
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Ensured directory exists: {directory}")
        except Exception as e:
            logger.error(f"Error creating directory {directory}: {e}")

# Create initial empty violations file if it doesn't exist
def ensure_violations_file():
    """Ensure the violations JSON file exists"""
    violations_file = "speed_violations/violations_report.json"
    if not os.path.exists(violations_file):
        try:
            with open(violations_file, 'w') as f:
                json.dump([], f)
        except Exception as e:
            logger.error(f"Error creating violations file: {e}")

# Initialize directories and files
ensure_directories()
ensure_violations_file()

app = FastAPI(
    title="Speed Plate Detection API",
    description="API for detecting speed violations and license plates from video streams",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the violations directory to serve images
app.mount("/violation-images", StaticFiles(directory="speed_violations/violations"), name="violation-images")

class ViolationResponse(BaseModel):
    vehicle_id: int
    speed: float
    timestamp: str
    license_plate: Optional[str] = None
    vehicle_image_path: Optional[str] = None
    plate_image_path: Optional[str] = None

@app.post("/process-video/", response_model=List[ViolationResponse])
async def process_video_endpoint(
    video: UploadFile = File(...),
    speed_limit: int = 50,
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.7
):
    """
    Process a video file to detect speed violations and license plates.
    
    Parameters:
    - video: The video file to process
    - speed_limit: Speed limit in km/h (default: 50)
    - confidence_threshold: Confidence threshold for detection (default: 0.3)
    - iou_threshold: IOU threshold for detection (default: 0.7)
    
    Returns:
    - List of violation records with vehicle information
    """
    try:
        # Create a temporary file to store the uploaded video
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            shutil.copyfileobj(video.file, temp_video)
            temp_video_path = temp_video.name

        # Ensure directories exist
        ensure_directories()
        
        # Create output video path with timestamp to avoid overwriting
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_video_path = os.path.join("speed_violations", "output", f"output_{timestamp}.mp4")
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        logger.info(f"Output video will be saved to: {output_video_path}")
        
        # Create a custom argument parser
        parser = argparse.ArgumentParser()
        parser.add_argument("--source_video_path", type=str, default=temp_video_path)
        parser.add_argument("--target_video_path", type=str, default=output_video_path)
        parser.add_argument("--confidence_threshold", type=float, default=confidence_threshold)
        parser.add_argument("--iou_threshold", type=float, default=iou_threshold)
        parser.add_argument("--speed_limit", type=int, default=speed_limit)
        parser.add_argument("--api_key", type=str, default="AIzaSyA-ey2zTapBuUvNCDVaR8YpXfymctSjJ5E")
        parser.add_argument("--output_file", type=str, default="speed_violations/violations_report.json")
        
        # Parse empty arguments to use defaults
        args = parser.parse_args([])
        
        # Override the default arguments
        args.source_video_path = temp_video_path
        args.target_video_path = output_video_path
        args.confidence_threshold = confidence_threshold
        args.iou_threshold = iou_threshold
        args.speed_limit = speed_limit
        
        # Set the args in the speed_plate_detector module
        import speed_plate_detector
        speed_plate_detector.args = args
        
        # Override the parse_arguments function to return our args
        def custom_parse_args():
            return args
        speed_plate_detector.parse_arguments = custom_parse_args
        
        # Process the video using the original main function
        violations = await process_video()
        
        # Clean up only the temporary input video file
        try:
            os.unlink(temp_video_path)
        except Exception as e:
            logger.error(f"Error cleaning up temporary input file: {e}")
        
        # Verify the output video was created
        if not os.path.exists(output_video_path):
            logger.error(f"Output video was not created at {output_video_path}")
            raise HTTPException(status_code=500, detail="Failed to create output video")
        
        logger.info(f"Successfully created output video at {output_video_path}")
        
        # Read violations from the JSON file
        violations_file = "speed_violations/violations_report.json"
        if os.path.exists(violations_file):
            with open(violations_file, 'r') as f:
                violations = json.load(f)
        else:
            violations = []
            # Create empty violations file
            with open(violations_file, 'w') as f:
                json.dump(violations, f)
        
        return violations

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/violations/", response_model=List[ViolationResponse])
async def get_violations():
    """
    Get all recorded violations.
    
    Returns:
    - List of all violation records
    """
    try:
        # Read violations from JSON file
        violations_file = "speed_violations/violations_report.json"
        if not os.path.exists(violations_file):
            # Create empty violations file
            with open(violations_file, 'w') as f:
                json.dump([], f)
            return []
            
        with open(violations_file, 'r') as f:
            violations = json.load(f)
            
        # Update image paths to use the new endpoint
        for violation in violations:
            if violation.get('vehicle_image_path'):
                violation['vehicle_image_path'] = violation['vehicle_image_path'].replace('speed_violations/violations/', '/violation-images/')
            if violation.get('plate_image_path'):
                violation['plate_image_path'] = violation['plate_image_path'].replace('speed_violations/violations/', '/violation-images/')
            
        return violations
    except Exception as e:
        logger.error(f"Error reading violations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/violations/{violation_id}/vehicle-image")
async def get_vehicle_image(violation_id: int):
    """
    Get the vehicle image for a specific violation.
    
    Parameters:
    - violation_id: The ID of the violation
    
    Returns:
    - The vehicle image file
    """
    try:
        image_path = f"speed_violations/violations/vehicle_{violation_id}.jpg"
        if not os.path.exists(image_path):
            # Return a default image or raise a more specific error
            raise HTTPException(
                status_code=404,
                detail=f"Vehicle image not found for violation ID {violation_id}"
            )
        return FileResponse(image_path)
    except Exception as e:
        logger.error(f"Error getting vehicle image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/violations/{violation_id}/plate-image")
async def get_plate_image(violation_id: int):
    """
    Get the license plate image for a specific violation.
    
    Parameters:
    - violation_id: The ID of the violation
    
    Returns:
    - The license plate image file
    """
    try:
        image_path = f"speed_violations/violations/plate_{violation_id}.jpg"
        if not os.path.exists(image_path):
            # Return a default image or raise a more specific error
            raise HTTPException(
                status_code=404,
                detail=f"Plate image not found for violation ID {violation_id}"
            )
        return FileResponse(image_path)
    except Exception as e:
        logger.error(f"Error getting plate image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 