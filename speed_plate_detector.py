import argparse
import logging
import cv2
import numpy as np
import os
import asyncio
import datetime
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import List, Optional, Dict
from ultralytics import YOLO
import supervision as sv
from deep_sort_realtime.deepsort_tracker import DeepSort
import sys

# Add the necessary path to import modules from matricule/src/prod
sys.path.append(os.path.abspath('./matricule/matricule/src/prod'))

from GeminiOcr import GeminiOCR
from Utils import Utils
from Filter import ImageProcessor, LicensePlateDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the speed limit (km/h)
SPEED_LIMIT = 50  # Adjust this value as needed

# Define virtual detection line (y-coordinate in the frame where speed is measured)




@dataclass
class ViolationRecord:
    """Class for keeping track of speed violations."""
    vehicle_id: int
    speed: float
    timestamp: str
    license_plate: Optional[str] = None
    vehicle_image_path: Optional[str] = None
    plate_image_path: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "vehicle_id": self.vehicle_id,
            "speed": self.speed,
            "timestamp": self.timestamp,
            "license_plate": self.license_plate,
            "vehicle_image_path": self.vehicle_image_path,
            "plate_image_path": self.plate_image_path
        }


# SOURCE = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])

SOURCE = np.array([[500, 1300], [3500, 1300], [5039, 2159], [-550, 2159]])

TARGET_WIDTH = 28
TARGET_HEIGHT = 50
TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)



class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vehicle Speed Detection with License Plate Recognition"
    )
    parser.add_argument(
        "--source_video_path",
        required=False,
        help="Path to the source video file",
        type=str,
        default="../data/vehicles.mp4"
    )
    parser.add_argument(
        "--target_video_path",
        required=False,
        help="Path to the target video file (output)",
        type=str,
        default="../output/speed_violations.mp4"
    )
    parser.add_argument(
        "--confidence_threshold",
        default=0.3,
        help="Confidence threshold for the model",
        type=float,
    )
    parser.add_argument(
        "--iou_threshold", 
        default=0.7, 
        help="IOU threshold for the model", 
        type=float
    )
    parser.add_argument(
        "--speed_limit",
        default=SPEED_LIMIT,
        help="Speed limit in km/h",
        type=int
    )    
    parser.add_argument(
        "--api_key",
        help="API key for Gemini OCR",
        type=str,
        default=""
    )    
    parser.add_argument(
        "--output_file",
        help="Path to save the violation records (JSON format)",
        type=str,
        default="speed_violations/violations_report.json"
    )

    return parser.parse_args()


async def main():
    args = parse_arguments()
    
    # Initialize the OCR model
    ocr = GeminiOCR(api_key=args.api_key, model_name="gemini-2.5-flash-preview-04-17")
    
    # Initialize list to store all violation records
    violation_records: List[ViolationRecord] = []

    # Create output directory for violation images
    violation_dir = os.path.join("speed_violations", "violations")
    os.makedirs(violation_dir, exist_ok=True)

    # Create or load existing JSON file
    output_path = args.output_file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load existing violations if file exists
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r') as json_file:
                existing_records = json.load(json_file)
                violation_records = [ViolationRecord(**record) for record in existing_records]
                logger.info(f"Loaded {len(violation_records)} existing violation records")
        except Exception as e:
            logger.error(f"Error loading existing violation records: {e}")
            violation_records = []

    def save_violation_record(record: ViolationRecord):
        """Save a single violation record to JSON file"""
        try:
            # Convert to dictionary
            record_dict = record.to_dict()
            logger.info(f"Attempting to save record for vehicle {record.vehicle_id} with data: {record_dict}")
            
            # Load existing records
            if os.path.exists(output_path):
                with open(output_path, 'r') as json_file:
                    records = json.load(json_file)
                    logger.info(f"Loaded {len(records)} existing records from {output_path}")
            else:
                records = []
                logger.info(f"No existing records file found at {output_path}, creating new one")
            
            # Check if record for this vehicle_id already exists
            existing_index = next((i for i, r in enumerate(records) if r["vehicle_id"] == record.vehicle_id), None)
            
            if existing_index is not None:
                # Update existing record
                logger.info(f"Updating existing record at index {existing_index} for vehicle {record.vehicle_id}")
                records[existing_index] = record_dict
            else:
                # Add new record
                logger.info(f"Adding new record for vehicle {record.vehicle_id}")
                records.append(record_dict)
            
            # Save updated records
            with open(output_path, 'w') as json_file:
                json.dump(records, json_file, indent=4)
                
            # Verify the save was successful
            with open(output_path, 'r') as json_file:
                saved_records = json.load(json_file)
                saved_record = next((r for r in saved_records if r["vehicle_id"] == record.vehicle_id), None)
                if saved_record is None:
                    raise Exception(f"Failed to verify save for vehicle {record.vehicle_id}")
                
            logger.info(f"Successfully saved and verified violation record for vehicle {record.vehicle_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving violation record for vehicle {record.vehicle_id}: {str(e)}")
            logger.exception("Full traceback:")
            return False

    # Initialize YOLO models for vehicle detection and license plate detection
    vehicle_model = YOLO("../yolo11m.pt")  # Adjust path as needed
    lpdetector = LicensePlateDetector("../matricule/matricule/models/best.pt")  # Adjust path as needed

    # Initialize video info and processing components
    video_info = sv.VideoInfo.from_video_path(video_path=args.source_video_path)
    
    byte_track = sv.ByteTrack(
        frame_rate=video_info.fps, track_activation_threshold=args.confidence_threshold
    )

    # Initialize DeepSort tracker for deeper tracking analysis if needed
    deep_tracker = DeepSort(
        max_age=30,
        n_init=3,
        max_iou_distance=0.7,
        max_cosine_distance=0.3,
        nn_budget=100,
    )

    # Initialize annotation components
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)

    colors = ("#007fff", "#0072e6", "#0066cc", "#0059b3", "#004c99", "#ff0000", "#d10000", "#00264d")
    color_palette = sv.ColorPalette(list(map(sv.Color.from_hex, colors)))

    box_annotator = sv.BoxAnnotator(
        color=color_palette,
        thickness=2,
        color_lookup=sv.ColorLookup.TRACK,
    )
    
    label_annotator = sv.RichLabelAnnotator(
        color=color_palette,
        border_radius=6,
        font_size=32,
        color_lookup=sv.ColorLookup.TRACK,
        text_padding=12,
    )
    
    trace_annotator = sv.TraceAnnotator(
        color=color_palette,
        position=sv.Position.CENTER,
        thickness=2,
        trace_length=video_info.fps,
        color_lookup=sv.ColorLookup.TRACK,
    )

    frame_generator = sv.get_video_frames_generator(source_path=args.source_video_path)

    polygon_zone = sv.PolygonZone(polygon=SOURCE)
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    # Dictionaries to store tracked data
    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))
    speeds = {}
    speed_violators = set()
    sharpness = {}
    processed_violators = set()
    crossed_line = set()  # Track vehicles that have crossed the detection line

    utils = Utils()
    processor = ImageProcessor()
    logger.info("Speed violation detection initialized")
    
    try:
    # Create video writer for output
        with sv.VideoSink(args.target_video_path, video_info) as sink:
            for frame in frame_generator:
                # Draw the ROI (polygon zone) in red on the frame
                cv2.polylines(frame, [SOURCE.astype(int)], isClosed=True, color=(0, 0, 255), thickness=2)

                # Detect vehicles with YOLO
                result = vehicle_model(frame)[0]
                detections = sv.Detections.from_ultralytics(result)
                detections = detections[detections.confidence > args.confidence_threshold]
                detections = detections[polygon_zone.trigger(detections)]
                detections = detections.with_nms(threshold=args.iou_threshold)
                
                # Track detections with ByteTrack
                detections = byte_track.update_with_detections(detections=detections)

                # Calculate points for speed estimation
                points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
                points = view_transformer.transform_points(points=points).astype(int)

                # Update coordinates for each tracked object
                for tracker_id, [_, y] in zip(detections.tracker_id, points):
                    coordinates[tracker_id].append(y)

                labels = []
                
                # Calculate speeds and check for violations
                current_violators = set()  # Track violators in current frame
                for tracker_id in detections.tracker_id:
                    if len(coordinates[tracker_id]) < video_info.fps / 2:
                        labels.append(f"#{tracker_id}")
                    else:                    
                        coordinate_start = coordinates[tracker_id][-1]
                        coordinate_end = coordinates[tracker_id][0]
                        distance = abs(coordinate_start - coordinate_end)
                        time = len(coordinates[tracker_id]) / video_info.fps
                        speed = distance / time * 3.6  # Convert to km/h
                        speeds[tracker_id] = speed
                        
                        speed_text = f"#{tracker_id} {int(speed)} km/h"

                        if tracker_id:
                            crossed_line.add(tracker_id)
                            # If vehicle is speeding, add to current violators
                            if speed > args.speed_limit:
                                speed_text += f" VIOLATION!"
                                current_violators.add(tracker_id)
                                if tracker_id not in speed_violators:
                                    speed_violators.add(tracker_id)
                        
                        labels.append(speed_text)

                # Process all current violators
                for tracker_id in current_violators:
                    # Find the detection index for this tracker_id
                    det_idx = list(detections.tracker_id).index(tracker_id)
                    bbox = detections.xyxy[det_idx].astype(int)
                    x1, y1, x2, y2 = bbox
                    vehicle_image = frame[y1:y2, x1:x2]
                    
                    if vehicle_image.size > 0:
                        try:
                            # Create or get violation record
                            violation_record = None
                            for record in violation_records:
                                if record.vehicle_id == tracker_id:
                                    violation_record = record
                                    break
                            
                            if violation_record is None:
                                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                violation_record = ViolationRecord(
                                    vehicle_id=int(tracker_id),
                                    speed=float(speeds[tracker_id]),
                                    timestamp=timestamp
                                )
                                violation_records.append(violation_record)
                            
                            detection_success, plate_image, coord_plate = lpdetector.detect_plate(vehicle_image)
                            if detection_success and plate_image is not None:
                                # Calculate quality metrics for the plate image
                                metrics = processor.calculate_quality_metrics(plate_image)
                                current_score = metrics.total_score
                                
                                # Check if we should update the images (if no images exist or if current image is better quality)
                                should_update = False
                                if tracker_id not in sharpness:
                                    should_update = True
                                elif current_score > sharpness[tracker_id]["score"]:
                                    should_update = True
                                    # Remove previous images if they exist
                                    try:
                                        if os.path.exists(sharpness[tracker_id]["image_path"]):
                                            os.remove(sharpness[tracker_id]["image_path"])
                                        if os.path.exists(sharpness[tracker_id]["plate_path"]):
                                            os.remove(sharpness[tracker_id]["plate_path"])
                                    except (FileNotFoundError, OSError) as e:
                                        logger.error(f"Error deleting old images: {e}")
                                
                                if should_update:
                                    # Save both vehicle and plate images with unique filenames
                                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                    
                                    # Save vehicle image with relative path
                                    vehicle_filename = os.path.join("speed_violations", "violations", f"vehicle_{tracker_id}.jpg")
                                    cv2.imwrite(vehicle_filename, vehicle_image)
                                    
                                    # Save plate image with relative path
                                    plate_filename = os.path.join("speed_violations", "violations", f"plate_{tracker_id}.jpg")
                                    cv2.imwrite(plate_filename, plate_image)
                                    
                                    # Store information about saved images
                                    sharpness[tracker_id] = {
                                        "score": current_score,
                                        "image_path": vehicle_filename,
                                        "plate_path": plate_filename,
                                        "ocr_processed": False,
                                        "speed": int(speeds[tracker_id]),
                                        "last_update": timestamp
                                    }
                                    
                                    # Update record with image paths
                                    violation_record.vehicle_image_path = vehicle_filename
                                    violation_record.plate_image_path = plate_filename
                                    
                                    # Save the record to JSON and verify it was saved
                                    if not save_violation_record(violation_record):
                                        logger.error(f"Failed to save record for vehicle {tracker_id} after saving images")
                                        # Try to save again after a short delay
                                        await asyncio.sleep(0.1)
                                        if not save_violation_record(violation_record):
                                            logger.error(f"Second attempt to save record for vehicle {tracker_id} also failed")
                                    
                                    # Process OCR immediately
                                    plate_number = await process_plate_ocr_immediate(ocr, tracker_id, plate_image, 
                                        plate_filename, violation_records)
                                    if plate_number:
                                        # Update existing record with plate number
                                        for record in violation_records:
                                            if record.vehicle_id == tracker_id:
                                                logger.info(f"Updating record for vehicle {tracker_id} with plate number {plate_number}")
                                                record.license_plate = plate_number
                                                if not save_violation_record(record):
                                                    logger.error(f"Failed to save record for vehicle {tracker_id} after updating plate number")
                                                break
                                        
                                        # Print violation summary in real-time
                                        print("\n==== New Speed Violation Detected ====")
                                        plate_info = f"License Plate: {plate_number}" if plate_number else "No plate detected"
                                        print(f"Vehicle ID: {tracker_id}")
                                        print(f"Speed: {speeds[tracker_id]:.1f} km/h")
                                        print(f"{plate_info}")
                                        print(f"Total violations so far: {len(violation_records)}")
                                        print("===============================")
                        except Exception as e:
                            logger.error(f"Error processing license plate for track {tracker_id}: {e}")

                # Annotate frame with detection information
                annotated_frame = frame.copy()
                annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
                annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
                annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
                
                # Write annotated frame to output video
                sink.write_frame(annotated_frame)
                
                # Display frame if GUI support is available
                try:
                    resized_frame = cv2.resize(annotated_frame, (1280, 720))
                    cv2.imshow("Speed Violation Detector", resized_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    # Skip display if OpenCV was built without GUI support
                        pass

    except Exception as e:
        logger.error(f"Error during video processing: {e}")
        raise
    finally:
        # Cleanup
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
                
                # Verify the output video was created
                if os.path.exists(args.target_video_path):
                    logger.info(f"Successfully saved output video to {args.target_video_path}")
                else:
                    logger.error(f"Failed to save output video to {args.target_video_path}")
                
            # Print final summary
            print("\n==== Final Speed Violation Summary ====")
            print(f"Total violations detected: {len(violation_records)}")
            for idx, record in enumerate(violation_records, 1):
                plate_info = f"License Plate: {record.license_plate}" if record.license_plate else "No plate detected"
                print(f"{idx}. Vehicle ID: {record.vehicle_id}, Speed: {record.speed:.1f} km/h, {plate_info}")
            print("===============================")
                
    return violation_records




async def process_plate_ocr_immediate(ocr, track_id, plate_image, plate_path, violation_records=None):
    """Process license plate with OCR immediately and return the plate number"""
    try:
        # Save the plate image first to ensure it exists
        cv2.imwrite(plate_path, plate_image)
        
        # First attempt: Extract text from plate image
        plate_number = ocr.extract_text_from_image(image_path=plate_path)
        
        # If plate OCR fails or returns no valid text, try OCR on the vehicle image
        if not plate_number or plate_number == "NO_PLATE_DETECTED" :
            # Get the vehicle image path from the violation records
            vehicle_path = None
            if violation_records:
                for record in violation_records:
                    if record.vehicle_id == track_id and record.vehicle_image_path:
                        vehicle_path = record.vehicle_image_path
                        break
            
            if vehicle_path and os.path.exists(vehicle_path):
                logger.info(f"Plate OCR failed for track {track_id}, attempting OCR on vehicle image")
                # Try OCR on the vehicle image
                plate_number = ocr.extract_text_from_image(image_path=vehicle_path)
        
        if plate_number and isinstance(plate_number, str):
            # Clean up the plate number (remove extra spaces, etc.)
            plate_number = plate_number.strip()
            logger.info(f"License plate detected for violator #{track_id}: {plate_number}")
            
            # Update the violation record if available
            if violation_records:
                for record in violation_records:
                    if record.vehicle_id == track_id:
                        record.license_plate = plate_number
                        break
            
            # Return the extracted plate number
            return plate_number
        else:
            logger.warning(f"No valid plate number detected for track {track_id} from either plate or vehicle image")
            return None
            
    except Exception as e:
        logger.error(f"Error in immediate OCR processing for track {track_id}: {e}")
        return None


if __name__ == "__main__":
    asyncio.run(main())
