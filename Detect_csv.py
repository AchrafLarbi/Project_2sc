import argparse
import csv
import os
from collections import defaultdict, deque
from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

SOURCE = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])

TARGET_WIDTH = 25
TARGET_HEIGHT = 250

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
        description="Vehicle Speed Estimation using Ultralytics and Supervision"
    )
    parser.add_argument(
        "--source_video_path",
        required=True,
        help="Path to the source video file",
        type=str,
    )
    parser.add_argument(
        "--target_video_path",
        required=True,
        help="Path to the target video file (output)",
        type=str,
    )
    parser.add_argument(
        "--confidence_threshold",
        default=0.3,
        help="Confidence threshold for the model",
        type=float,
    )
    parser.add_argument(
        "--iou_threshold", 
        default=0.5, 
        help="IOU threshold for the model", 
        type=float
    )
    parser.add_argument(
        "--csv_output_path",
        default="speed_data.csv",
        help="Path to save the CSV file with speed data",
        type=str,
    )

    return parser.parse_args()

colors = ("#007fff", "#0072e6", "#0066cc", "#0059b3", "#004c99", "#004080", "#003366", "#00264d")
color_palette = sv.ColorPalette(list(map(sv.Color.from_hex, colors)))

if __name__ == "__main__":
    args = parse_arguments()

    video_info = sv.VideoInfo.from_video_path(video_path=args.source_video_path)
    model = YOLO("yolo11m.pt", task="detect")

    byte_track = sv.ByteTrack(
        frame_rate=video_info.fps, track_activation_threshold=args.confidence_threshold
    )

    thickness = sv.calculate_optimal_line_thickness(
        resolution_wh=video_info.resolution_wh
    )
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)

    box_annotator = sv.BoxAnnotator(
        color=color_palette,
        thickness=2,
        color_lookup=sv.ColorLookup.TRACK,
    )
    label_annotator = sv.RichLabelAnnotator(
        color=color_palette,
        border_radius=2,
        font_size=16,
        color_lookup=sv.ColorLookup.TRACK,
        text_padding=6,
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

    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))
    
    # Initialize data collection for CSV
    speed_data = []
    frame_count = 0
    
    # Set the desired fixed window size
    fixed_width = 1920
    fixed_height = 1080

    # Create a named window with a fixed size
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("frame", fixed_width, fixed_height)
    
    with sv.VideoSink(args.target_video_path, video_info) as sink:
        for frame in frame_generator:
            frame_count += 1
            timestamp = frame_count / video_info.fps
            
            # Draw the ROI (polygon zone) in red on the frame
            cv2.polylines(frame, [SOURCE.astype(int)], isClosed=True, color=(0, 0, 255), thickness=2)

            result = model(frame)[0]
            detections = sv.Detections.from_ultralytics(result)
            detections = detections[detections.confidence > args.confidence_threshold]
            detections = detections[polygon_zone.trigger(detections)]
            detections = detections.with_nms(threshold=args.iou_threshold)
            detections = byte_track.update_with_detections(detections=detections)

            points = detections.get_anchors_coordinates(
                anchor=sv.Position.BOTTOM_CENTER
            )
            points = view_transformer.transform_points(points=points).astype(int)

            for tracker_id, [_, y] in zip(detections.tracker_id, points):
                coordinates[tracker_id].append(y)

            labels = []
            
            for i, tracker_id in enumerate(detections.tracker_id):
                # Get vehicle class from the detection if available
                vehicle_class = "unknown"
                if hasattr(detections, 'class_id') and i < len(detections.class_id):
                    class_id = int(detections.class_id[i])
                    vehicle_class = model.model.names[class_id]
                
                if len(coordinates[tracker_id]) < video_info.fps / 2:
                    labels.append(f"#{tracker_id}")
                else:
                    coordinate_start = coordinates[tracker_id][-1]
                    coordinate_end = coordinates[tracker_id][0]
                    distance = abs(coordinate_start - coordinate_end)
                    time = len(coordinates[tracker_id]) / video_info.fps
                    speed = distance / time * 3.6
                    speed_int = int(speed)
                    
                    labels.append(f"#{tracker_id} {speed_int} km/h")
                    
                    # Save to our data collection
                    speed_data.append({
                        "tracker_id": tracker_id,
                        "speed_kmh": speed_int,
                        "frame_number": frame_count,
                        "timestamp": timestamp,
                        "vehicle_class": vehicle_class,
                        "date_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })

            annotated_frame = frame.copy()
            annotated_frame = trace_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
            annotated_frame = box_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels
            )

            sink.write_frame(annotated_frame)
            # Resize the frame to fit the fixed window size
            display_frame = cv2.resize(annotated_frame, (fixed_width, fixed_height))

            # Show the resized frame in the window
            cv2.imshow("frame", display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()
    
    # Save collected data to CSV
    if speed_data:
        with open(args.csv_output_path, 'w', newline='') as csvfile:
            fieldnames = speed_data[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(speed_data)
        
        print(f"Speed data saved to {args.csv_output_path}")

# python script_name.py --source_video_path input.mp4 --target_video_path output.mp4 --csv_output_path speed_data.csv
#python ultralytics_example.py  --source_video_path data/highway.mp4  --target_video_path data/highway-result.mp4  --confidence_threshold 0.3  --iou_threshold 0.5