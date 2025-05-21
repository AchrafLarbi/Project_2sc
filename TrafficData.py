import numpy as np
import cv2
import argparse
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict, deque
import time
from datetime import datetime

SOURCE = np.array(
    [
    [739, 275],
    [895, 275],
    [786, 572],
    [213, 551]
])

TARGET_WIDTH = 15
TARGET_HEIGHT = 200

TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)

# Constants for traffic flow analysis
LANE_COUNT = 2  # Number of lanes in the monitored area
ROAD_LENGTH_METERS = 50  # Approximate length of the monitored road segment in meters
TIME_INTERVAL_SECONDS = 60  # Time interval for vehicle counting (1 minute)
PIXEL_TO_METER_RATIO = TARGET_HEIGHT / ROAD_LENGTH_METERS  # Conversion ratio

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

# Add this function to assign vehicles to specific lanes
def assign_to_lane(point, source_polygon, num_lanes):
    """
    Assign a vehicle to a specific lane based on its position.
    
    Args:
        point: The (x, y) coordinates of the vehicle's bottom center
        source_polygon: The ROI polygon coordinates
        num_lanes: Number of lanes to divide the ROI into
        
    Returns:
        lane_index: The lane number (0 to num_lanes-1)
    """
    # Extract x, y from point
    x, y = point
    
    # Get the top and bottom lines of the ROI
    top_left, top_right = source_polygon[0], source_polygon[1]
    bottom_left, bottom_right = source_polygon[3], source_polygon[2]
    
    # Calculate lane width at the y-position of the point
    # This handles perspective where lanes may be wider at the bottom than at the top
    y_ratio = (y - top_left[1]) / (bottom_left[1] - top_left[1]) if (bottom_left[1] - top_left[1]) != 0 else 0
    
    # Interpolate the left and right boundaries at the point's y-position
    left_x = top_left[0] + y_ratio * (bottom_left[0] - top_left[0])
    right_x = top_right[0] + y_ratio * (bottom_right[0] - top_right[0])
    
    # Calculate the total width at this y-position
    total_width = right_x - left_x
    
    # Calculate the relative position of the point within this width
    rel_pos = (x - left_x) / total_width if total_width != 0 else 0
    
    # Ensure rel_pos is between 0 and 1
    rel_pos = max(0, min(1, rel_pos))
    
    # Determine lane index (0 to num_lanes-1)
    lane_index = min(int(rel_pos * num_lanes), num_lanes - 1)
    
    return lane_index


def determine_traffic_status(metrics):
    """
    Determine traffic status based on multiple metrics.
    
    Args:
        metrics: Dictionary containing traffic flow metrics
        
    Returns:
        status: String describing the traffic status
        status_color: BGR color tuple for the status
    """
    # Extract metrics
    avg_speed = metrics['average_speed_kmh']
    density = metrics['vehicle_density_vpkm']
    vehicle_count = metrics['vehicle_count_current']
    flow_rate = metrics['flow_rate_vph']
    lane_occupancy = metrics['lane_occupancy_percentage']
    
    # Calculate average lane occupancy
    avg_occupancy = sum(lane_occupancy.values()) / len(lane_occupancy) if lane_occupancy else 0
    
    # Calculate lane imbalance (how unevenly distributed traffic is across lanes)
    lane_values = list(lane_occupancy.values())
    lane_imbalance = max(lane_values) - min(lane_values) if lane_values else 0
    
    # Check for lane blockage (if any lane is significantly more occupied than others)
    lane_blockage = any(occ > 70 for occ in lane_occupancy.values())
    
    # Initialize score components (higher is worse)
    speed_score = 0
    density_score = 0
    occupancy_score = 0
    imbalance_score = 0
    
    # Speed scoring (lower speeds = higher score)
    if avg_speed < 20:
        speed_score = 5     # Very slow
    elif avg_speed < 40:
        speed_score = 3     # Slow
    elif avg_speed < 60:
        speed_score = 1     # Moderate
    else:
        speed_score = 0     # Free-flowing
    
    # Density scoring (higher density = higher score)
    if density > 40:
        density_score = 5   # Very high density
    elif density > 25:
        density_score = 4   # High density
    elif density > 15:
        density_score = 2   # Moderate density
    elif density > 5:
        density_score = 1   # Low density
    else:
        density_score = 0   # Very low density
    
    # Occupancy scoring (higher occupancy = higher score)
    if avg_occupancy > 80:
        occupancy_score = 5  # Very high occupancy
    elif avg_occupancy > 60:
        occupancy_score = 4  # High occupancy
    elif avg_occupancy > 40:
        occupancy_score = 2  # Moderate occupancy
    elif avg_occupancy > 20:
        occupancy_score = 1  # Low occupancy
    else:
        occupancy_score = 0  # Very low occupancy
    
    # Lane imbalance scoring
    if lane_imbalance > 50:
        imbalance_score = 3  # Severe imbalance
    elif lane_imbalance > 30:
        imbalance_score = 2  # Moderate imbalance
    elif lane_imbalance > 15:
        imbalance_score = 1  # Slight imbalance
    else:
        imbalance_score = 0  # Well balanced
    
    # Add bonus points for special conditions
    bonus = 0
    if lane_blockage:
        bonus += 3          # Lane blockage is a severe condition
    if flow_rate > 3600:
        bonus += 1          # Very high flow rate
    
    # Calculate total score
    total_score = speed_score + density_score + occupancy_score + imbalance_score + bonus
    
    # Determine status based on total score
    if total_score >= 12:
        status = "SEVERELY CONGESTED"
        status_color = (0, 0, 180)  # Dark red
    elif total_score >= 9:
        status = "CONGESTED"
        status_color = (0, 0, 255)  # Red
    elif total_score >= 6:
        status = "SLOW"
        status_color = (0, 165, 255)  # Orange
    elif total_score >= 3:
        status = "MODERATE"
        status_color = (0, 255, 255)  # Yellow
    elif total_score >= 1:
        status = "FLOWING"
        status_color = (0, 255, 0)  # Green
    else:
        status = "FREE FLOW"
        status_color = (0, 255, 0)  # Green
    
    # Check for special conditions that override the above status
    if lane_blockage and avg_speed < 30:
        status = "LANE BLOCKAGE"
        status_color = (0, 0, 200)  # Dark red
    elif lane_imbalance > 50 and avg_speed < 40:
        status = "UNBALANCED FLOW"
        status_color = (0, 165, 255)  # Orange
    
    # Add detailed information to the status
    details = []
    if speed_score >= 3:
        details.append("slow speeds")
    if density_score >= 3:
        details.append("high density")
    if occupancy_score >= 3:
        details.append("high occupancy")
    if imbalance_score >= 2:
        details.append("lane imbalance")
    
    if details:
        detail_text = ", ".join(details)
        status += f" ({detail_text})"
    
    return status, status_color, total_score

    
# Update TrafficFlowAnalyzer class
class TrafficFlowAnalyzer:
    def __init__(self, lanes=LANE_COUNT, time_interval=TIME_INTERVAL_SECONDS, road_length=ROAD_LENGTH_METERS, source_polygon=SOURCE):
        self.lanes = lanes
        self.time_interval = time_interval  # seconds
        self.road_length = road_length  # meters
        self.source_polygon = source_polygon  # ROI polygon

        # Data structures for traffic metrics
        self.current_interval_start = time.time()
        self.vehicle_count_total = 0
        self.vehicle_count_interval = 0
        self.speeds = []
        
        # Track vehicles that have been counted
        self.tracked_vehicles = set()
        self.vehicles_in_frame = set()
        
        # For lane-specific tracking
        self.vehicles_in_lanes = [set() for _ in range(lanes)]
        self.lane_occupied_frames = [0] * lanes
        self.total_frames = 0
        
        # History of interval data
        self.interval_history = []
        
    def update(self, detections, frame_count=1):
        """Update traffic flow metrics with new detections"""
        current_time = time.time()
        self.total_frames += frame_count
        
        # Get current vehicle IDs in ROI
        current_ids = set(detections.tracker_id) if len(detections.tracker_id) > 0 else set()
        self.vehicles_in_frame = current_ids
        
        # Count new vehicles that have never been seen before
        new_vehicles = current_ids - self.tracked_vehicles
        if len(new_vehicles) > 0:
            self.vehicle_count_interval += len(new_vehicles)
            self.vehicle_count_total += len(new_vehicles)
            # Add newly detected vehicles to tracked set
            self.tracked_vehicles.update(new_vehicles)
        
        # Reset vehicles in lanes for this frame
        self.vehicles_in_lanes = [set() for _ in range(self.lanes)]
        
        # Assign vehicles to lanes
        for i, tracker_id in enumerate(detections.tracker_id):
            if i < len(detections.xyxy):
                box = detections.xyxy[i]
                # Get bottom center of the bounding box
                bottom_center_x = (box[0] + box[2]) / 2
                bottom_center_y = box[3]  # Bottom y-coordinate
                
                # Assign to lane
                lane_index = assign_to_lane(
                    (bottom_center_x, bottom_center_y), 
                    self.source_polygon, 
                    self.lanes
                )
                
                # Add vehicle to the appropriate lane
                self.vehicles_in_lanes[lane_index].add(tracker_id)
        
        # Update lane occupancy
        for lane_idx in range(self.lanes):
            if len(self.vehicles_in_lanes[lane_idx]) > 0:
                self.lane_occupied_frames[lane_idx] += 1
        
        # Record speeds of all detected vehicles
        for i, tracker_id in enumerate(detections.tracker_id):
            if hasattr(detections, 'speed') and i < len(detections.speed):
                speed = detections.speed[i]
                if speed > 0:
                    self.speeds.append(speed)
        
        # Check if time interval has elapsed
        if current_time - self.current_interval_start >= self.time_interval:
            # Calculate metrics for the completed interval
            interval_data = self.calculate_interval_metrics()
            
            # Store the data
            self.interval_history.append(interval_data)
            
            # Reset interval counters
            self.current_interval_start = current_time
            self.vehicle_count_interval = 0
            self.speeds = []
            self.lane_occupied_frames = [0] * self.lanes
            self.total_frames = 0
        
        return self.get_current_metrics()
    
    def calculate_interval_metrics(self):
        """Calculate metrics for the completed time interval"""
        # Average speed calculation
        avg_speed = sum(self.speeds) / len(self.speeds) if self.speeds else 0
        
        # Vehicle density (vehicles per km per lane)
        density = (len(self.vehicles_in_frame) / self.lanes) / (self.road_length / 1000)
        
        # Lane occupancy percentages
        lane_occupancy = {}
        for lane in range(self.lanes):
            occupancy_percent = (self.lane_occupied_frames[lane] / self.total_frames) * 100 if self.total_frames > 0 else 0
            lane_occupancy[f"lane_{lane+1}"] = occupancy_percent
        
        # Flow rate (vehicles per hour) - based on actual count during interval
        flow_rate = (self.vehicle_count_interval / self.time_interval) * 3600 # *3600 to do intervel to hour
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return {
            "timestamp": timestamp,
            "vehicle_count": self.vehicle_count_interval,
            "flow_rate_vph": flow_rate,
            "average_speed_kmh": avg_speed,
            "vehicle_density_vpkm": density,
            "lane_occupancy_percentage": lane_occupancy
        }
    
    def get_current_metrics(self):
        """Get the current metrics for display"""
        # Calculate current metrics without resetting counters
        avg_speed = sum(self.speeds) / len(self.speeds) if self.speeds else 0
        
        # Current density based on current frame
        density = (len(self.vehicles_in_frame) / self.lanes) / (self.road_length / 1000)
        
        # Lane occupancy
        lane_occupancy = {}
        vehicles_per_lane = {}
        for lane in range(self.lanes):
            occupancy_percent = (self.lane_occupied_frames[lane] / self.total_frames) * 100 if self.total_frames > 0 else 0
            lane_occupancy[f"lane_{lane+1}"] = occupancy_percent
            vehicles_per_lane[f"lane_{lane+1}"] = len(self.vehicles_in_lanes[lane])
        
        # Calculate elapsed time properly
        elapsed_time = time.time() - self.current_interval_start
        
        # Calculate flow rate based on actual count during the current interval
        # Avoid division by zero
        flow_rate = (self.vehicle_count_interval / elapsed_time) * 3600 if elapsed_time > 0 else 0
        
        # For very short elapsed times, flow rate can be unrealistically high
        # Apply a reasonable cap if needed
        if flow_rate > 7200:  # Cap at 7200 vehicles per hour (2 per second)
            flow_rate = 7200
        
        return {
            "vehicle_count_current": len(self.vehicles_in_frame),  # Current vehicles in frame
            "vehicle_count_interval": self.vehicle_count_interval,  # Vehicles counted in this interval
            "vehicle_count_total": self.vehicle_count_total,  # Total unique vehicles counted
            "flow_rate_vph": flow_rate,
            "average_speed_kmh": avg_speed,
            "vehicle_density_vpkm": density,
            "lane_occupancy_percentage": lane_occupancy,
            "vehicles_per_lane": vehicles_per_lane,
            "interval_elapsed": elapsed_time,
            "interval_total": self.time_interval
        }

    def export_data(self, filename="traffic_flow_data.csv"):
        import csv
        
        if not self.interval_history:
            return False
        
        try:
            with open(filename, 'w', newline='') as csvfile:
                # Determine all possible fields by examining the first record
                first_record = self.interval_history[0]
                fieldnames = ["timestamp", "vehicle_count", "flow_rate_vph", "average_speed_kmh", "vehicle_density_vpkm"]
                
                # Add lane occupancy fields
                lane_occupancy = first_record.get("lane_occupancy_percentage", {})
                for lane_key in lane_occupancy.keys():
                    fieldnames.append(f"occupancy_{lane_key}")
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for interval_data in self.interval_history:
                    # Flatten the lane occupancy data
                    row_data = {k: v for k, v in interval_data.items() if k != "lane_occupancy_percentage"}
                    for lane_key, value in interval_data.get("lane_occupancy_percentage", {}).items():
                        row_data[f"occupancy_{lane_key}"] = value
                    
                    writer.writerow(row_data)
                    
            return True
        except Exception as e:
            print(f"Error exporting data: {e}")
            return False
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vehicle Speed Estimation using Ultralytics and Supervision with Traffic Flow Analysis"
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
        default=0.7, 
        help="IOU threshold for the model", 
        type=float
    )
    parser.add_argument(
        "--time_interval", 
        default=60, 
        help="Time interval for traffic flow analysis in seconds", 
        type=int
    )
    parser.add_argument(
        "--lanes", 
        default=2, 
        help="Number of lanes in the monitored area", 
        type=int
    )
    parser.add_argument(
        "--road_length", 
        default=50, 
        help="Length of the monitored road segment in meters", 
        type=float
    )
    parser.add_argument(
        "--export_data", 
        action="store_true", 
        help="Export traffic flow data to CSV after processing"
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
    speeds = defaultdict(float)  # Store calculated speeds for each tracked vehicle

    # Initialize traffic flow analyzer with source polygon for lane detection
    traffic_analyzer = TrafficFlowAnalyzer(
        lanes=args.lanes,
        time_interval=args.time_interval,
        road_length=args.road_length,
        source_polygon=SOURCE
    )

    # Set the desired fixed window size
    fixed_width = 1920
    fixed_height = 1080

    # Create a named window with a fixed size
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("frame", fixed_width, fixed_height)

    # For FPS calculation
    start_time = time.time()
    frame_count = 0

    # Define lane colors for visualization
    lane_colors = [
        (0, 255, 0),   # Green
        (255, 255, 0),  # Yellow
        (0, 255, 255),  # Cyan
        (255, 0, 255),  # Magenta
        (128, 0, 255),  # Purple
    ]
    
    # Ensure we have enough colors for all lanes
    while len(lane_colors) < args.lanes:
        lane_colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

    with sv.VideoSink(args.target_video_path, video_info) as sink:
        for frame in frame_generator:
            frame_count += 1
            # Draw the ROI (polygon zone) in red on the frame
            cv2.polylines(frame, [SOURCE.astype(int)], isClosed=True, color=(0, 0, 255), thickness=2)

            # For visualizing lanes (dividing ROI into equal parts)
            h, w = frame.shape[:2]
            
            # Draw lane dividers and labels
            for i in range(args.lanes + 1):
                if i > 0 and i < args.lanes:
                    # Draw lane dividers
                    pt1 = tuple(SOURCE[0].astype(int) + (SOURCE[1] - SOURCE[0]) * i // args.lanes)
                    pt2 = tuple(SOURCE[3].astype(int) + (SOURCE[2] - SOURCE[3]) * i // args.lanes)
                    cv2.line(frame, pt1, pt2, (255, 255, 255), 2)
                
                # Label each lane
                if i < args.lanes:
                    # Find the center point of each lane for labeling
                    center_top = tuple((SOURCE[0].astype(int) + (SOURCE[1] - SOURCE[0]) * (i + 0.5) // args.lanes).astype(int))
                    center_bottom = tuple((SOURCE[3].astype(int) + (SOURCE[2] - SOURCE[3]) * (i + 0.5) // args.lanes).astype(int))
                    
                    # Calculate the midpoint for placing the lane label
                    label_point = ((center_top[0] + center_bottom[0]) // 2, (center_top[1] + center_bottom[1]) // 2)
                    
                    # Draw lane number
                    cv2.putText(frame, f"Lane {i+1}", label_point, cv2.FONT_HERSHEY_SIMPLEX, 
                                0.7, lane_colors[i], 2)

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

            # Store speeds for each tracked vehicle
            detections.speed = []
            labels = []
            lane_assignments = []
            
            for idx, (tracker_id, [_, y]) in enumerate(zip(detections.tracker_id, points)):
                coordinates[tracker_id].append(y)
                
                # Get bottom center of the bounding box for lane assignment
                if idx < len(detections.xyxy):
                    box = detections.xyxy[idx]
                    bottom_center_x = (box[0] + box[2]) / 2
                    bottom_center_y = box[3]  # Bottom y-coordinate
                    
                    # Assign to lane
                    lane_index = assign_to_lane(
                        (bottom_center_x, bottom_center_y), 
                        SOURCE, 
                        args.lanes
                    )
                    lane_assignments.append(lane_index)
                else:
                    lane_assignments.append(0)  # Default to lane 0 if no box
                
                if len(coordinates[tracker_id]) < video_info.fps / 2:
                    labels.append(f"#{tracker_id} (Lane {lane_assignments[-1]+1})")
                    detections.speed.append(0)
                else:
                    coordinate_start = coordinates[tracker_id][-1]
                    coordinate_end = coordinates[tracker_id][0]
                    distance = abs(coordinate_start - coordinate_end)
                    time_elapsed = len(coordinates[tracker_id]) / video_info.fps
                    speed = distance / time_elapsed * 3.6
                    speeds[tracker_id] = speed
                    labels.append(f"#{tracker_id} {int(speed)} km/h (Lane {lane_assignments[-1]+1})")
                    detections.speed.append(speed)

            # Update traffic flow metrics
            traffic_metrics = traffic_analyzer.update(detections)

            # Create traffic flow metrics display
            metrics_text = [
                f"Vehicles: {traffic_metrics['vehicle_count_total']} total, {traffic_metrics['vehicle_count_current']} current",
                f"Flow Rate: {traffic_metrics['flow_rate_vph']:.1f} veh/hour",
                f"Avg Speed: {traffic_metrics['average_speed_kmh']:.1f} km/h",
                f"Density: {traffic_metrics['vehicle_density_vpkm']:.2f} veh/km/lane",
                f"Time: {int(traffic_metrics['interval_elapsed'])}s / {traffic_metrics['interval_total']}s"
            ]
            
            # Add lane occupancy to metrics display with lane-specific colors
            y_offset = 30
            for i, (lane, occupancy) in enumerate(traffic_metrics['lane_occupancy_percentage'].items()):
                lane_text = f"{lane.replace('_', ' ').title()}: {occupancy:.1f}% occupied, {traffic_metrics['vehicles_per_lane'][lane]} vehicles"
                cv2.putText(frame, lane_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, lane_colors[i], 2)
                y_offset += 25

            # Add remaining metrics
            for text in metrics_text:
                cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 25

            # Calculate current FPS
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25

            # Draw traffic flow status (simplified)
            avg_speed = traffic_metrics['average_speed_kmh']
            density = traffic_metrics['vehicle_density_vpkm']
            
             # Determine traffic status using our enhanced function
            status, status_color, status_score = determine_traffic_status(traffic_metrics)
            
            # Display traffic status with score
            cv2.putText(frame, f"Traffic Status: {status}", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            y_offset += 30
            

            # Highlight bounding boxes with lane-specific colors
            for i, lane_idx in enumerate(lane_assignments):
                if i < len(detections.xyxy):
                    box = detections.xyxy[i].astype(int)
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), lane_colors[lane_idx], 2)

            annotated_frame = frame.copy()
            annotated_frame = trace_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
            
            # We'll skip the default box annotator since we're drawing lane-colored boxes
            # Instead we'll just add the labels
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
        
        # Export traffic data if requested
        if args.export_data:
            export_filename = f"traffic_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            if traffic_analyzer.export_data(export_filename):
                print(f"Traffic flow data exported to {export_filename}")
            else:
                print("No traffic flow data to export or export failed")

# python TrafficData.py --source_video_path data/highway.mp4 --target_video_path data/highway-result2.mp4 --lanes 3 --time_interval 60 --road_length 15 --export_data 