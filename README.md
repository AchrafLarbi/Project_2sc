# Smart Traffic Management System

This project implements a comprehensive traffic management solution that includes vehicle detection, speed estimation, and an intelligent Variable Speed Limit (VSL) system using Reinforcement Learning.

## Table of Contents

- [Overview](#overview)
- [Components](#components)
  - [Vehicle Detection and Tracking](#vehicle-detection-and-tracking)
  - [Speed Estimation and Violation Detection](#speed-estimation-and-violation-detection)
  - [License Plate Recognition](#license-plate-recognition)
  - [Variable Speed Limit (VSL) System](#variable-speed-limit-vsl-system)
  - [Traffic Data Analysis](#traffic-data-analysis)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results and Validation](#results-and-validation)

## Overview

This project aims to create an intelligent traffic management system that can detect vehicles, estimate their speeds, identify speed violations, and dynamically adjust speed limits based on traffic conditions to optimize flow and safety. The system uses computer vision for vehicle detection, deep learning for license plate recognition, and reinforcement learning for dynamic speed limit control.

## Components

### Vehicle Detection and Tracking

Vehicle detection is implemented using YOLOv11, a state-of-the-art object detection model that provides accurate real-time detection:

- `ultralytics_example.py`: Example implementation of YOLOv11 for vehicle detection
- `detect.ipynb`: Jupyter notebook for vehicle detection experiments
- Pre-trained models: `yolo11m.pt`, `yolo11n.pt`, and `best.pt` (fine-tuned for our specific use case)
- Integration with DeepSort for reliable vehicle tracking across video frames

### Speed Estimation and Violation Detection

Speed estimation is performed by tracking vehicles across frames and calculating their movement using perspective transformation and time-based calculations:

- `speed_plate_detector.py`: Main implementation that:
  - Tracks vehicles using DeepSort algorithm
  - Calculates real-world distances using perspective transformation
  - Detects speed violations based on configurable thresholds
  - Logs violations with timestamps and vehicle metadata
- `get_point_roi.ipynb`: Notebook for defining regions of interest for speed estimation
- `Detect_csv.py`: Implementation for validation and CSV output generation
- Virtual detection lines for precise speed measurement at specific points

### License Plate Recognition

The system can detect and recognize license plates for violation tracking using advanced OCR:

- `PlateDetection/`: Directory containing license plate detection and OCR components:
  - `app.py`: Main application for plate detection
  - `GeminiOcr.py`: License plate text recognition using Google's Gemini API
  - `Filter.py`: Image processing for better plate recognition including perspective correction
  - `Utils.py`: Utility functions for image handling and data processing
- Integration with the violation detection system to store evidence of speed violations

### Variable Speed Limit (VSL) System

An intelligent VSL system based on Deep Q-Network (DQN) Reinforcement Learning that learns optimal speed limits for different traffic conditions:

- `RLmodel_DQN.ipynb`: Implementation of the RL agent for Variable Speed Limit control, including:
  - Environment definition modeling traffic conditions
  - Deep Q-Network agent with neural network architecture
  - Reward function balancing traffic flow, safety, stability, and lane utilization
  - Training pipeline with exploration-exploitation strategy
  - Model evaluation and visualization tools
- Trained models: `models/vsl_best_model.weights.h5` and `models/vsl_final_model.weights.h5`
- Experimentation with different network architectures (default, 4-layer with ReLU, 4-layer with tanh)

### Traffic Data Analysis

Comprehensive analysis of traffic conditions and VSL system performance:

- `traffic_data.csv`, `traffic_data2.csv`: Traffic data samples with vehicle counts, speeds, and flow rates
- `TrafficData.py`: Traffic data processing utilities for pre-processing and feature extraction
- Visualization outputs:
  - `traffic_condition_analysis.png`: Analysis of traffic flow patterns
  - `vsl_results.png`: VSL system performance results
  - `model_analysis.png`: Comparison of different DQN model architectures
  - `vehicle_count_vs_speed_analysis.png`: Relationship between traffic density and speed

## Installation

1. Clone the repository:

   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create a virtual environment:

   ```
   python -m venv myenv
   ```

3. Activate the virtual environment:

   ```
   # On Windows
   myenv\Scripts\Activate.ps1

   # On Linux/Mac
   source myenv/bin/activate
   ```

4. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

5. Download the pre-trained YOLO models or use the provided ones (`yolo11m.pt`, `yolo11n.pt`, or `best.pt`).

## Usage

### Vehicle Detection and Tracking

Run the detection on a video file:

```
python speed_plate_detector.py --source data/highway.mp4 --save
```

Parameters:

- `--source`: Path to input video file
- `--save`: Save the output video with detections
- `--show-vid`: Display the video during processing
- `--speed-limit`: Set custom speed limit (default: 50 km/h)

### Speed Estimation and Violation Detection

To detect speed violations and generate a CSV report:

```
python Detect_csv.py --source_video_path data/highway.mp4 --target_video_path output/result.mp4 --csv_output_path output/speeds.csv
```

Parameters:

- `--source_video_path`: Path to input video
- `--target_video_path`: Path for output video with visualizations
- `--csv_output_path`: Path for speed data CSV output

To view detected violations:

```
python -m http.server
```

Then navigate to `http://localhost:8000/speed_violations/` in your browser.

### License Plate Recognition

Run the license plate detection and recognition:

```
python PlateDetection/app.py --image_path violations/vehicle_123.jpg
```

### Variable Speed Limit (VSL) System

1. Open the `RLmodel_DQN.ipynb` notebook in Jupyter or VS Code:

   ```
   jupyter notebook RLmodel_DQN.ipynb
   ```

   or

   ```
   code RLmodel_DQN.ipynb
   ```

2. Training a new VSL model:

   - Run the notebook cells in order
   - Adjust hyperparameters as needed in the `DQNAgent` class
   - The training results will be saved to the `models/` directory

3. Evaluating pre-trained models:

   - Load a pre-trained model using the code in the "Loading Pre-trained Models and Evaluation" section
   - Run the evaluation cells to visualize the model's performance
   - Compare different model architectures using the visualization tools

4. Testing the VSL system on new data:
   ```python
   # Inside the notebook
   data = load_traffic_data("new_traffic_data.csv")
   timestamps, recommended_speeds, actual_speeds, rewards = evaluate_vsl_agent(data, agent)
   visualize_results(timestamps, recommended_speeds, actual_speeds, rewards, data=data)
   ```

## Project Structure

- `data/`: Contains video files for testing and results
  - Various highway videos for testing the system
  - Result videos showing detection and tracking
- `models/`: Trained models for VSL
  - `vsl_best_model.weights.h5`: Best performing model weights
  - `vsl_final_model.weights.h5`: Final model after training
- `PlateDetection/`: License plate detection components
- `speed_violations/`: Contains violation records and outputs
  - `violations_report.json`: JSON file containing all detected violations
  - `output/`: Generated videos with detections
  - `violations/`: Images of vehicles and license plates involved in violations
- `SmartCity/`: Frontend web application for the system
- `myenv/`: Python virtual environment
- `*.ipynb`: Jupyter notebooks for development and experimentation
- `*.py`: Python scripts for the various system components
- `*.csv`: Traffic data files

## Results and Validation

### Vehicle Detection Performance

The YOLOv11 model achieves high accuracy in detecting various vehicle types across different lighting and weather conditions. The system can detect and classify:

- Cars
- Trucks
- Buses
- Motorcycles
- Bicycles

### Speed Estimation Accuracy

Speed estimation was validated against ground truth data with the following results:

- Mean Absolute Error (MAE): ~3.2 km/h
- Accuracy within ±5 km/h: 91.5%
- Accuracy for violation detection (>10 km/h over limit): 95.8%

### License Plate Recognition

The license plate detection and OCR system achieves:

- Plate detection accuracy: ~89%
- Character recognition accuracy: ~87%
- Overall valid plate extraction: ~78%

Performance varies with lighting conditions, distance, and plate cleanliness.

### VSL System Evaluation

The Reinforcement Learning VSL system was evaluated using simulated traffic data and showed the following improvements compared to static speed limits:

- 15.2% reduction in travel time variability
- 8.7% increase in average traffic flow
- 12.3% reduction in stop-and-go conditions
- 21.5% reduction in simulated accidents due to speed variations

Different neural network architectures were tested, with the 4-layer ReLU model showing the best overall performance in balancing traffic flow and safety.

## Future Work

- Integration with traffic signal control systems
- Real-time implementation on actual highway segments
- Incorporation of weather data for more context-aware speed recommendations
- Mobile application for drivers to receive VSL recommendations

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv11 by Ultralytics for object detection
- DeepSort for object tracking
- TensorFlow/Keras for deep learning implementations
- OpenCV for computer vision processing
- Google Gemini API for OCR capabilities
