# MediaPipe Keypoints Smoothing

A comprehensive pipeline for cleaning and smoothing video-based motion-tracking data, specifically MediaPipe keypoints, for robust gesture analysis and multimodal interaction research.

## 🔬 Research Context

This project is part of a larger research framework for analyzing multimodal communication, particularly:
- **Gesture segmentation** and classification
- **Kinematic feature extraction** for movement analysis  
- **Multimodal similarity analysis** combining speech and gesture features
- **Temporal alignment** of different modalities in conversation

## 🎯 What This Project Does

This project addresses common challenges in motion capture data analysis:

1. **Data Extraction & Cleanup**: Load raw MediaPipe CSV files (body, hands, face), remove NaN values, and inspect temporal jitter
2. **Smoothing Techniques**: Apply and compare multiple filtering methods to reduce noise while preserving meaningful motion
3. **Evaluation**: Visually and quantitatively assess smoothing effectiveness by overlaying filtered signals on original data

## 📊 Smoothing Process

The notebook follows a systematic 5-step approach to smoothing:

1. **Begin with Cleaned Data**: Start with preprocessed data free of missing values and outliers
2. **Inspect the Data**: Visualize time series plots and video overlays to identify noise patterns
3. **Select Smoothing Technique**: Choose appropriate filtering method based on data characteristics
4. **Evaluate Results**: Compare smoothed data against original using visual and quantitative metrics
5. **Save Processed Data**: Export cleaned and smoothed data for further analysis

## 🔧 Smoothing Techniques

The project implements and compares several filtering methods:

- **Zero-Phase Low-Pass Butterworth Filter**: Forward-backward application (`filtfilt`) for effective noise reduction without phase distortion
- **Savitzky-Golay Filter**: Polynomial smoothing over a moving window to preserve higher-order moments
- **Gaussian Neighbor-Averaging**: Weighted moving average using a Gaussian kernel
- **MediaPipe Built-in Smoothing**: Exploration of MediaPipe's native smoothing when `static_image_mode=False`
- **Custom Smoothing**: Flexible custom functions for specific research needs

## 📁 Project Structure

```
Smoothing/
├── README.md
├── .gitignore
├── .gitattributes
├── environment.yml
├── helper_functions.py              # Core MediaPipe extraction functions
├── notebooks/
│   └── Smoothing.ipynb
├── src/
│   ├── 5step.png                   # Process diagram
│   └── Input_Videos/               # Sample input videos
├── data/
│   └── interim/                    # Processed CSV files
├── results/
│   ├── Output_TimeSeries/          # Extracted keypoints (CSV)
│   └── Output_Video_overlay/       # Annotated videos
└── scripts/
    └── smooth.py                   # Command-line utilities
```

## 🚀 Quick Start

### Prerequisites

```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate smoothing
```

### Run the Interactive Notebook

```bash
jupyter lab
# Open notebooks/Smoothing.ipynb
```

### Basic Usage

1. **Place your video** in `src/Input_Videos/` or update the path in the notebook
2. **Run the extraction** to generate CSV files with MediaPipe keypoints
3. **Apply smoothing** using various filtering techniques
4. **Compare results** by viewing overlay videos and analyzing data quality

## 🔧 Key Functions

### `extract_mediapipe_keypoints_to_csv()`
Extracts body, hand, and face landmarks from video using MediaPipe Holistic and saves directly to CSV files.

**Parameters:**
- `video_path`: Path to input video file
- `output_dir`: Directory to save output files
- `static_mode`: Whether to use static image mode (more accurate but jittery)

**Returns:**
- `body_csv_path`: Path to body landmarks CSV
- `hands_csv_path`: Path to hand landmarks CSV  
- `face_csv_path`: Path to face landmarks CSV

### `overlay_keypoints_from_csv()`
Creates annotated videos with keypoints overlaid on the original video frames.

**Parameters:**
- `video_path`: Original video file
- `df_body`, `df_hands`, `df_face`: DataFrames with landmark data
- `output_video_path`: Path for the annotated video

## 📈 Data Format

### Input
- **Video files**: MP4 format with human subjects performing gestures
- **MediaPipe output**: 33 body landmarks + 42 hand landmarks + 6 key face landmarks

### Output
- **CSV files**: Time series data with columns for each landmark (X, Y, Z, visibility)
- **Annotated videos**: MP4 files with keypoints overlaid on original frames
- **Smoothed data**: Filtered time series for further analysis

### Data Structure
```python
# Body landmarks: 33 points × 4 coordinates (X, Y, Z, visibility)
# Hand landmarks: 42 points × 3 coordinates (X, Y, Z) - 21 per hand
# Face landmarks: 6 key points × 3 coordinates (X, Y, Z)
```

## 🎛️ Configuration

### Smoothing Parameters

- **Butterworth Filter**: `sampling_rate`, `order`, `lowpass_cutoff`
- **Savitzky-Golay**: `window_length`, `polyorder`
- **Gaussian**: `sigma` (kernel width)
- **MediaPipe**: `static_image_mode`, `smooth_landmarks`

### Quality Control

- **Jitter Analysis**: Frame-to-frame variance measurement
- **Visual Inspection**: Overlay comparison between original and smoothed data
- **Quantitative Metrics**: Signal-to-noise ratio, smoothness indices

## 📚 Research Applications

This smoothing pipeline is essential for:

- **Gesture Recognition**: Clean data improves classification accuracy
- **Motion Analysis**: Smoothed trajectories enable better kinematic analysis
- **Multimodal Alignment**: Consistent data quality across different modalities
- **Comparative Studies**: Standardized smoothing enables cross-participant analysis

## 🔗 Related Projects

- **MediaPipe Keypoints Extraction**: [../MediaPipe_keypoints_extraction/](../MediaPipe_keypoints_extraction/)
- **Gesture Kinematic Analysis**: [../Speed_Acceleration_Jerk/](../Speed_Acceleration_Jerk/)
- **Gesture Segmentation**: [../Submovements_Holds/](../Submovements_Holds/)

## 📖 References

- **MediaPipe Documentation**: [https://mediapipe.dev/](https://mediapipe.dev/)
- **Signal Processing**: Challis, J. H. (2021). *Experimental Methods in Biomechanics*
- **Gesture Analysis**: McNeill, D. (1992). *Hand and Mind: What Gestures Reveal About Thought*

## 🤝 Contributing

This project is part of the MPI Multimodal Interaction Research framework. For questions or contributions, please refer to the main project documentation.

## 📄 License

This project is part of the MPI research framework. Please refer to the main project license for usage terms.