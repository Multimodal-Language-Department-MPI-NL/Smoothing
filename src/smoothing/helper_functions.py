# this is  a file with helper functions for the course
# This file contains functions to extract keypoints from videos using MediaPipe and save them to CSV files.
# It also includes a function to overlay the keypoints on the original video and save it as a new video file.
# The functions are designed to work with body, hand, and face landmarks.
# the inport statement below is used to import the necessary libraries
# for the script to work properly
from glob import glob
import os
import pandas as pd
import numpy as np
from IPython.display import Video
from scipy.signal import savgol_filter
import mediapipe as mp
import cv2
import csv
import tempfile
import matplotlib.pyplot as plt

def extract_mediapipe_keypoints_to_csv(video_path, output_dir=None, static_mode=True):
    """
    Extract body, hand, and face landmarks from a video using MediaPipe holistic
    and save directly to CSV files.
    
    Args:
        video_path (str): Path to the input video file
        output_dir (str): Directory to save output files (if None, uses temp files)
        static_mode (bool): Whether to use static image mode (more accurate but jittery)
        
    Returns:
        tuple: (body_csv_path, hands_csv_path, face_csv_path) - Paths to the CSV files
    """



    
    # Setup output directories
    if output_dir is None:
        output_dir = tempfile.mkdtemp()
        output_ts_path = output_dir
    else:
        os.makedirs(os.path.join(output_dir, "Output_TimeSeries"), exist_ok=True)
        output_ts_path = os.path.join(output_dir, "Output_TimeSeries")
    
    # Generate output file paths
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    body_csv_path = os.path.join(output_ts_path, f'{video_basename}_body.csv')
    hands_csv_path = os.path.join(output_ts_path, f'{video_basename}_hands.csv')
    face_csv_path = os.path.join(output_ts_path, f'{video_basename}_face.csv')
    
    # Initialize MediaPipe modules
    global mp_holistic
    mp_holistic = mp.solutions.holistic
    
    # Define landmarks
    global markersbody
    markersbody = ['NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_OUTER', 'RIGHT_EYE', 'RIGHT_EYE_INNER',
                'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 
                'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX',
                'LEFT_THUMB', 'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE',
                'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']

    global markershands
    markershands  = ['LEFT_WRIST_H', 'LEFT_THUMB_CMC', 'LEFT_THUMB_MCP', 'LEFT_THUMB_IP', 'LEFT_THUMB_TIP', 'LEFT_INDEX_FINGER_MCP',
                  'LEFT_INDEX_FINGER_PIP', 'LEFT_INDEX_FINGER_DIP', 'LEFT_INDEX_FINGER_TIP', 'LEFT_MIDDLE_FINGER_MCP', 
                   'LEFT_MIDDLE_FINGER_PIP', 'LEFT_MIDDLE_FINGER_DIP', 'LEFT_MIDDLE_FINGER_TIP', 'LEFT_RING_FINGER_MCP', 
                   'LEFT_RING_FINGER_PIP', 'LEFT_RING_FINGER_DIP', 'LEFT_RING_FINGER_TIP', 'LEFT_PINKY_FINGER_MCP', 
                   'LEFT_PINKY_FINGER_PIP', 'LEFT_PINKY_FINGER_DIP', 'LEFT_PINKY_FINGER_TIP',
                  'RIGHT_WRIST_H', 'RIGHT_THUMB_CMC', 'RIGHT_THUMB_MCP', 'RIGHT_THUMB_IP', 'RIGHT_THUMB_TIP', 'RIGHT_INDEX_FINGER_MCP',
                  'RIGHT_INDEX_FINGER_PIP', 'RIGHT_INDEX_FINGER_DIP', 'RIGHT_INDEX_FINGER_TIP', 'RIGHT_MIDDLE_FINGER_MCP', 
                   'RIGHT_MIDDLE_FINGER_PIP', 'RIGHT_MIDDLE_FINGER_DIP', 'RIGHT_MIDDLE_FINGER_TIP', 'RIGHT_RING_FINGER_MCP', 
                   'RIGHT_RING_FINGER_PIP', 'RIGHT_RING_FINGER_DIP', 'RIGHT_RING_FINGER_TIP', 'RIGHT_PINKY_FINGER_MCP', 
                   'RIGHT_PINKY_FINGER_PIP', 'RIGHT_PINKY_FINGER_DIP', 'RIGHT_PINKY_FINGER_TIP']
    
    # only keep key face landmarks with human-readable names
    facemarks = [
        'NOSE_TIP',       # index 1
        'LEFT_EYE_OUTER', # index 33
        'MOUTH_LEFT',     # index 61
        'CHIN',           # index 199
        'RIGHT_EYE_OUTER',# index 263
        'MOUTH_RIGHT'     # index 291
    ]
    

    

    # Set up column names for time series
    markerxyzbody = ['time']
    markerxyzhands = ['time']
    markerxyzface = ['time']

    # Add coordinate columns to the marker lists
    for mark in markersbody:
        for pos in ['X', 'Y', 'Z', 'visibility']:
            nm = pos + "_" + mark
            markerxyzbody.append(nm)
    for mark in markershands:
        for pos in ['X', 'Y', 'Z']: 
            nm = pos + "_" + mark
            markerxyzhands.append(nm)
    for mark in facemarks:
        for pos in ['X', 'Y', 'Z']:
            nm = pos + "_" + mark
            markerxyzface.append(nm)
    
    # Helper functions for processing landmarks
    def num_there(s):
        return any(i.isdigit() for i in s)

    def makegoginto_str(gogobj):
        if gogobj is None:
            return []
        gogobj = str(gogobj).strip("[]")
        gogobj = gogobj.split("\n")
        return(gogobj[:-1])

    def listpostions(newsamplemarks):
        newsamplemarks = makegoginto_str(newsamplemarks)
        tracking_p = []
        for value in newsamplemarks:
            if num_there(value):
                stripped = value.split(':', 1)[1]
                stripped = stripped.strip()
                tracking_p.append(stripped)
        return(tracking_p)
    
    # Open CSV files for writing
    body_file = open(body_csv_path, 'w+', newline='')
    hands_file = open(hands_csv_path, 'w+', newline='')
    face_file = open(face_csv_path, 'w+', newline='')
    
    body_writer = csv.writer(body_file)
    hands_writer = csv.writer(hands_file)
    face_writer = csv.writer(face_file)
    
    # Write headers to CSV files
    body_writer.writerow(markerxyzbody)
    hands_writer.writerow(markerxyzhands)
    face_writer.writerow(markerxyzface)
    
    # Process the video
    capture = cv2.VideoCapture(video_path)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    samplerate = capture.get(cv2.CAP_PROP_FPS)
    
    # Initialize time counter
    time = 0
    frames_processed = 0
    
    with mp_holistic.Holistic(
            static_image_mode=static_mode,
            enable_segmentation=True,
            refine_face_landmarks=True) as holistic:
        
        while True:
            ret, frame_bgr = capture.read()
            if not ret:
                break
            
            frames_processed += 1
            if frames_processed % 30 == 0:  # Print progress every 30 frames
                print(f"Processing frame {frames_processed}/{total_frames} ({100*frames_processed/total_frames:.1f}%)")
            
            image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)
            
            if results.segmentation_mask is not None:  # check if there is a pose found
                # Extract landmarks for timeseries
                samplebody = listpostions(results.pose_landmarks)
                sampleface = listpostions(results.face_landmarks)
                sampleLH = listpostions(results.left_hand_landmarks)
                sampleRH = listpostions(results.right_hand_landmarks)
                
                # Handle missing hand landmarks for consistent timeseries length
                num_hand_coords = (len(markerxyzhands) - 1) // 2 
                if not results.left_hand_landmarks: sampleLH = [""] * num_hand_coords
                if not results.right_hand_landmarks: sampleRH = [""] * num_hand_coords
                
                # Combine and add time index
                samplehands = sampleLH + sampleRH
                samplebody.insert(0, time)
                samplehands.insert(0, time)
                sampleface.insert(0, time)
                
                # Write directly to CSV files
                body_writer.writerow(samplebody)
                hands_writer.writerow(samplehands)
                face_writer.writerow(sampleface)
            else:
                # If no pose detected, add empty data with timestamp
                empty_body = [""] * (len(markerxyzbody) - 1)
                empty_hands = [""] * (len(markerxyzhands) - 1)
                empty_face = [""] * (len(markerxyzface) - 1)
                
                empty_body.insert(0, time)
                empty_hands.insert(0, time)
                empty_face.insert(0, time)
                
                # Write empty rows to CSV files
                body_writer.writerow(empty_body)
                hands_writer.writerow(empty_hands)
                face_writer.writerow(empty_face)
            
            time = time + (1000/samplerate)
    
    # Close CSV files
    body_file.close()
    hands_file.close()
    face_file.close()
    
    # Clean up resources
    capture.release()
    
    print(f"Keypoint extraction complete. {frames_processed} frames processed.")
    print(f"CSV files saved to:")
    print(f"  - Body: {body_csv_path}")
    print(f"  - Hands: {hands_csv_path}")
    print(f"  - Face: {face_csv_path}")
    
    # Return the CSV file paths
    return body_csv_path, hands_csv_path, face_csv_path



def overlay_keypoints_from_csv(video_path, df_body, df_hands, df_face, output_video_path):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('Error opening video file:', video_path)
        return

    # Get frame dimensions and FPS.
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frameWidth, frameHeight))
    if not out.isOpened():
        print('Error opening video writer:', output_video_path)
        return

    # Read CSV files.
    
    frame_count = min(len(df_body), len(df_hands), len(df_face))

    # Simple helper: convert normalized coordinate [0, 1] to pixel coordinate.
    def norm_to_pixel(x_norm, y_norm):
        if pd.isna(x_norm) or pd.isna(y_norm):
            return None
        x = int(x_norm * frameWidth)
        y = int(y_norm * frameHeight)
        return (x, y)

    for frame_idx in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        # --- Body landmarks (CSV: time, then for each marker [X, Y, Z, visibility]) ---
        body_points = {}
        for i, marker in enumerate(markersbody):
            col_index = 1 + i * 4  # Skip the time column.
            try:
                x_norm = float(df_body.iloc[frame_idx, col_index])
                y_norm = float(df_body.iloc[frame_idx, col_index + 1])
            except Exception:
                continue
            pt = norm_to_pixel(x_norm, y_norm)
            if pt is not None:
                body_points[i] = pt
                cv2.circle(frame, pt, 3, (0, 255, 0), -1)
        for connection in mp_holistic.POSE_CONNECTIONS:
            if connection[0] in body_points and connection[1] in body_points:
                cv2.line(frame, body_points[connection[0]], body_points[connection[1]], (0, 255, 0), 2)

        # --- Left hand landmarks (CSV: time, then 21 markers [X, Y, Z]) ---
        left_points = {}
        for i in range(21):
            col_index = 1 + i * 3  # Skip the time column.
            try:
                x_norm = float(df_hands.iloc[frame_idx, col_index])
                y_norm = float(df_hands.iloc[frame_idx, col_index + 1])
            except Exception:
                continue
            pt = norm_to_pixel(x_norm, y_norm)
            if pt is not None:
                left_points[i] = pt
                cv2.circle(frame, pt, 3, (255, 0, 0), -1)
        for connection in mp_holistic.HAND_CONNECTIONS:
            if connection[0] in left_points and connection[1] in left_points:
                cv2.line(frame, left_points[connection[0]], left_points[connection[1]], (255, 0, 0), 2)

        # --- Right hand landmarks (CSV: time then next 21 markers [X, Y, Z]) ---
        right_points = {}
        for i in range(21, 42):
            col_index = 1 + 21 * 3 + (i - 21) * 3
            try:
                x_norm = float(df_hands.iloc[frame_idx, col_index])
                y_norm = float(df_hands.iloc[frame_idx, col_index + 1])
            except Exception:
                continue
            pt = norm_to_pixel(x_norm, y_norm)
            if pt is not None:
                right_points[i - 21] = pt
                cv2.circle(frame, pt, 3, (0, 255, 255), -1)
        for connection in mp_holistic.HAND_CONNECTIONS:
            if connection[0] in right_points and connection[1] in right_points:
                cv2.line(frame, right_points[connection[0]], right_points[connection[1]], (0, 255, 255), 2)

        # --- Face landmarks (CSV: time then 478 markers [X, Y, Z]) ---
        face_points = {}
        for i in range(478):
            col_index = 1 + i * 3  # Skip the time column.
            try:
                x_norm = float(df_face.iloc[frame_idx, col_index])
                y_norm = float(df_face.iloc[frame_idx, col_index + 1])
            except Exception:
                continue
            pt = norm_to_pixel(x_norm, y_norm)
            if pt is not None:
                face_points[i] = pt
                cv2.circle(frame, pt, 1, (0, 0, 255), -1)
        

        out.write(frame)

    cap.release()
    out.release()
    print("Overlay video saved to", output_video_path) 
    return output_video_path

                    