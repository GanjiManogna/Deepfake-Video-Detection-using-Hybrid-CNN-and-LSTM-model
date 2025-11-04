import os
import cv2
import numpy as np
from tqdm import tqdm
import mediapipe as mp
from pathlib import Path

class VideoPreprocessor:
    def __init__(self, input_dir, output_base):
        """
        Initialize the VideoPreprocessor
        
        Args:
            input_dir (str): Path to directory containing video files
            output_base (str): Base directory for all outputs
        """
        self.input_dir = input_dir
        self.output_base = output_base
        
        # Initialize MediaPipe Face Mesh with lower confidence threshold
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3  # Lowered from 0.5 to detect more faces
        )
        
        # Create output directories
        self.frames_dir = os.path.join(output_base, 'frames')
        self.faces_dir = os.path.join(output_base, 'faces')
        self.flow_dir = os.path.join(output_base, 'optical_flow')
        
        for d in [self.frames_dir, self.faces_dir, self.flow_dir]:
            os.makedirs(d, exist_ok=True)
            
        # Initialize counters for debugging
        self.faces_detected = 0
        self.faces_not_detected = 0
    
    def extract_frames(self, video_path, output_dir, frame_interval=5):
        """Extract frames from video at specified interval"""
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        saved_count = 0
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create output directory for this video
        video_name = Path(video_path).stem
        video_output_dir = os.path.join(output_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)
        
        print(f"Extracting frames from {video_name}...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                frame_path = os.path.join(video_output_dir, f"frame_{frame_count:04d}.jpg")
                cv2.imwrite(frame_path, frame)
                saved_count += 1
                
            frame_count += 1
            
            # Show progress
            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")
        
        cap.release()
        print(f"Extracted {saved_count} frames from {video_name}")
        return video_output_dir
    
    def detect_faces(self, frame):
        """Detect and align face in frame"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and get face landmarks
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            # Get face landmarks
            face_landmarks = results.multi_face_landmarks[0]
            
            # Get the face bounding box
            h, w, _ = frame.shape
            x_coords = [int(lm.x * w) for lm in face_landmarks.landmark]
            y_coords = [int(lm.y * h) for lm in face_landmarks.landmark]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # Add some padding
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)
            
            # Extract face
            face = frame[y_min:y_max, x_min:x_max]
            return face, (x_min, y_min, x_max, y_max)
        
        return None, None
    
    def calculate_optical_flow(self, prev_frame, next_frame):
        """Calculate dense optical flow between two consecutive frames"""
        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate dense optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, next_gray, 
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # Convert flow to RGB for visualization
        hsv = np.zeros_like(prev_frame)
        hsv[..., 1] = 255
        
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def is_blurry(self, frame, threshold=50.0):
        """
        Check if frame is blurry using Laplacian variance
        
        Args:
            frame: Input frame (BGR or grayscale)
            threshold: Lower values make the blur detection less sensitive
                      Default: 50.0 (lowered from 100.0 to be less aggressive)
        """
        try:
            if frame is None:
                return True
                
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
            laplacian_var = cv2.Laplacian(frame, cv2.CV_64F).var()
            return laplacian_var < threshold
            
        except Exception as e:
            print(f"\nError in blur detection: {e}")
            return False  # If there's an error, don't skip the frame

    def process_video(self, video_path, target_frames=50):
        """Process a single video file with smart frame extraction"""
        video_name = Path(video_path).stem
        frames_output = os.path.join(self.frames_dir, video_name)
        faces_output = os.path.join(self.faces_dir, video_name)
        flow_output = os.path.join(self.flow_dir, video_name)
        
        # Reset counters
        self.faces_detected = 0
        self.faces_not_detected = 0
        
        # Clear and create output directories
        for d in [frames_output, faces_output, flow_output]:
            if os.path.exists(d):
                import shutil
                shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
            
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"\nProcessing: {video_name}")
        print(f"Duration: {duration:.1f}s, Frames: {total_frames}, FPS: {fps:.1f}")
        
        # Calculate frame interval based on video duration
        if duration > 5:  # For longer videos, sample frames
            frame_interval = max(1, total_frames // target_frames)
            print(f"Sampling 1 frame every {frame_interval} frames")
        else:  # For short videos, use all frames
            frame_interval = 1
            print("Using all frames (short video)")
            
        # Adjust target frames for very short videos
        if total_frames < target_frames:
            target_frames = total_frames
            print(f"Adjusting target frames to {target_frames} (video is short)")
        
        saved_frames = 0
        prev_frame = None
        
        with tqdm(total=total_frames, desc=f"Processing {video_name}") as pbar:
            for frame_idx in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames based on interval
                if frame_idx % frame_interval != 0:
                    pbar.update(1)
                    continue
                    
                # Check if we've reached target frames
                if saved_frames >= target_frames:
                    pbar.update(total_frames - frame_idx)  # Update progress bar to 100%
                    break
                    
                # Skip blurry frames (with less aggressive threshold)
                if self.is_blurry(frame, threshold=50.0):  # Lower threshold from 100.0
                    pbar.update(1)
                    continue
                
                # Detect face
                face, bbox = self.detect_faces(frame)
                if face is None or face.size == 0:
                    self.faces_not_detected += 1
                    if self.faces_not_detected % 100 == 0:  # Log every 100th failure
                        print(f"\nWarning: No face detected in {self.faces_not_detected} frames")
                    pbar.update(1)
                    continue
                    
                self.faces_detected += 1
                
                # Save frame and face
                frame_path = os.path.join(frames_output, f"frame_{saved_frames:04d}.jpg")
                face_path = os.path.join(faces_output, f"face_{saved_frames:04d}.jpg")
                
                # Save with 90% JPEG quality
                cv2.imwrite(frame_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                cv2.imwrite(face_path, face, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                
                # Calculate and save optical flow
                if prev_frame is not None and saved_frames < target_frames:
                    try:
                        flow = self.calculate_optical_flow(prev_frame, frame)
                        if flow is not None:
                            flow_path = os.path.join(flow_output, f"flow_{saved_frames:04d}.jpg")
                            cv2.imwrite(flow_path, flow, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    except Exception as e:
                        print(f"\nError calculating optical flow: {e}")
                
                prev_frame = frame.copy()
                saved_frames += 1
                pbar.update(1)
                
                if saved_frames >= target_frames:
                    print(f"\nSuccessfully saved {saved_frames} frames from {video_name}")
                    print(f"Frames saved to: {frames_output}")
                    print(f"Faces saved to: {faces_output}")
                    print(f"Optical flow saved to: {flow_output}")
                    break
        
        cap.release()
        
        # Print summary
        print(f"\n=== Processing Summary for {video_name} ===")
        print(f"Total frames processed: {frame_idx+1}")
        print(f"Frames with faces detected: {self.faces_detected}")
        print(f"Frames without faces: {self.faces_not_detected}")
        print(f"Successfully saved {saved_frames} frames")
        print(f"Frames saved to: {frames_output}")
        print(f"Faces saved to: {faces_output}")
        print(f"Optical flow saved to: {flow_output}")
        
        # If no frames were saved, remove the empty directories
        if saved_frames == 0:
            print("No frames were saved. Removing empty output directories...")
            for d in [frames_output, faces_output, flow_output]:
                if os.path.exists(d):
                    import shutil
                    shutil.rmtree(d)
        
        return saved_frames

def find_videos(directory, extensions=('.mp4', '.avi', '.mov')):
    """Recursively find all video files in directory"""
    video_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extensions):
                video_files.append(os.path.join(root, file))
    return video_files

def main(start_from=0):
    # Get the absolute path of the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    output_base = os.path.join(project_root, "preprocessing")
    
    # Define dataset paths
    dataset_root = os.path.join(project_root, "data", "faceforensics_data")
    
    # Find all video files and sort them for consistent ordering
    print("Searching for video files...")
    video_files = sorted(find_videos(dataset_root))
    
    if not video_files:
        print("No video files found in the dataset directory.")
        return
    
    total_videos = len(video_files)
    print(f"Found {total_videos} video files in total.")
    
    # Validate start_from parameter
    if start_from < 1 or start_from > total_videos:
        print(f"Invalid start_from value: {start_from}. Using 1 instead.")
        start_from = 1
    
    print(f"\nStarting from video {start_from} of {total_videos}")
    
    # Initialize preprocessor
    preprocessor = VideoPreprocessor("", output_base)
    
    # Get the subset of videos to process
    videos_to_process = video_files[start_from-1:]  # -1 because list is 0-indexed
    remaining_count = len(videos_to_process)
    
    print(f"\nVideos to process in this run: {remaining_count} (Videos {start_from} to {total_videos})")
    
    # First pass: Check which videos are already processed
    to_process = []
    processed_in_this_range = 0
    
    print("\nChecking for already processed videos in this range...")
    for i, video_path in enumerate(videos_to_process, start_from):
        video_name = Path(video_path).stem
        output_dir = os.path.join(preprocessor.frames_dir, video_name)
        
        # Check if video is already processed
        if os.path.exists(output_dir):
            frame_count = len([f for f in os.listdir(output_dir) if f.startswith('frame_') and f.endswith('.jpg')])
            if frame_count >= 50:  # Consider it processed if it has 50+ frames
                processed_in_this_range += 1
                if processed_in_this_range % 10 == 0:
                    print(f"Found {processed_in_this_range} already processed videos in this range...")
                continue
        to_process.append(video_path)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"PROCESSING SUMMARY")
    print(f"{'='*50}")
    print(f"Total videos: {total_videos}")
    print(f"Starting from video: {start_from}")
    print(f"Videos in this range: {remaining_count}")
    print(f"Already processed in this range: {processed_in_this_range}")
    print(f"Videos to process now: {len(to_process)}")
    print(f"{'='*50}\n")
    
    if not to_process:
        print("No videos need processing in this range!")
        return
    
    # Process the videos
    for i, video_path in enumerate(to_process, 1):
        video_idx = start_from + i - 1  # Calculate the actual video number
        video_name = Path(video_path).stem
        remaining = len(to_process) - i
        
        print(f"\n{'='*50}")
        print(f"PROCESSING VIDEO {i} of {len(to_process)}")
        print(f"Video {video_idx}/{total_videos} - {video_name}")
        print(f"Remaining in this batch: {remaining}")
        print(f"{'='*50}")
        
        try:
            preprocessor.process_video(video_path)
            print(f"\n✅ SUCCESS: Processed {video_name}")
            print(f"   Progress: {i}/{len(to_process)} ({(i/len(to_process))*100:.1f}% of this batch)")
            print(f"   Overall: {video_idx}/{total_videos} ({(video_idx/total_videos)*100:.1f}% of total)")
        except Exception as e:
            print(f"\n❌ ERROR processing {video_name}: {str(e)}")
            # Save failed video name to a log file
            with open(os.path.join(output_base, 'failed_videos.txt'), 'a') as f:
                f.write(f"{video_path}\t{str(e)}\n")
            continue

if __name__ == "__main__":
    # Start processing from video 1869 (1-indexed) to process remaining 767 videos
    main(start_from=1869)
