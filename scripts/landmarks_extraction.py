import cv2
import mediapipe as mp
import numpy as np
import os
from pathlib import Path
import json
from tqdm import tqdm
import time

class PSLLandmarkExtractor:
    def __init__(self, base_path="psl_dataset"):
        self.base_path = Path(base_path)
        self.raw_videos_path = self.base_path / "raw_videos"
        self.processed_landmarks_path = self.base_path / "processed_landmarks"
        
        # Initialize MediaPipe
        self.mp_holistic = mp.solutions.holistic
        
        # Create processed directories if they don't exist
        self.setup_output_directories()
        
    def setup_output_directories(self):
        sentence_folders = [
            "hello", 
            "please", 
            "thanks",
            "nice_to_meet_you",
    ]
    
        for folder in sentence_folders:
            output_dir = self.processed_landmarks_path / folder
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f" Created/Verified directory: {output_dir}")
    
    def extract_landmarks_from_frame(self, results):
        """Extract landmarks from MediaPipe results for a single frame"""
        landmarks = []
        
        # Left hand landmarks (21 points × 3 coordinates = 63 values)
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 63)  # Fill with zeros if no hand detected
        
        # Right hand landmarks (21 points × 3 coordinates = 63 values)
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 63)
        
        # Pose landmarks (33 points × 3 coordinates = 99 values)
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 99)
        
        return landmarks  # Total: 225 features per frame (without face)
    
    def extract_landmarks_from_video(self, video_path):
        """Extract landmarks from a single video file"""
        print(f" Processing: {video_path.name}")
        
        landmarks_sequence = []
        frame_count = 0
        
        with self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=False
        ) as holistic:
            
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                print(f" Error: Could not open video {video_path}")
                return None
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            print(f" Video Info: {total_frames} frames, {fps:.2f} FPS")
            
            # Process each frame
            with tqdm(total=total_frames, desc=f"Processing {video_path.name}", unit="frames") as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    
                    # Convert BGR to RGB (MediaPipe requirement)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Process frame
                    results = holistic.process(rgb_frame)
                    
                    # Extract landmarks
                    frame_landmarks = self.extract_landmarks_from_frame(results)
                    landmarks_sequence.append(frame_landmarks)
                    
                    pbar.update(1)
            
            cap.release()
            
        print(f"✅ Extracted {frame_count} frames")
        return np.array(landmarks_sequence)
    
    def process_all_videos(self, skip_existing=True):
    
        print(" Starting landmark extraction for all videos...")
        print(f" Input directory: {self.raw_videos_path}")
        print(f" Output directory: {self.processed_landmarks_path}")
        print(f"  Skip existing: {skip_existing}")
        print("-" * 60)
        
        total_videos_found = 0
        total_videos_processed = 0
        successful_extractions = 0
        skipped_videos = 0
        failed_extractions = []
        
        # Process each sentence category
        for sentence_folder in self.raw_videos_path.iterdir():
            if sentence_folder.is_dir():
                print(f"\n Processing category: {sentence_folder.name}")
                print("=" * 50)
                
                # Get all video files in this category
                video_files = list(sentence_folder.glob("*.mp4")) + list(sentence_folder.glob("*.avi"))
                total_videos_found += len(video_files)
                print(f"📹 Found {len(video_files)} video files")
                
                if not video_files:
                    print(f"No video files found in {sentence_folder}")
                    continue
                
                # Process each video
                for video_file in video_files:
                    try:
                        # Check if already processed
                        output_filename = f"{video_file.stem}.npy"
                        output_path = self.processed_landmarks_path / sentence_folder.name / output_filename
                        
                        if skip_existing and output_path.exists():
                            print(f" Skipping (already processed): {video_file.name}")
                            skipped_videos += 1
                            continue
                        
                        total_videos_processed += 1
                        start_time = time.time()
                        
                        # Extract landmarks
                        landmarks = self.extract_landmarks_from_video(video_file)
                        
                        if landmarks is not None:
                            # Save landmarks
                            np.save(output_path, landmarks)
                            
                            processing_time = time.time() - start_time
                            file_size = output_path.stat().st_size / 1024  # KB
                            
                            print(f" Saved: {output_path}")
                            print(f"  Processing time: {processing_time:.2f}s")
                            print(f" Landmarks shape: {landmarks.shape}")
                            print(f" File size: {file_size:.1f} KB")
                            print("-" * 40)
                            
                            successful_extractions += 1
                        else:
                            failed_extractions.append(str(video_file))
                            print(f" Failed to process: {video_file}")
                    
                    except Exception as e:
                        failed_extractions.append(str(video_file))
                        print(f" Error processing {video_file}: {str(e)}")
        
        # Summary report
        print("\n" + "="*60)
        print(" PROCESSING SUMMARY")
        print("="*60)
        print(f" Total videos found: {total_videos_found}")
        print(f" Skipped (already processed): {skipped_videos}")
        print(f" Videos processed: {total_videos_processed}")
        print(f" Successfully extracted: {successful_extractions}/{total_videos_processed} videos")
        print(f" Failed extractions: {len(failed_extractions)}")
        
        if failed_extractions:
            print("\nFailed files:")
            for failed_file in failed_extractions:
                print(f"  - {failed_file}")
        
        print(f"\n All landmarks saved to: {self.processed_landmarks_path}")
        
        return successful_extractions, failed_extractions
    
def create_labels_file(base_path="psl_dataset"):
    """Create the sentence labels JSON file"""
    labels_dir = Path(base_path) / "labels"
    labels_dir.mkdir(exist_ok=True)
    
    labels = {
        "hello": {
            "urdu_text": "سلام",
            "english": "Hello",
            "class_id": 0
        },
        "please": {
            "urdu_text": "براہ مہربانی",
            "english": "Please",
            "class_id": 1
        },
        "thanks": {
            "urdu_text": "شکریہ",
            "english": "Thanks",
            "class_id": 2
        },
        "nice_to_meet_you": {
            "urdu_text": "آپ سے مل کر خوشی ہوئی",
            "english": "Nice to meet you",
            "class_id": 3
        },
       
    }
    
    labels_file = labels_dir / "sentence_labels.json"
    with open(labels_file, 'w', encoding='utf-8') as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    
    print(f" Created labels file: {labels_file}")

if __name__ == "__main__":
    print(" PSL Landmark Extraction Tool")
    print("="*50)
    
    # Create labels file
    create_labels_file()
    
    # Initialize extractor
    extractor = PSLLandmarkExtractor()
    
    successful, failed = extractor.process_all_videos(skip_existing=True)
    
    print(f"\n Extraction completed!")
    print(f" {successful} videos processed successfully")
    
    if failed:
        print(f"{len(failed)} videos failed")
    else:
        print(" All videos processed without errors!")