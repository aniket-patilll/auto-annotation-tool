import os
import cv2
from pathlib import Path
from tqdm import tqdm

def extract_frames(video_path: Path, output_dir: Path, target_fps: float = None):
    """
    Extract frames from a video file and save them as images.
    
    Args:
        video_path (Path): Path to the input video file.
        output_dir (Path): Directory where extracted frames will be saved.
        target_fps (float, optional): Number of frames to extract per second of video. 
                                      If None, extracts all frames.
    """
    # Create the output subfolder for this specific video
    video_stem = video_path.stem
    video_output_dir = output_dir / video_stem
    video_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path.name}")
        return
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Fallback if FPS detection fails
    if not video_fps or video_fps <= 0:
        video_fps = 30.0
        
    # Determine the step/interval for frame extraction
    if target_fps is not None and target_fps > 0:
        # e.g., if video is 30 FPS and target is 2 FPS, we take every 15th frame
        frame_interval = max(1, int(video_fps / target_fps))
    else:
        frame_interval = 1
        
    print(f"\nProcessing: {video_path.name}")
    print(f"Original Video: {total_frames} frames @ {video_fps:.2f} FPS")
    if target_fps:
        print(f"Targeting ~{target_fps} FPS (Extracting 1 frame every {frame_interval} frames)")
    else:
        print("Extracting ALL frames")

    frame_idx = 0
    extracted_count = 0
    
    # Use tqdm progress bar for sequential frame reading
    with tqdm(total=total_frames, desc="Extracting", unit="frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % frame_interval == 0:
                frame_name = video_output_dir / f"{video_stem}_f{frame_idx:06d}.jpg"
                # Save with high quality JPEG (95)
                cv2.imwrite(str(frame_name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                extracted_count += 1
                
            frame_idx += 1
            pbar.update(1)
            
    cap.release()
    print(f"Successfully extracted {extracted_count} frames to: {video_output_dir.relative_to(output_dir.parent) if output_dir.parent else video_output_dir}")

def process_directory(input_dir: str, output_dir: str, target_fps: float = None):
    """
    Finds all video files in input_dir and extracts frames for each.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Supported video formats
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}
    
    # Find all videos (case-insensitive check)
    video_files = [
        p for p in input_path.iterdir()
        if p.is_file() and p.suffix.lower() in video_extensions
    ]
    
    if not video_files:
        print(f"No matching video files found in: {input_dir}")
        print(f"Supported extensions: {', '.join(video_extensions)}")
        return
        
    print(f"Found {len(video_files)} video(s) to process.")
    for video_file in video_files:
        extract_frames(video_file, output_path, target_fps)
        
    print("\nAll videos processed successfully!")

if __name__ == "__main__":
    # --- GOOGLE COLAB / LOCAL CONFIGURATION ---
    # Edit these paths as needed. If using Google Drive in Colab, you can mount it first.
    
    # Example for Colab Google Drive mount:
    # from google.colab import drive
    # drive.mount('/content/drive')
    
    INPUT_FOLDER = "./videos"      # Path to folder containing video files
    OUTPUT_FOLDER = "./frames"     # Path where frame folders will be saved
    TARGET_FPS = 2.0               # Set to None to extract all frames, or e.g., 2.0 to extract 2 frames/sec
    
    # Let's create dummy input folder if running locally for the first time
    if not os.path.exists(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER, exist_ok=True)
        print(f"Created '{INPUT_FOLDER}' folder. Place your video files there and run the script.")
    else:
        process_directory(INPUT_FOLDER, OUTPUT_FOLDER, TARGET_FPS)
