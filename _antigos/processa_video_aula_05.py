import os
import re
import shutil
import subprocess
import tempfile
import whisper
import argparse
import glob
import logging
from tqdm import tqdm
from time import time
from PIL import Image, ImageFilter
import pytesseract
from multiprocessing import Process, Queue, Pool
from spellchecker import SpellChecker
from langdetect import detect, DetectorFactory

# Fix seed for langdetect consistency
DetectorFactory.seed = 0

# Initialize spell checker for Portuguese
spell = SpellChecker(language="pt")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def extract_frames(video_path, temp_dir, fps=4, resolution="1280:720"):
    """Extract frames from video using ffmpeg and collect timestamps."""
    try:
        frame_pattern = os.path.join(temp_dir, "frame_%06d.png")
        command = [
            "ffmpeg", "-i", video_path, "-vf", f"fps={fps},scale={resolution},showinfo", frame_pattern
        ]

        process = subprocess.Popen(command, stderr=subprocess.PIPE, text=True)
        log_data = []

        for line in tqdm(process.stderr, desc="Extracting frames", unit="line"):
            if "pts_time" in line:
                log_data.append(line)

        process.wait()

        if process.returncode != 0:
            raise RuntimeError("Error processing video with ffmpeg.")

        return log_data
    except Exception as e:
        logging.error(f"Error extracting frames: {e}")
        raise

def parse_log_data(log_data):
    """Parse ffmpeg logs to get timestamps for each frame."""
    frame_times = []
    try:
        for line in log_data:
            match = re.search(r'pts_time:([0-9.]+)', line)
            if match:
                timestamp = float(match.group(1))
                minutes = int(timestamp // 60)
                seconds = int(timestamp % 60)
                milliseconds = int((timestamp - int(timestamp)) * 1000)
                frame_times.append((minutes, seconds, milliseconds))
    except Exception as e:
        logging.error(f"Error parsing log data: {e}")
        raise

    return frame_times

def rename_frames(frame_times, temp_dir):
    """Rename frames based on extracted timestamps."""
    try:
        for i, (minutes, seconds, milliseconds) in enumerate(tqdm(frame_times, desc="Renaming frames", unit="frame")):
            original_name = os.path.join(temp_dir, f"frame_{i+1:06d}.png")
            new_name = os.path.join(temp_dir, f"frame_{minutes:02d}-{seconds:02d}-{milliseconds:03d}.png")

            if os.path.exists(original_name):
                os.rename(original_name, new_name)
            else:
                logging.warning(f"File {original_name} not found. Skipping...")
    except Exception as e:
        logging.error(f"Error renaming frames: {e}")
        raise

def preprocess_image(image):
    """Preprocess the image to improve OCR accuracy."""
    try:
        # Convert to grayscale
        image = image.convert("L")

        # Apply thresholding
        image = image.point(lambda x: 0 if x < 140 else 255, '1')

        # Apply median filter to reduce noise
        image = image.filter(ImageFilter.MedianFilter())

        # Resize image to double the size
        image = image.resize((image.width * 2, image.height * 2), Image.ANTIALIAS)

        return image
    except Exception as e:
        logging.error(f"Error preprocessing image: {e}")
        raise

def is_human_readable_text(text):
    """Check if the text is legible and in Portuguese."""
    try:
        # Remove non-letter characters
        cleaned_text = re.sub(r'[^A-Za-zÀ-ÖØ-öø-ÿ\s]', '', text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        if not cleaned_text:
            return False

        # Detect language
        language = detect(cleaned_text)
        if language != 'pt':
            return False

        # Split into words
        words = cleaned_text.split()

        # Words with 4 or more letters
        long_words = [word for word in words if len(word) >= 4]

        if len(long_words) > 0:
            return True

        return False
    except Exception as e:
        logging.error(f"Error in text validation: {e}")
        return False

def process_frame_with_ocr(args):
    """Process a single frame with OCR and save the text."""
    frame_path, output_dir = args
    try:
        image = Image.open(frame_path)

        # Preprocess the image
        image = preprocess_image(image)

        # Configure Tesseract for better accuracy
        custom_config = r'--oem 1 --psm 6 -l por'

        # Perform OCR
        text = pytesseract.image_to_string(image, config=custom_config)

        if text.strip() and is_human_readable_text(text):
            # Copy the frame to the output directory
            output_frame_path = os.path.join(output_dir, os.path.basename(frame_path))
            shutil.copy(frame_path, output_frame_path)

            # Save the extracted text
            text_output_path = os.path.join(output_dir, os.path.basename(frame_path).replace(".png", ".txt"))
            with open(text_output_path, "w", encoding="utf-8") as text_file:
                text_file.write(text.strip())
    except Exception as e:
        logging.error(f"Error processing frame {frame_path}: {e}")

def detect_text_in_frames(temp_dir, output_dir, batch_size=5):
    """Scan frames and identify those containing text."""
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        frame_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith(".png")]

        args_list = [(frame, output_dir) for frame in frame_files]

        with Pool(processes=batch_size) as pool:
            list(tqdm(pool.imap_unordered(process_frame_with_ocr, args_list),
                      total=len(frame_files), desc="Processing frames", unit="frame"))
    except Exception as e:
        logging.error(f"Error detecting text in frames: {e}")
        raise

def process_frames(video_path, temp_dir, progress_queue, batch_size):
    """Process frames: extraction, renaming, and OCR."""
    try:
        log_data = extract_frames(video_path, temp_dir)
        frame_times = parse_log_data(log_data)
        rename_frames(frame_times, temp_dir)

        # Output directory for frames with text
        video_dir = os.path.dirname(video_path)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(video_dir, f"frames_{video_name}")

        detect_text_in_frames(temp_dir, output_dir, batch_size=batch_size)
        progress_queue.put("Frame processing completed!")
    except Exception as e:
        logging.error(f"Error processing frames for {video_path}: {e}")
        progress_queue.put(f"Error processing frames: {e}")
    finally:
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir)
            logging.info(f"Temporary directory {temp_dir} removed.")
        except Exception as e:
            logging.warning(f"Could not remove temporary directory {temp_dir}: {e}")

def transcribe_audio(video_path, model_name="medium", language="Portuguese"):
    """Transcribe audio from video using Whisper."""
    try:
        whisper_model = whisper.load_model(model_name)
        result = whisper_model.transcribe(video_path, language=language, task="transcribe")

        srt_path = video_path.replace(".mp4", ".srt")
        cronometed_speech_path = video_path.replace(".mp4", "-Fala.Cronometrada.txt")

        with open(srt_path, "w", encoding="utf-8") as srt_file:
            for segment in result['segments']:
                start_time = segment['start']
                end_time = segment['end']
                text = segment['text'].strip()

                srt_file.write(f"{segment['id']}\n")
                srt_file.write(f"{format_timestamp(start_time)} --> {format_timestamp(end_time)}\n")
                srt_file.write(f"{text}\n\n")

        with open(cronometed_speech_path, "w", encoding="utf-8") as txt_file:
            for segment in result['segments']:
                start_time = format_timestamp(segment['start'])
                text = segment['text'].strip()
                txt_file.write(f"{start_time}: {text}\n")

        logging.info(f"Transcription files generated: {srt_path}, {cronometed_speech_path}")
    except Exception as e:
        logging.error(f"Error transcribing audio for {video_path}: {e}")
        raise

def format_timestamp(seconds):
    """Format seconds into SRT timestamp format."""
    try:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
    except Exception as e:
        logging.error(f"Error formatting timestamp: {e}")
        return "00:00:00,000"

def process_transcription(video_path, model_name, progress_queue):
    """Process audio transcription."""
    try:
        transcribe_audio(video_path, model_name=model_name)
        progress_queue.put("Audio transcription completed!")
    except Exception as e:
        logging.error(f"Error in transcription process: {e}")
        progress_queue.put(f"Error in transcription: {e}")

def find_files_with_mask(mask, recursive):
    """Find files based on the provided mask."""
    try:
        if recursive:
            return glob.glob(mask, recursive=True)
        else:
            return glob.glob(mask)
    except Exception as e:
        logging.error(f"Error finding files with mask {mask}: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="Process videos to extract frames and generate subtitles with Whisper.")
    parser.add_argument("file_mask", type=str, help="File mask for processing (e.g., *.mp4).")
    parser.add_argument("--model", type=str, default="medium", help="Whisper model to use (default: medium).")
    parser.add_argument("--recursive", action="store_true", help="Recursively search subdirectories.")
    parser.add_argument("--batches", type=int, default=5, help="Number of parallel batches (default: 5).")
    parser.add_argument("--temp_dir", type=str, help="Temporary directory to store frames.")
    args = parser.parse_args()

    file_mask = args.file_mask
    model_name = args.model
    recursive = args.recursive
    batch_size = args.batches

    # Use system temp directory if temp_dir not provided
    if args.temp_dir:
        temp_dir_base = args.temp_dir
    else:
        temp_dir_base = tempfile.gettempdir()

    # Find video files to process
    video_files = find_files_with_mask(file_mask, recursive)

    if not video_files:
        logging.error(f"No files found with mask: {file_mask}")
        return

    start_time = time()

    for video_path in video_files:
        try:
            temp_dir = os.path.join(temp_dir_base, os.path.splitext(os.path.basename(video_path))[0])
            os.makedirs(temp_dir, exist_ok=True)

            logging.info(f"Processing video: {video_path}")
            logging.info(f"Temporary directory: {temp_dir}")

            # Queues to monitor progress
            progress_queue_frames = Queue()
            progress_queue_transcription = Queue()

            # Create and start separate processes
            frame_process = Process(target=process_frames, args=(video_path, temp_dir, progress_queue_frames, batch_size))
            transcription_process = Process(target=process_transcription, args=(video_path, model_name, progress_queue_transcription))

            frame_process.start()
            transcription_process.start()

            # Monitor progress
            while frame_process.is_alive() or transcription_process.is_alive():
                if not progress_queue_frames.empty():
                    msg_frames = progress_queue_frames.get()
                    logging.info(f"Frames: {msg_frames}")

                if not progress_queue_transcription.empty():
                    msg_transcription = progress_queue_transcription.get()
                    logging.info(f"Transcription: {msg_transcription}")

            # Ensure both processes have completed
            frame_process.join()
            transcription_process.join()
        except Exception as e:
            logging.error(f"Error processing video {video_path}: {e}")

    total_time = time() - start_time
    logging.info(f"Process completed in {total_time:.2f} seconds.")

if __name__ == "__main__":
    main()
