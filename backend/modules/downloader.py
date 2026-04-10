"""
YouTube Video Downloader Module
Uses yt-dlp to download videos from YouTube
"""

import os
import re
import unicodedata
import yt_dlp
from typing import Callable, Optional, Dict, Any


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing emojis, special characters, and unsafe characters.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename safe for FFmpeg and Windows
    """
    # Remove file extension temporarily
    name, ext = os.path.splitext(filename)

    # Remove emojis and special unicode characters
    # Keep only ASCII letters, numbers, spaces, underscores, hyphens, and dots
    sanitized = ""
    for char in name:
        # Check if character is a basic ASCII printable character
        if ord(char) < 128:
            # Allow alphanumeric, space, underscore, hyphen, dot
            if char.isalnum() or char in ' _-.':
                sanitized += char
        else:
            # Try to normalize unicode characters (é -> e, etc.)
            normalized = unicodedata.normalize('NFKD', char)
            ascii_char = normalized.encode('ascii', 'ignore').decode('ascii')
            # Only add if it's alphanumeric or allowed punctuation
            for c in ascii_char:
                if c.isalnum() or c in ' _-.':
                    sanitized += c

    # Replace multiple spaces/underscores with single
    sanitized = re.sub(r'[\s_]+', '_', sanitized)

    # Remove leading/trailing underscores and dots
    sanitized = sanitized.strip('_.')

    # Ensure filename is not empty
    if not sanitized:
        sanitized = "video"

    # Limit length to 200 characters
    if len(sanitized) > 200:
        sanitized = sanitized[:200]

    return sanitized + ext


class YouTubeDownloader:
    """Downloads YouTube videos using yt-dlp"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def get_video_info(self, url: str) -> Dict[str, Any]:
        """Get video information without downloading"""
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return {
                'title': info.get('title', 'Unknown'),
                'duration': info.get('duration', 0),
                'thumbnail': info.get('thumbnail', ''),
                'uploader': info.get('uploader', 'Unknown'),
                'view_count': info.get('view_count', 0),
                'description': info.get('description', ''),
            }
    
    def download(
        self, 
        url: str, 
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> str:
        """
        Download video from YouTube
        
        Args:
            url: YouTube video URL
            progress_callback: Callback function(progress_percent, status_message)
            
        Returns:
            Path to downloaded video file
        """
        output_template = os.path.join(self.output_dir, '%(title)s.%(ext)s')
        
        def progress_hook(d):
            if d['status'] == 'downloading':
                total = d.get('total_bytes') or d.get('total_bytes_estimate', 0)
                downloaded = d.get('downloaded_bytes', 0)
                speed = d.get('speed', 0)
                eta = d.get('eta', 0)

                if total > 0:
                    percent = (downloaded / total) * 100
                    speed_str = f"{speed / 1024 / 1024:.2f} MB/s" if speed else "..."
                    eta_str = f"{eta}s" if eta else "..."
                    downloaded_str = f"{downloaded / 1024 / 1024:.1f}MB"
                    total_str = f"{total / 1024 / 1024:.1f}MB"

                    print(f"\r[YT-DLP] Downloading: {percent:.1f}% | {downloaded_str}/{total_str} | Speed: {speed_str} | ETA: {eta_str}   ", end="", flush=True)

                    if progress_callback:
                        progress_callback(percent, f"Downloading: {speed_str}")
                else:
                    print(f"\r[YT-DLP] Downloading: {downloaded / 1024 / 1024:.1f}MB downloaded...   ", end="", flush=True)

            elif d['status'] == 'finished':
                filename = d.get('filename', 'unknown')
                print(f"\n[YT-DLP] Download finished: {os.path.basename(filename)}")
                if progress_callback:
                    progress_callback(100, "Download complete, processing...")
            elif d['status'] == 'error':
                print(f"\n[YT-DLP ERROR] Download failed: {d.get('error', 'Unknown error')}")

        print(f"\n[YT-DLP] Starting download: {url}")
        print(f"[YT-DLP] Output dir: {self.output_dir}")

        ydl_opts = {
            'format': 'bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best',
            'outtmpl': output_template,
            'progress_hooks': [progress_hook],
            'merge_output_format': 'mp4',
            'quiet': False,
            'no_warnings': False,
            'verbose': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"[YT-DLP] Extracting video info...")
            info = ydl.extract_info(url, download=True)
            print(f"[YT-DLP] Video title: {info.get('title', 'Unknown')}")
            print(f"[YT-DLP] Duration: {info.get('duration', 0)} seconds")
            # Get the actual output filename
            filename = ydl.prepare_filename(info)
            # Ensure it has .mp4 extension
            if not filename.endswith('.mp4'):
                base = os.path.splitext(filename)[0]
                filename = base + '.mp4'

            # Sanitize filename to remove emojis and special characters
            dirname = os.path.dirname(filename)
            basename = os.path.basename(filename)
            sanitized_basename = sanitize_filename(basename)

            if sanitized_basename != basename:
                sanitized_path = os.path.join(dirname, sanitized_basename)
                if os.path.exists(filename):
                    os.rename(filename, sanitized_path)
                    filename = sanitized_path

            return filename
    
    def download_audio_only(
        self, 
        url: str, 
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> str:
        """
        Download only audio from YouTube (for faster transcription)
        
        Args:
            url: YouTube video URL
            progress_callback: Callback function(progress_percent, status_message)
            
        Returns:
            Path to downloaded audio file
        """
        output_template = os.path.join(self.output_dir, '%(title)s_audio.%(ext)s')
        
        def progress_hook(d):
            if d['status'] == 'downloading':
                total = d.get('total_bytes') or d.get('total_bytes_estimate', 0)
                downloaded = d.get('downloaded_bytes', 0)
                speed = d.get('speed', 0)

                if total > 0:
                    percent = (downloaded / total) * 100
                    speed_str = f"{speed / 1024 / 1024:.2f} MB/s" if speed else "..."
                    print(f"\r[YT-DLP AUDIO] Downloading: {percent:.1f}% | Speed: {speed_str}   ", end="", flush=True)
                    if progress_callback:
                        progress_callback(percent, "Downloading audio...")
                else:
                    print(f"\r[YT-DLP AUDIO] Downloading: {downloaded / 1024 / 1024:.1f}MB   ", end="", flush=True)

            elif d['status'] == 'finished':
                print(f"\n[YT-DLP AUDIO] Download finished, converting to WAV...")
                if progress_callback:
                    progress_callback(100, "Audio download complete")

        print(f"\n[YT-DLP AUDIO] Starting audio download: {url}")

        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_template,
            'progress_hooks': [progress_hook],
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'quiet': False,
            'no_warnings': False,
            'verbose': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"[YT-DLP AUDIO] Extracting info...")
            info = ydl.extract_info(url, download=True)
            print(f"[YT-DLP AUDIO] Title: {info.get('title', 'Unknown')}")
            filename = ydl.prepare_filename(info)
            # Replace extension with wav
            base = os.path.splitext(filename)[0]
            wav_filename = base + '.wav'

            # Sanitize filename to remove emojis and special characters
            dirname = os.path.dirname(wav_filename)
            basename = os.path.basename(wav_filename)
            sanitized_basename = sanitize_filename(basename)

            if sanitized_basename != basename:
                sanitized_path = os.path.join(dirname, sanitized_basename)
                if os.path.exists(wav_filename):
                    os.rename(wav_filename, sanitized_path)
                    wav_filename = sanitized_path

            return wav_filename


if __name__ == "__main__":
    # Test the downloader
    downloader = YouTubeDownloader("./test_output")
    
    def on_progress(percent, status):
        print(f"\r{status} - {percent:.1f}%", end="")
    
    # Test with a short video
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    info = downloader.get_video_info(test_url)
    print(f"Video: {info['title']}, Duration: {info['duration']}s")
