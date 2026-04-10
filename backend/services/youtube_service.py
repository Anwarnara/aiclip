"""
YouTube Scraper Service
Search and download YouTube videos using YouTube Data API and yt-dlp
"""

import os
import re
import json
import subprocess
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime

# YouTube API key path
YT_API_KEY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "yt_api_key.txt"
)

# YouTube Video Categories (Indonesia region)
VIDEO_CATEGORIES = {
    "0": "All",
    "1": "Film & Animation",
    "2": "Autos & Vehicles",
    "10": "Music",
    "15": "Pets & Animals",
    "17": "Sports",
    "18": "Short Movies",
    "19": "Travel & Events",
    "20": "Gaming",
    "21": "Videoblogging",
    "22": "People & Blogs",
    "23": "Comedy",
    "24": "Entertainment",
    "25": "News & Politics",
    "26": "Howto & Style",
    "27": "Education",
    "28": "Science & Technology",
    "29": "Nonprofits & Activism",
    "30": "Movies",
    "31": "Anime/Animation",
    "32": "Action/Adventure",
    "33": "Classics",
    "34": "Comedy Films",
    "35": "Documentary",
    "36": "Drama",
    "37": "Family",
    "38": "Foreign",
    "39": "Horror",
    "40": "Sci-Fi/Fantasy",
    "41": "Thriller",
    "42": "Shorts",
    "43": "Shows",
    "44": "Trailers",
    # Custom categories (will add keyword to search)
    "podcast": "Podcast",
    "interview": "Interview",
    "talkshow": "Talk Show",
    "reaction": "Reaction",
    "review": "Review",
}

# Country/Region codes
REGION_CODES = {
    "ID": "Indonesia",
    "US": "United States",
    "GB": "United Kingdom",
    "JP": "Japan",
    "KR": "South Korea",
    "IN": "India",
    "BR": "Brazil",
    "DE": "Germany",
    "FR": "France",
    "RU": "Russia",
    "CA": "Canada",
    "AU": "Australia",
    "MX": "Mexico",
    "IT": "Italy",
    "ES": "Spain",
    "NL": "Netherlands",
    "PH": "Philippines",
    "TH": "Thailand",
    "VN": "Vietnam",
    "MY": "Malaysia",
    "SG": "Singapore",
    "TW": "Taiwan",
    "HK": "Hong Kong",
    "SA": "Saudi Arabia",
    "AE": "UAE",
    "EG": "Egypt",
    "TR": "Turkey",
    "PL": "Poland",
    "AR": "Argentina",
    "CL": "Chile",
    "CO": "Colombia",
}


def load_api_key() -> str:
    """Load YouTube API key from file"""
    try:
        if os.path.exists(YT_API_KEY_PATH):
            with open(YT_API_KEY_PATH, 'r') as f:
                return f.read().strip()
    except Exception:
        pass
    return ""


class YouTubeService:
    """Service for searching and downloading YouTube videos"""

    def __init__(self):
        self.api_key = load_api_key()
        self.base_url = "https://www.googleapis.com/youtube/v3"

    def _parse_duration(self, duration: str) -> int:
        """Parse ISO 8601 duration to seconds"""
        # PT1H2M3S -> 3723 seconds
        match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration)
        if not match:
            return 0
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)
        return hours * 3600 + minutes * 60 + seconds

    def _format_duration(self, seconds: int) -> str:
        """Format seconds to HH:MM:SS or MM:SS"""
        if seconds >= 3600:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            secs = seconds % 60
            return f"{hours}:{minutes:02d}:{secs:02d}"
        else:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{minutes}:{secs:02d}"

    def _format_view_count(self, count: str) -> str:
        """Format view count to readable format"""
        try:
            n = int(count)
            if n >= 1_000_000_000:
                return f"{n/1_000_000_000:.1f}B"
            elif n >= 1_000_000:
                return f"{n/1_000_000:.1f}M"
            elif n >= 1_000:
                return f"{n/1_000:.1f}K"
            return str(n)
        except:
            return count

    def search_videos(
        self,
        query: str,
        max_results: int = 20,
        duration_filter: str = "any",  # any, short (<4min), medium (4-20min), long (>20min)
        channel_id: Optional[str] = None,
        order: str = "relevance",  # relevance, date, viewCount, rating
        published_after: Optional[str] = None,  # ISO 8601 date
        category_id: Optional[str] = None,  # Video category/genre ID
        region_code: str = "ID",  # Country code (ID, US, GB, JP, etc.)
    ) -> Dict[str, Any]:
        """
        Search YouTube videos

        Args:
            query: Search query
            max_results: Maximum number of results (max 50)
            duration_filter: Filter by duration
            channel_id: Filter by channel ID
            order: Sort order
            published_after: Filter videos published after this date
            category_id: Filter by video category/genre

        Returns:
            Dict with 'videos' list and 'total_results'
        """
        import requests

        if not self.api_key:
            return {"error": "YouTube API key not configured", "videos": []}

        # Handle custom categories by adding keyword to query
        custom_categories = {
            "podcast": "podcast",
            "interview": "interview",
            "talkshow": "talk show",
            "reaction": "reaction",
            "review": "review"
        }

        search_query = query
        actual_category_id = None

        if category_id and category_id != "0":
            if category_id in custom_categories:
                # Add custom category keyword to search query
                search_query = f"{query} {custom_categories[category_id]}"
            else:
                # Use as YouTube category ID (numeric)
                actual_category_id = category_id

        # Build search params
        params = {
            "part": "snippet",
            "q": search_query,
            "type": "video",
            "maxResults": min(max_results, 50),
            "order": order,
            "key": self.api_key,
            "regionCode": region_code,  # Filter by country
        }

        # Duration filter
        if duration_filter == "short":
            params["videoDuration"] = "short"
        elif duration_filter == "medium":
            params["videoDuration"] = "medium"
        elif duration_filter == "long":
            params["videoDuration"] = "long"

        # Category/Genre filter (only for numeric YouTube category IDs)
        if actual_category_id:
            params["videoCategoryId"] = actual_category_id

        # Channel filter
        if channel_id:
            params["channelId"] = channel_id

        # Date filter
        if published_after:
            params["publishedAfter"] = published_after

        try:
            # Search request
            response = requests.get(
                f"{self.base_url}/search",
                params=params,
                timeout=30
            )

            if response.status_code != 200:
                return {"error": f"API error: {response.status_code}", "videos": []}

            data = response.json()
            video_ids = [item["id"]["videoId"] for item in data.get("items", [])]

            if not video_ids:
                return {"videos": [], "total_results": 0}

            # Get video details (duration, views, etc.)
            details_response = requests.get(
                f"{self.base_url}/videos",
                params={
                    "part": "snippet,contentDetails,statistics",
                    "id": ",".join(video_ids),
                    "key": self.api_key,
                },
                timeout=30
            )

            if details_response.status_code != 200:
                return {"error": f"Details API error: {details_response.status_code}", "videos": []}

            details_data = details_response.json()

            videos = []
            for item in details_data.get("items", []):
                snippet = item.get("snippet", {})
                content = item.get("contentDetails", {})
                stats = item.get("statistics", {})

                duration_seconds = self._parse_duration(content.get("duration", "PT0S"))

                # Get raw stats for AI analysis
                view_count = int(stats.get("viewCount", 0))
                like_count = int(stats.get("likeCount", 0))
                comment_count = int(stats.get("commentCount", 0))

                # Calculate engagement rate
                engagement_rate = 0
                if view_count > 0:
                    engagement_rate = ((like_count + comment_count) / view_count) * 100

                # Get category name
                category_id = snippet.get("categoryId", "0")
                category_name = VIDEO_CATEGORIES.get(category_id, "Unknown")

                # Get tags
                tags = snippet.get("tags", [])[:10]  # First 10 tags

                videos.append({
                    "id": item["id"],
                    "title": snippet.get("title", ""),
                    "description": snippet.get("description", ""),  # Full description
                    "channel": snippet.get("channelTitle", ""),
                    "channel_id": snippet.get("channelId", ""),
                    "thumbnail": snippet.get("thumbnails", {}).get("medium", {}).get("url", ""),
                    "duration": self._format_duration(duration_seconds),
                    "duration_seconds": duration_seconds,
                    # Formatted stats for display
                    "views": self._format_view_count(str(view_count)),
                    "likes": self._format_view_count(str(like_count)),
                    "comments": self._format_view_count(str(comment_count)),
                    # Raw stats for AI
                    "view_count": view_count,
                    "like_count": like_count,
                    "comment_count": comment_count,
                    "engagement_rate": round(engagement_rate, 2),
                    # Additional data
                    "published": snippet.get("publishedAt", "")[:10],
                    "published_full": snippet.get("publishedAt", ""),
                    "category_id": category_id,
                    "category_name": category_name,
                    "tags": tags,
                    "language": snippet.get("defaultLanguage", snippet.get("defaultAudioLanguage", "")),
                    "url": f"https://www.youtube.com/watch?v={item['id']}"
                })

            return {
                "videos": videos,
                "total_results": data.get("pageInfo", {}).get("totalResults", len(videos))
            }

        except Exception as e:
            return {"error": str(e), "videos": []}

    def get_channel_id(self, channel_name: str) -> Optional[str]:
        """Get channel ID from channel name or handle"""
        import requests

        if not self.api_key:
            return None

        # Try searching for channel
        try:
            response = requests.get(
                f"{self.base_url}/search",
                params={
                    "part": "snippet",
                    "q": channel_name,
                    "type": "channel",
                    "maxResults": 1,
                    "key": self.api_key,
                },
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                items = data.get("items", [])
                if items:
                    return items[0]["id"]["channelId"]
        except:
            pass

        return None

    def get_video_formats(self, video_url: str) -> List[Dict[str, Any]]:
        """
        Get available formats for a video using yt-dlp

        Returns list of formats with resolution, format_id, ext, filesize
        """
        try:
            cmd = [
                "yt-dlp",
                "--dump-json",
                "--no-download",
                video_url
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore',
                timeout=60
            )

            if result.returncode != 0:
                return []

            data = json.loads(result.stdout)
            formats = []
            seen_resolutions = set()

            for fmt in data.get("formats", []):
                # Only video formats with audio or combined
                vcodec = fmt.get("vcodec", "none")
                acodec = fmt.get("acodec", "none")
                height = fmt.get("height", 0)
                ext = fmt.get("ext", "")

                # Skip audio-only
                if vcodec == "none":
                    continue

                # Get resolution label
                if height:
                    resolution = f"{height}p"
                else:
                    resolution = fmt.get("format_note", "unknown")

                # Prefer mp4 and avoid duplicates
                if resolution in seen_resolutions and ext != "mp4":
                    continue

                # Get filesize
                filesize = fmt.get("filesize") or fmt.get("filesize_approx", 0)
                if filesize:
                    if filesize >= 1_000_000_000:
                        size_str = f"{filesize/1_000_000_000:.1f} GB"
                    elif filesize >= 1_000_000:
                        size_str = f"{filesize/1_000_000:.1f} MB"
                    else:
                        size_str = f"{filesize/1_000:.1f} KB"
                else:
                    size_str = "Unknown"

                formats.append({
                    "format_id": fmt.get("format_id", ""),
                    "resolution": resolution,
                    "ext": ext,
                    "filesize": size_str,
                    "has_audio": acodec != "none",
                    "height": height
                })

                seen_resolutions.add(resolution)

            # Sort by resolution (highest first)
            formats.sort(key=lambda x: x.get("height", 0), reverse=True)

            # Add "best" option
            formats.insert(0, {
                "format_id": "best",
                "resolution": "Best Quality",
                "ext": "mp4",
                "filesize": "Auto",
                "has_audio": True,
                "height": 9999
            })

            return formats

        except Exception as e:
            print(f"Error getting formats: {e}")
            return []

    def download_video(
        self,
        video_url: str,
        output_dir: str,
        format_id: str = "best",
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Download video using yt-dlp

        Args:
            video_url: YouTube video URL
            output_dir: Output directory
            format_id: Format ID to download
            progress_callback: Callback(percent, status)

        Returns:
            Dict with 'success', 'path', 'error'
        """
        os.makedirs(output_dir, exist_ok=True)

        # Build yt-dlp command
        output_template = os.path.join(output_dir, "%(title)s.%(ext)s")

        cmd = [
            "yt-dlp",
            "--no-playlist",
            "--newline",
            "--progress",
            "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "--retries", "3",
            "--fragment-retries", "3",
            "--sleep-interval", "1",
            "-o", output_template,
        ]

        # Format selection
        if format_id == "best":
            cmd.extend(["-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"])
        else:
            cmd.extend(["-f", f"{format_id}+bestaudio/best"])

        cmd.append(video_url)

        print(f"[YT-DLP] Starting download: {video_url}")
        print(f"[YT-DLP] Command: yt-dlp -f {format_id} ...")

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='ignore'
            )

            downloaded_path = None
            error_lines = []
            last_percent = -1

            for line in process.stdout:
                line = line.strip()

                # Log important lines
                if "[download]" in line or "[info]" in line or "ERROR" in line or "[Merger]" in line:
                    print(f"[YT-DLP] {line}")

                # Capture error lines
                if "ERROR" in line or "error" in line.lower():
                    error_lines.append(line)

                # Parse progress
                if "[download]" in line and "%" in line:
                    match = re.search(r'(\d+\.?\d*)%', line)
                    if match:
                        percent = float(match.group(1))
                        # Only log every 10%
                        if int(percent) // 10 > last_percent // 10:
                            last_percent = int(percent)
                        if progress_callback:
                            progress_callback(percent, f"Downloading: {percent:.1f}%")

                # Get downloaded file path
                if "[download] Destination:" in line:
                    downloaded_path = line.replace("[download] Destination:", "").strip()
                elif "has already been downloaded" in line:
                    match = re.search(r'\[download\] (.+?) has already', line)
                    if match:
                        downloaded_path = match.group(1)
                elif "[Merger]" in line or "Merging formats" in line:
                    if progress_callback:
                        progress_callback(95, "Merging audio and video...")

            process.wait()

            if process.returncode == 0:
                # Find the actual downloaded file
                if not downloaded_path or not os.path.exists(downloaded_path):
                    # Look for most recent .mp4 file in output_dir (exclude .part and other temp files)
                    files = [os.path.join(output_dir, f) for f in os.listdir(output_dir)]
                    # Only get complete video files (.mp4, .mkv, .webm) - not .part, .temp, etc.
                    video_extensions = ('.mp4', '.mkv', '.webm', '.avi', '.mov')
                    files = [f for f in files if os.path.isfile(f) and f.lower().endswith(video_extensions)]
                    if files:
                        downloaded_path = max(files, key=os.path.getmtime)
                    else:
                        print(f"[YT-DLP ERROR] No valid video file found in {output_dir}")
                        return {
                            "success": False,
                            "path": None,
                            "error": "No valid video file found after download"
                        }

                # Verify the file is valid (not .part and has size)
                if downloaded_path:
                    if downloaded_path.endswith('.part'):
                        print(f"[YT-DLP ERROR] Downloaded file is incomplete (.part): {downloaded_path}")
                        return {
                            "success": False,
                            "path": None,
                            "error": f"Download incomplete - file is .part: {os.path.basename(downloaded_path)}"
                        }
                    if os.path.getsize(downloaded_path) < 1024:
                        print(f"[YT-DLP ERROR] Downloaded file too small: {downloaded_path}")
                        return {
                            "success": False,
                            "path": None,
                            "error": f"Download failed - file too small: {os.path.basename(downloaded_path)}"
                        }

                if progress_callback:
                    progress_callback(100, "Download complete!")

                print(f"[YT-DLP] Success: {downloaded_path}")
                return {
                    "success": True,
                    "path": downloaded_path,
                    "error": None
                }
            else:
                error_msg = "; ".join(error_lines) if error_lines else f"Download failed (exit code {process.returncode})"
                print(f"[YT-DLP ERROR] Failed: {error_msg}")
                return {
                    "success": False,
                    "path": None,
                    "error": error_msg
                }

        except Exception as e:
            print(f"[YT-DLP ERROR] Exception: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "path": None,
                "error": str(e)
            }

    def get_ai_recommendations(
        self,
        videos: List[Dict[str, Any]],
        purpose: str = "viral clips"
    ) -> Dict[str, Any]:
        """
        Use AI to analyze videos and recommend the best ones for processing

        Args:
            videos: List of video data with stats
            purpose: What the videos will be used for

        Returns:
            Dict with 'recommendations' list
        """
        import requests

        if not videos:
            return {"recommendations": [], "error": "No videos to analyze"}

        # Build video summary for AI with ALL data
        video_summaries = []
        for i, v in enumerate(videos):
            tags_str = ", ".join(v.get('tags', [])[:5]) if v.get('tags') else "No tags"
            desc = v.get('description', '')[:300] if v.get('description') else "No description"

            video_summaries.append(
                f"{i+1}. \"{v['title']}\"\n"
                f"   Channel: {v['channel']}\n"
                f"   Category: {v.get('category_name', 'Unknown')}\n"
                f"   Duration: {v['duration']} ({v.get('duration_seconds', 0)} detik)\n"
                f"   Published: {v.get('published', 'Unknown')}\n"
                f"   Stats: Views={v['views']} | Likes={v['likes']} | Comments={v['comments']}\n"
                f"   Raw Stats: {v.get('view_count', 0):,} views | {v.get('like_count', 0):,} likes | {v.get('comment_count', 0):,} comments\n"
                f"   Engagement Rate: {v.get('engagement_rate', 0):.2f}%\n"
                f"   Tags: {tags_str}\n"
                f"   Language: {v.get('language', 'Unknown')}\n"
                f"   Description: {desc}"
            )

        videos_text = "\n\n".join(video_summaries)

        system_prompt = f"""Kamu adalah AI content strategist yang ahli memilih video untuk dijadikan {purpose}.

TUGAS: Analisis daftar video YouTube dan rekomendasikan 3-5 video TERBAIK untuk diproses.

KRITERIA PEMILIHAN:
1. ENGAGEMENT TINGGI - Video dengan likes/comments tinggi relatif terhadap views
2. DURASI IDEAL - Video dengan durasi 5-30 menit ideal untuk clip extraction
3. KONTEN MENARIK - Judul dan deskripsi menunjukkan konten yang viral-worthy
4. POTENSI CLIP - Video yang kemungkinan memiliki momen menarik untuk dipotong

OUTPUT FORMAT (JSON):
{{
  "recommendations": [
    {{
      "index": 1,
      "reason": "Alasan singkat mengapa video ini direkomendasikan",
      "score": 95,
      "clip_potential": "Tinggi/Sedang/Rendah"
    }}
  ],
  "summary": "Ringkasan singkat analisis"
}}

PENTING: index adalah nomor urut video (1-based). Output hanya JSON."""

        user_prompt = f"""Analisis video berikut dan rekomendasikan yang terbaik untuk {purpose}:

{videos_text}

Pilih 3-5 video terbaik. Output JSON saja."""

        try:
            api_url = "https://ultyweb.com/account/v1/chat/completions"
            api_key = "gam_master_u7w3k9x2m5q8r1t4y6p0s3v8n2b5j7h"

            response = requests.post(
                api_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gemini-2.5-flash",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "account_id": "auto"
                },
                timeout=60
            )

            if response.status_code != 200:
                return {"recommendations": [], "error": f"API error: {response.status_code}"}

            result = response.json()
            content = result['choices'][0]['message']['content']

            # Parse JSON from response
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                ai_result = json.loads(json_match.group())
                return ai_result
            else:
                return {"recommendations": [], "error": "Failed to parse AI response"}

        except Exception as e:
            return {"recommendations": [], "error": str(e)}


# Global instance
youtube_service = YouTubeService()
