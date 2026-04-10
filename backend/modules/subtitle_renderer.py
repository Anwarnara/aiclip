"""
Subtitle Renderer Module
Renders karaoke-style animated subtitles on video frames
Like TikTok/CapCut auto-captions
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import os


# Custom font folder path
CUSTOM_FONT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "font")


def get_available_fonts() -> List[Dict[str, str]]:
    """
    Get list of available fonts from custom font folder and system.
    Returns list of dicts with 'name' and 'path' keys.
    """
    fonts = []

    # Custom fonts from project font folder
    if os.path.exists(CUSTOM_FONT_DIR):
        for filename in os.listdir(CUSTOM_FONT_DIR):
            if filename.lower().endswith(('.ttf', '.otf')):
                font_name = os.path.splitext(filename)[0]
                font_path = os.path.join(CUSTOM_FONT_DIR, filename)
                fonts.append({
                    'name': font_name,
                    'path': font_path,
                    'source': 'custom'
                })

    # System fonts (Windows)
    system_fonts = [
        ('Arial Bold', 'C:/Windows/Fonts/arialbd.ttf'),
        ('Arial', 'C:/Windows/Fonts/arial.ttf'),
        ('Impact', 'C:/Windows/Fonts/impact.ttf'),
        ('Segoe UI Bold', 'C:/Windows/Fonts/segoeuib.ttf'),
        ('Segoe UI', 'C:/Windows/Fonts/segoeui.ttf'),
        ('Comic Sans MS', 'C:/Windows/Fonts/comic.ttf'),
        ('Verdana Bold', 'C:/Windows/Fonts/verdanab.ttf'),
    ]

    for name, path in system_fonts:
        if os.path.exists(path):
            fonts.append({
                'name': name,
                'path': path,
                'source': 'system'
            })

    return fonts


class SubtitleRenderer:
    """Renders animated karaoke-style subtitles on video frames"""

    def __init__(
        self,
        frame_width: int = 1080,
        frame_height: int = 1920,
        font_size: int = 48,
        font_path: str = None,  # Custom font path
        position: int = 85,  # 0-100 percentage from top (85 = near bottom)
        style: str = "uppercase",  # "uppercase" or "bold"
        text_color: str = "#FFFFFF",
        highlight_color: str = "#FFFF00",
        bg_enabled: bool = True,
        bg_color: str = "#000000",
        bg_opacity: float = 0.5,
        max_words_per_line: int = 5  # Maximum words per line
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.font_size = font_size
        self.font_path = font_path
        self.position = position
        self.style = style
        self.text_color = self._hex_to_rgb(text_color)
        self.highlight_color = self._hex_to_rgb(highlight_color)
        self.bg_enabled = bg_enabled
        self.bg_color = self._hex_to_rgb(bg_color)
        self.bg_opacity = bg_opacity
        self.max_words_per_line = max_words_per_line

        # Load font
        self.font = None
        self.font_bold = None
        self._load_fonts()

        # Cache for segments (built from word timestamps)
        self.segments: List[Dict] = []

    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def _load_fonts(self):
        """Load fonts for rendering"""
        # Priority: custom font path > custom font folder > system fonts
        font_paths = []

        # 1. Use specified font path if provided
        if self.font_path and os.path.exists(self.font_path):
            font_paths.append(self.font_path)

        # 2. Custom fonts from project folder
        if os.path.exists(CUSTOM_FONT_DIR):
            for filename in os.listdir(CUSTOM_FONT_DIR):
                if filename.lower().endswith(('.ttf', '.otf')):
                    font_paths.append(os.path.join(CUSTOM_FONT_DIR, filename))

        # 3. System fonts
        font_paths.extend([
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/impact.ttf",
            "C:/Windows/Fonts/seguisb.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ])

        for path in font_paths:
            if os.path.exists(path):
                try:
                    self.font = ImageFont.truetype(path, self.font_size)
                    # Try to load bold version
                    if 'bd' not in path.lower() and 'bold' not in path.lower():
                        bold_path = path.replace('.ttf', 'bd.ttf').replace('Sans.ttf', 'Sans-Bold.ttf')
                        if os.path.exists(bold_path):
                            self.font_bold = ImageFont.truetype(bold_path, self.font_size)
                        else:
                            self.font_bold = self.font
                    else:
                        self.font_bold = self.font
                    break
                except:
                    continue

        if self.font is None:
            self.font = ImageFont.load_default()
            self.font_bold = self.font

    def build_segments_from_words(self, words: List[Dict]) -> List[Dict]:
        """
        Build display segments from word timestamps.
        Each segment contains multiple words that will be displayed together.
        Splits into lines of max_words_per_line words each.

        Args:
            words: List of word dicts with 'word', 'start', 'end' keys

        Returns:
            List of segments, each containing:
            - words: list of words in this segment
            - start: start time of segment
            - end: end time of segment
            - lines: list of line strings for display
        """
        if not words:
            return []

        segments = []
        current_words = []

        for word in words:
            word_text = word.get('word', '').strip()
            if not word_text:
                continue

            current_words.append({
                'word': word_text,
                'start': word.get('start', 0),
                'end': word.get('end', 0)
            })

            # Create segment when we have enough words or hit punctuation
            is_sentence_end = word_text.endswith(('.', '!', '?', ',', ':', ';'))
            has_enough_words = len(current_words) >= self.max_words_per_line * 2

            if is_sentence_end or has_enough_words:
                if current_words:
                    segment = self._create_segment(current_words)
                    segments.append(segment)
                    current_words = []

        # Don't forget remaining words
        if current_words:
            segment = self._create_segment(current_words)
            segments.append(segment)

        self.segments = segments
        return segments

    def _create_segment(self, words: List[Dict]) -> Dict:
        """Create a display segment from a list of words"""
        # Split into lines
        lines = []
        current_line = []

        for word in words:
            current_line.append(word)
            if len(current_line) >= self.max_words_per_line:
                lines.append(current_line)
                current_line = []

        if current_line:
            lines.append(current_line)

        return {
            'words': words,
            'lines': lines,
            'start': words[0]['start'],
            'end': words[-1]['end']
        }

    def get_segment_for_time(
        self,
        current_time: float,
        clip_start: float
    ) -> Tuple[Optional[Dict], int, int]:
        """
        Get the segment to display for current time.

        Args:
            current_time: Current frame time (relative to clip start)
            clip_start: Start time of clip in original video

        Returns:
            Tuple of (segment, current_word_index_in_segment, current_line_index)
            Returns (None, -1, -1) if no segment should be displayed
        """
        abs_time = clip_start + current_time

        for segment in self.segments:
            # Check if we're within this segment's time range
            # Add small buffer before segment starts for smoother transition
            if segment['start'] - 0.1 <= abs_time <= segment['end'] + 0.3:
                # Find which word is currently being spoken
                current_word_idx = -1
                current_line_idx = 0

                for i, word in enumerate(segment['words']):
                    if word['start'] <= abs_time:
                        current_word_idx = i

                # Find which line the current word is on
                if current_word_idx >= 0:
                    word_count = 0
                    for line_idx, line in enumerate(segment['lines']):
                        if word_count + len(line) > current_word_idx:
                            current_line_idx = line_idx
                            break
                        word_count += len(line)

                return segment, current_word_idx, current_line_idx

        return None, -1, -1

    def render_subtitle(
        self,
        frame: np.ndarray,
        current_time: float,
        clip_start: float
    ) -> np.ndarray:
        """
        Render subtitle on frame for current time.

        Args:
            frame: Video frame (numpy array, BGR)
            current_time: Current frame time (relative to clip start)
            clip_start: Start time of clip in original video

        Returns:
            Frame with subtitle rendered
        """
        segment, current_word_idx, current_line_idx = self.get_segment_for_time(
            current_time, clip_start
        )

        if segment is None:
            return frame

        # Convert to PIL for text rendering (keep as RGB, no alpha)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_image)

        # Build lines with word index tracking
        lines_to_render = []
        global_word_idx = 0

        for line_idx, line_words in enumerate(segment['lines']):
            line_parts = []
            for word in line_words:
                is_current = (global_word_idx == current_word_idx)
                word_text = word['word']

                if is_current:
                    if self.style == 'uppercase':
                        word_text = word_text.upper()
                    color = self.highlight_color  # Yellow tuple (255, 255, 0)
                    font = self.font_bold if self.style == 'bold' else self.font
                else:
                    # Keep lowercase for non-current words
                    word_text = word_text.lower()
                    color = self.text_color  # White tuple (255, 255, 255)
                    font = self.font

                line_parts.append({
                    'text': word_text,
                    'color': color,
                    'font': font,
                    'is_current': is_current
                })
                global_word_idx += 1

            lines_to_render.append(line_parts)

        if not lines_to_render:
            return frame

        # Calculate dimensions for each line
        line_heights = []
        line_widths = []

        for line_parts in lines_to_render:
            line_width = 0
            max_height = 0
            for part in line_parts:
                bbox = draw.textbbox((0, 0), part['text'] + ' ', font=part['font'])
                part['width'] = bbox[2] - bbox[0]
                part['height'] = bbox[3] - bbox[1]
                line_width += part['width']
                max_height = max(max_height, part['height'])
            line_widths.append(line_width)
            line_heights.append(max_height)

        total_height = sum(line_heights) + (len(lines_to_render) - 1) * 10  # 10px line spacing
        max_width = max(line_widths) if line_widths else 0

        # Calculate Y position based on percentage (0 = top, 100 = bottom)
        # Position indicates where the CENTER of the subtitle should be
        position_percent = self.position / 100.0
        y_center = int(self.frame_height * position_percent)
        y_start = y_center - (total_height // 2)

        # Draw background if enabled (semi-transparent, only in rectangle area)
        if self.bg_enabled and max_width > 0:
            padding_x = 30
            padding_y = 15
            bg_x1 = max(0, (self.frame_width - max_width) // 2 - padding_x)
            bg_y1 = max(0, y_start - padding_y)
            bg_x2 = min(self.frame_width, (self.frame_width + max_width) // 2 + padding_x)
            bg_y2 = min(self.frame_height, y_start + total_height + padding_y)

            # Convert to numpy for blending just the background region
            frame_np = np.array(pil_image)

            # Extract the region where background will be
            region = frame_np[bg_y1:bg_y2, bg_x1:bg_x2].copy()

            # Create background color overlay for just this region
            bg_color_rgb = np.array(self.bg_color, dtype=np.uint8)
            bg_overlay = np.full_like(region, bg_color_rgb)

            # Blend only this region
            alpha = self.bg_opacity
            blended_region = cv2.addWeighted(region, 1 - alpha, bg_overlay, alpha, 0)

            # Put blended region back
            frame_np[bg_y1:bg_y2, bg_x1:bg_x2] = blended_region

            # Convert back to PIL
            pil_image = Image.fromarray(frame_np)
            draw = ImageDraw.Draw(pil_image)

        # Draw each line
        y_cursor = y_start
        for line_idx, line_parts in enumerate(lines_to_render):
            line_width = line_widths[line_idx]
            x_cursor = (self.frame_width - line_width) // 2

            for part in line_parts:
                text_to_draw = part['text'] + ' '
                color_rgb = part['color']  # Already RGB tuple

                # Draw shadow (black outline for visibility)
                shadow_color = (0, 0, 0)
                for dx in [-2, -1, 0, 1, 2]:
                    for dy in [-2, -1, 0, 1, 2]:
                        if dx != 0 or dy != 0:
                            draw.text(
                                (x_cursor + dx, y_cursor + dy),
                                text_to_draw,
                                font=part['font'],
                                fill=shadow_color
                            )

                # Draw main text
                draw.text(
                    (x_cursor, y_cursor),
                    text_to_draw,
                    font=part['font'],
                    fill=color_rgb
                )
                x_cursor += part['width']

            y_cursor += line_heights[line_idx] + 10  # 10px line spacing

        # Convert back to OpenCV BGR
        result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return result

    def update_settings(
        self,
        font_size: int = None,
        position: int = None,
        style: str = None,
        text_color: str = None,
        highlight_color: str = None,
        bg_enabled: bool = None,
        bg_color: str = None,
        bg_opacity: float = None
    ):
        """Update renderer settings"""
        if font_size is not None and font_size != self.font_size:
            self.font_size = font_size
            self._load_fonts()

        if position is not None:
            self.position = position

        if style is not None:
            self.style = style

        if text_color is not None:
            self.text_color = self._hex_to_rgb(text_color)

        if highlight_color is not None:
            self.highlight_color = self._hex_to_rgb(highlight_color)

        if bg_enabled is not None:
            self.bg_enabled = bg_enabled

        if bg_color is not None:
            self.bg_color = self._hex_to_rgb(bg_color)

        if bg_opacity is not None:
            self.bg_opacity = bg_opacity
