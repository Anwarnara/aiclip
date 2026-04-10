"""
Whisper AI Transcriber Module
Uses OpenAI Whisper with CUDA for audio transcription
"""

import os
import whisper
import torch
from typing import Callable, Optional, List, Dict, Any


class WhisperTranscriber:
    """Transcribes audio using OpenAI Whisper with CUDA acceleration"""

    def __init__(self, model_name: str = "medium", device: str = "cuda", model_path: str = None):
        """
        Initialize the Whisper transcriber

        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
            device: Device to use (cuda, cpu)
            model_path: Optional path to folder containing model files
        """
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_path = model_path  # Folder containing .pt files
        self.model = None

    def load_model(self, progress_callback: Optional[Callable[[str], None]] = None) -> None:
        """Load the Whisper model from local path or download"""
        if progress_callback:
            progress_callback(f"Loading Whisper {self.model_name} model on {self.device}...")

        # Set CUDA optimizations before loading
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Check for local model file first
        if self.model_path:
            local_model_file = os.path.join(self.model_path, f"{self.model_name}.pt")
            if os.path.exists(local_model_file):
                if progress_callback:
                    progress_callback(f"Loading local model from {local_model_file}...")
                self.model = whisper.load_model(local_model_file, device=self.device)
            else:
                if progress_callback:
                    progress_callback(f"Local model not found, downloading {self.model_name}...")
                self.model = whisper.load_model(self.model_name, device=self.device)
        else:
            self.model = whisper.load_model(self.model_name, device=self.device)

        # Warm-up GPU for consistent speed
        if torch.cuda.is_available() and progress_callback:
            progress_callback("Warming up GPU...")
            # Run a small dummy inference to warm up cudnn
            import numpy as np
            dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
            try:
                with torch.no_grad():
                    mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(dummy_audio)).to(self.device)
                    _ = self.model.encoder(mel.unsqueeze(0))
                torch.cuda.synchronize()
            except:
                pass  # Ignore warm-up errors

        if progress_callback:
            progress_callback(f"Model loaded successfully!")
    
    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Transcribe audio file

        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'id' for Indonesian, 'en' for English)
            progress_callback: Callback function(progress_percent, status_message)

        Returns:
            Dictionary with transcription results including segments with timestamps
        """
        if self.model is None:
            self.load_model()

        if progress_callback:
            progress_callback(0, "Starting transcription...")

        # Clear CUDA cache and optimize for consistent speed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Set to high performance mode (doesn't affect accuracy)
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Use fp16 on CUDA for speed without losing accuracy
        use_fp16 = (self.device == "cuda")

        result = self.model.transcribe(
            audio_path,
            language=language,
            verbose=False,
            word_timestamps=True,
            task="transcribe",
            fp16=use_fp16
        )
        
        if progress_callback:
            progress_callback(100, "Transcription complete!")
        
        # Format segments
        segments = []
        for segment in result.get('segments', []):
            segments.append({
                'id': segment.get('id', 0),
                'start': segment.get('start', 0),
                'end': segment.get('end', 0),
                'text': segment.get('text', '').strip(),
                'words': segment.get('words', [])
            })
        
        return {
            'text': result.get('text', ''),
            'language': result.get('language', 'unknown'),
            'segments': segments,
            'duration': segments[-1]['end'] if segments else 0
        }
    
    def format_transcript_for_ai(self, transcription: Dict[str, Any]) -> str:
        """
        Format transcription for AI analysis with per-word timestamps for precise cutting

        Args:
            transcription: Transcription result from transcribe()

        Returns:
            Formatted string with per-word timestamps [MM:SS.s]word
        """
        lines = []
        for segment in transcription.get('segments', []):
            words = segment.get('words', [])

            if words:
                # Format with per-word timestamps
                word_parts = []
                for word in words:
                    word_start = word.get('start', segment['start'])
                    word_text = word.get('word', '').strip()
                    if word_text:
                        timestamp = self._format_timestamp_precise(word_start)
                        word_parts.append(f"[{timestamp}]{word_text}")
                lines.append(" ".join(word_parts))
            else:
                # Fallback to segment-level
                start = self._format_timestamp_precise(segment['start'])
                end = self._format_timestamp_precise(segment['end'])
                text = segment['text']
                lines.append(f"[{start} → {end}] {text}")

        return "\n".join(lines)

    def get_word_timestamps(self, transcription: Dict[str, Any]) -> List[Dict]:
        """
        Extract all word-level timestamps from transcription

        Args:
            transcription: Transcription result from transcribe()

        Returns:
            List of word dicts with 'word', 'start', 'end' keys
        """
        all_words = []
        for segment in transcription.get('segments', []):
            words = segment.get('words', [])
            for word in words:
                all_words.append({
                    'word': word.get('word', '').strip(),
                    'start': word.get('start', 0),
                    'end': word.get('end', 0)
                })
        return all_words

    def find_speech_gaps(self, transcription: Dict[str, Any], min_gap: float = 0.3) -> List[Dict]:
        """
        Find gaps/pauses in speech - ideal cut points

        Args:
            transcription: Transcription result
            min_gap: Minimum gap duration to consider as pause (seconds)

        Returns:
            List of gap dicts with 'start', 'end', 'duration'
        """
        words = self.get_word_timestamps(transcription)
        gaps = []

        for i in range(1, len(words)):
            gap_start = words[i-1]['end']
            gap_end = words[i]['start']
            gap_duration = gap_end - gap_start

            if gap_duration >= min_gap:
                gaps.append({
                    'start': gap_start,
                    'end': gap_end,
                    'duration': gap_duration,
                    'midpoint': (gap_start + gap_end) / 2,
                    'before_word': words[i-1]['word'],
                    'after_word': words[i]['word']
                })

        return gaps

    def find_nearest_cut_point(
        self,
        transcription: Dict[str, Any],
        target_time: float,
        search_range: float = 3.0,
        prefer_after: bool = True
    ) -> float:
        """
        Find the nearest natural cut point (speech gap) to a target time

        Args:
            transcription: Transcription result
            target_time: The approximate time we want to cut
            search_range: How far to search before/after target (seconds)
            prefer_after: If True, prefer gaps after target time

        Returns:
            Best cut point timestamp
        """
        gaps = self.find_speech_gaps(transcription, min_gap=0.2)

        # Find gaps within search range
        candidates = []
        for gap in gaps:
            distance = gap['midpoint'] - target_time
            if abs(distance) <= search_range:
                # Score: prefer longer gaps and closer to target
                score = gap['duration'] * 2 - abs(distance)
                if prefer_after and distance > 0:
                    score += 0.5  # Slight preference for gaps after target
                candidates.append({
                    'time': gap['midpoint'],
                    'score': score,
                    'gap': gap
                })

        if candidates:
            # Return the best scoring candidate
            best = max(candidates, key=lambda x: x['score'])
            return best['time']

        # If no gaps found, find the end of the nearest word
        words = self.get_word_timestamps(transcription)
        nearest_word_end = target_time

        min_distance = float('inf')
        for word in words:
            # Prefer cutting at word END (not start)
            distance = abs(word['end'] - target_time)
            if distance < min_distance:
                min_distance = distance
                nearest_word_end = word['end']

        return nearest_word_end
    
    def format_transcript_chunks(self, transcription: Dict[str, Any], chunk_minutes: int = 5) -> List[Dict]:
        """
        Split long transcripts into chunks for API processing
        
        Args:
            transcription: Transcription result
            chunk_minutes: Duration of each chunk in minutes
            
        Returns:
            List of chunk dictionaries with start_time, end_time, text
        """
        segments = transcription.get('segments', [])
        if not segments:
            return []
        
        chunk_seconds = chunk_minutes * 60
        chunks = []
        current_chunk_lines = []
        chunk_start = 0
        chunk_end = chunk_seconds
        
        for segment in segments:
            seg_start = segment['start']
            seg_end = segment['end']
            
            # Check if segment crosses chunk boundary
            if seg_start >= chunk_end and current_chunk_lines:
                # Save current chunk
                chunks.append({
                    'start_time': chunk_start,
                    'end_time': seg_start,
                    'start_formatted': self._format_timestamp_precise(chunk_start),
                    'end_formatted': self._format_timestamp_precise(seg_start),
                    'text': "\n".join(current_chunk_lines)
                })
                
                # Start new chunk
                current_chunk_lines = []
                chunk_start = seg_start
                chunk_end = chunk_start + chunk_seconds
            
            # Add segment to current chunk with precise timestamp
            start_fmt = self._format_timestamp_precise(seg_start)
            end_fmt = self._format_timestamp_precise(seg_end)
            current_chunk_lines.append(f"[{start_fmt} → {end_fmt}] {segment['text']}")
        
        # Add final chunk
        if current_chunk_lines:
            chunks.append({
                'start_time': chunk_start,
                'end_time': segments[-1]['end'],
                'start_formatted': self._format_timestamp_precise(chunk_start),
                'end_formatted': self._format_timestamp_precise(segments[-1]['end']),
                'text': "\n".join(current_chunk_lines)
            })
        
        return chunks

    def format_transcript_chunks_by_tokens(self, transcription: Dict[str, Any], max_tokens: int = 4000) -> List[Dict]:
        """
        Split long transcripts into chunks based on approximate token count.
        Uses per-word timestamps for precise AI cutting.

        Args:
            transcription: Transcription result
            max_tokens: Maximum tokens per chunk (approx 4 chars = 1 token)

        Returns:
            List of chunk dictionaries with start_time, end_time, text
        """
        segments = transcription.get('segments', [])
        if not segments:
            return []

        # Approximate: 1 token ≈ 4 characters
        max_chars = max_tokens * 4

        chunks = []
        current_chunk_lines = []
        current_chars = 0
        chunk_start = 0

        for segment in segments:
            seg_start = segment['start']
            seg_end = segment['end']
            words = segment.get('words', [])

            # Format line with per-word timestamps if available
            if words:
                # Format: [00:00.5]kata [00:00.8]kata [00:01.2]kata
                word_parts = []
                for word in words:
                    word_start = word.get('start', seg_start)
                    word_text = word.get('word', '').strip()
                    if word_text:
                        timestamp = self._format_timestamp_precise(word_start)
                        word_parts.append(f"[{timestamp}]{word_text}")
                line = " ".join(word_parts)
            else:
                # Fallback to segment-level timestamp
                start_fmt = self._format_timestamp_precise(seg_start)
                end_fmt = self._format_timestamp_precise(seg_end)
                line = f"[{start_fmt} → {end_fmt}] {segment['text']}"

            line_chars = len(line)

            # Check if adding this segment would exceed limit
            if current_chars + line_chars > max_chars and current_chunk_lines:
                # Save current chunk
                chunks.append({
                    'start_time': chunk_start,
                    'end_time': seg_start,
                    'start_formatted': self._format_timestamp_precise(chunk_start),
                    'end_formatted': self._format_timestamp_precise(seg_start),
                    'text': "\n".join(current_chunk_lines)
                })

                # Start new chunk
                current_chunk_lines = []
                current_chars = 0
                chunk_start = seg_start

            current_chunk_lines.append(line)
            current_chars += line_chars + 1  # +1 for newline

        # Add final chunk
        if current_chunk_lines:
            chunks.append({
                'start_time': chunk_start,
                'end_time': segments[-1]['end'],
                'start_formatted': self._format_timestamp_precise(chunk_start),
                'end_formatted': self._format_timestamp_precise(segments[-1]['end']),
                'text': "\n".join(current_chunk_lines)
            })

        return chunks

    def _format_timestamp_precise(self, seconds: float) -> str:
        """Format seconds to MM:SS.s format (precise to 0.1 second)"""
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes:02d}:{secs:04.1f}"
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds to MM:SS format"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get information about the current device"""
        info = {
            'device': self.device,
            'cuda_available': torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory'] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        
        return info
    
    def unload_model(self) -> None:
        """Unload model and release VRAM"""
        if self.model is not None:
            del self.model
            self.model = None

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def analyze_speaker_timeline(
        self,
        audio_path: str,
        transcription: Dict[str, Any],
        window_duration: float = 0.5
    ) -> List[Dict]:
        """
        Analyze audio to determine speaker timeline based on audio energy in stereo channels.
        Uses the transcription word timestamps to segment speaker activity.

        For stereo audio: left channel = speaker 1, right channel = speaker 2
        For mono audio: uses word timing patterns to estimate speaker changes

        Args:
            audio_path: Path to audio file
            transcription: Transcription result with word timestamps
            window_duration: Duration of analysis windows in seconds

        Returns:
            List of speaker segments: [{'start': float, 'end': float, 'speaker': 'left'|'right'|'both'}]
        """
        import numpy as np
        import subprocess
        import tempfile
        import os

        speaker_timeline = []

        try:
            # Extract audio to WAV using FFmpeg for consistent format
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                temp_wav = tmp.name

            # Extract audio keeping stereo if available
            cmd = [
                'ffmpeg', '-y', '-i', audio_path,
                '-vn', '-acodec', 'pcm_s16le', '-ar', '16000',
                temp_wav
            ]
            subprocess.run(cmd, capture_output=True, check=True)

            # Read the WAV file
            import wave
            with wave.open(temp_wav, 'rb') as wav:
                n_channels = wav.getnchannels()
                sample_rate = wav.getframerate()
                n_frames = wav.getnframes()
                audio_data = wav.readframes(n_frames)

            # Convert to numpy array
            audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)

            if n_channels == 2:
                # Stereo: split into left and right
                left_channel = audio[0::2]
                right_channel = audio[1::2]

                # Analyze energy in windows aligned with transcription segments
                samples_per_window = int(window_duration * sample_rate)

                words = self.get_word_timestamps(transcription)

                for word in words:
                    start_sample = int(word['start'] * sample_rate)
                    end_sample = int(word['end'] * sample_rate)

                    if start_sample >= len(left_channel) or end_sample > len(left_channel):
                        continue

                    # Get energy for this word's time range
                    left_energy = np.mean(np.abs(left_channel[start_sample:end_sample]))
                    right_energy = np.mean(np.abs(right_channel[start_sample:end_sample]))

                    # Determine dominant speaker
                    if left_energy > right_energy * 1.5:
                        speaker = 'left'
                    elif right_energy > left_energy * 1.5:
                        speaker = 'right'
                    else:
                        speaker = 'both'

                    # Add or extend segment
                    if speaker_timeline and speaker_timeline[-1]['speaker'] == speaker:
                        # Extend existing segment
                        speaker_timeline[-1]['end'] = word['end']
                    else:
                        # Start new segment
                        speaker_timeline.append({
                            'start': word['start'],
                            'end': word['end'],
                            'speaker': speaker
                        })

            else:
                # Mono audio: use pause detection to estimate speaker changes
                # Simple heuristic: longer pauses between words may indicate speaker switch
                words = self.get_word_timestamps(transcription)

                if words:
                    current_speaker = 'left'  # Start with left
                    segment_start = words[0]['start']

                    for i, word in enumerate(words):
                        if i > 0:
                            gap = word['start'] - words[i-1]['end']

                            # Long pause (> 1 second) may indicate speaker change
                            if gap > 1.0:
                                # End current segment
                                speaker_timeline.append({
                                    'start': segment_start,
                                    'end': words[i-1]['end'],
                                    'speaker': current_speaker
                                })

                                # Switch speaker and start new segment
                                current_speaker = 'right' if current_speaker == 'left' else 'left'
                                segment_start = word['start']

                    # Add final segment
                    if words:
                        speaker_timeline.append({
                            'start': segment_start,
                            'end': words[-1]['end'],
                            'speaker': current_speaker
                        })

            # Clean up temp file
            if os.path.exists(temp_wav):
                os.remove(temp_wav)

        except Exception as e:
            print(f"[TRANSCRIBER] Speaker analysis failed: {e}")
            # Return empty timeline - dynamic focus will be disabled
            return []

        return speaker_timeline


if __name__ == "__main__":
    # Test the transcriber
    transcriber = WhisperTranscriber(model_name="tiny", device="cuda")
    
    print("Device info:", transcriber.get_device_info())
    
    def on_progress(percent, status):
        print(f"{status} - {percent:.1f}%")
    
    # You can test with an audio file here
    # result = transcriber.transcribe("test.wav", progress_callback=on_progress)
    # print(transcriber.format_transcript_for_ai(result))
