"""
AI Clip Analyzer Module
Uses Gemini AI to find interesting clips from transcription
"""

import requests
import json
import re
from typing import List, Dict, Any, Optional, Callable


class ClipAnalyzer:
    """Analyzes transcription to find interesting clip segments using AI"""

    def __init__(
        self,
        api_url: str = "http://104.234.26.223:8090/v1/messages",
        api_key: str = "test",
        model: str = "gemini-3-pro-high",
        ai_type: str = "A"  # "A" = Anthropic format, "B" = Gemini raw format
    ):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.ai_type = ai_type
        self._transcription_data = None  # Store for cut point adjustment

    def set_transcription_data(self, transcription: Dict[str, Any]):
        """Store transcription data for precise cut point finding"""
        self._transcription_data = transcription
    
    def analyze_long_video(
        self,
        transcript_chunks: List[Dict],
        video_duration: float,
        min_clip_duration: int = 15,
        max_clip_duration: int = 60,
        clips_per_chunk: int = 2,
        cooldown_seconds: int = 2,
        auto_clip_count: bool = False,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze very long videos by processing chunks separately

        Args:
            transcript_chunks: List of chunk dicts from format_transcript_chunks()
            video_duration: Total video duration
            min_clip_duration: Min clip length in seconds
            max_clip_duration: Max clip length in seconds
            clips_per_chunk: Number of clips to find per chunk (ignored if auto_clip_count=True)
            cooldown_seconds: Seconds to wait between chunk requests
            auto_clip_count: If True, let AI decide number of clips per chunk
            progress_callback: Progress callback

        Returns:
            Combined list of clips from all chunks
        """
        import time

        all_clips = []
        total_chunks = len(transcript_chunks)

        for i, chunk in enumerate(transcript_chunks):
            if progress_callback:
                progress = (i / total_chunks) * 90
                progress_callback(progress, f"Analyzing chunk {i+1}/{total_chunks} ({chunk['start_formatted']} - {chunk['end_formatted']})")

            try:
                chunk_clips = self.analyze(
                    transcript=chunk['text'],
                    video_duration=chunk['end_time'],
                    min_clip_duration=min_clip_duration,
                    max_clip_duration=max_clip_duration,
                    num_clips=clips_per_chunk,
                    auto_clip_count=auto_clip_count,
                    progress_callback=None  # Don't pass callback to avoid nested progress
                )
                if progress_callback:
                    progress_callback(progress, f"Chunk {i+1}: Found {len(chunk_clips)} clips")
                all_clips.extend(chunk_clips)

                # Cooldown between chunks (except for last chunk)
                if cooldown_seconds > 0 and i < total_chunks - 1:
                    if progress_callback:
                        progress_callback(progress, f"Cooldown {cooldown_seconds}s before next chunk...")
                    time.sleep(cooldown_seconds)

            except Exception as e:
                error_msg = str(e)[:100]
                if progress_callback:
                    progress_callback(progress, f"Chunk {i+1} error: {error_msg}")
                # Log full error for debugging
                print(f"[ANALYZER ERROR] Chunk {i+1}: {str(e)}")

        if progress_callback:
            progress_callback(95, f"Adjusting {len(all_clips)} clip cut points...")

        # Re-adjust all clip cut points with full transcription data
        if self._transcription_data:
            adjusted_clips = []
            for clip in all_clips:
                adj_start, adj_end = self._adjust_cut_points(
                    clip['start'], clip['end']
                )
                clip['start'] = adj_start
                clip['end'] = adj_end
                clip['start_formatted'] = self._format_duration(adj_start)
                clip['end_formatted'] = self._format_duration(adj_end)
                clip['duration'] = adj_end - adj_start
                adjusted_clips.append(clip)
            all_clips = adjusted_clips

        if progress_callback:
            progress_callback(100, f"Found {len(all_clips)} clips from {total_chunks} chunks")

        # Sort by start time
        all_clips.sort(key=lambda c: c['start'])

        return all_clips
    
    def analyze(
        self,
        transcript: str,
        video_duration: float,
        min_clip_duration: int = 15,
        max_clip_duration: int = 60,
        num_clips: int = 5,
        auto_clip_count: bool = False,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze transcript to find interesting clips

        Args:
            transcript: Formatted transcript with timestamps
            video_duration: Total video duration in seconds
            min_clip_duration: Minimum clip duration in seconds
            max_clip_duration: Maximum clip duration in seconds
            num_clips: Number of clips to find (if not auto)
            auto_clip_count: If True, let AI decide number of clips
            progress_callback: Callback function(progress_percent, status_message)

        Returns:
            List of clip dictionaries with start, end, title, reason
        """
        if progress_callback:
            progress_callback(0, "Analyzing transcript with AI...")

        # Determine clip count instruction
        if auto_clip_count:
            clip_count_instruction = "temukan SEMUA bagian yang menurutmu viral dan menarik (tidak dibatasi jumlah, tapi pastikan kualitas tinggi)"
        else:
            clip_count_instruction = f"temukan {num_clips} bagian paling menarik"

        system_prompt = f"""Kamu adalah AI editor video profesional untuk TikTok/Reels/Shorts.

TUGAS: {clip_count_instruction} dari transkrip berikut.

🎯 PRIORITAS #1: KONTEKS LENGKAP
Penonton harus LANGSUNG PAHAM apa yang terjadi tanpa perlu menonton video penuh.
Durasi {min_clip_duration}-{max_clip_duration} detik adalah SARAN FLEKSIBEL.

📋 FORMAT TRANSKRIP:
Setiap kata punya timestamp: [MM:SS.s]kata
Contoh: [00:05.2]Halo [00:05.8]semuanya

🔑 ATURAN PEMOTONGAN YANG BENAR:

1️⃣ MULAI CLIP DENGAN KONTEKS:
   ❌ SALAH: "...jadi akhirnya saya nge-grab"
   ✅ BENAR: "Saya seorang dokter spesialis, tapi gaji saya cuma 1.5 juta, jadi akhirnya saya nge-grab"

   - Jika ada cerita, MULAI dari awal cerita (siapa, apa, mengapa)
   - Jika ada pertanyaan, SERTAKAN pertanyaan + jawaban lengkap
   - Jika ada konflik, SERTAKAN setup + klimaks + resolusi

2️⃣ AKHIRI CLIP DENGAN KESIMPULAN:
   ❌ SALAH: "...terus akhirnya..." (menggantung)
   ✅ BENAR: "...terus akhirnya saya berhasil menolong ibu itu melahirkan" (lengkap)

   - Pastikan ada ending yang jelas (kesimpulan/reaksi/punchline)
   - Jangan potong di tengah penjelasan

3️⃣ CLIP HARUS MANDIRI:
   Tanyakan: "Jika penonton HANYA melihat clip ini, apakah mereka paham?"
   - Siapa yang berbicara harus jelas
   - Topik apa yang dibahas harus jelas
   - Tidak ada referensi ke "tadi", "itu", "yang saya bilang sebelumnya"

🧠 CARA MENENTUKAN START & END:

LANGKAH 1: Temukan MOMEN VIRAL (bagian paling menarik)
LANGKAH 2: Mundur ke BELAKANG untuk menemukan AWAL KONTEKS
   - Kapan topik ini mulai dibahas?
   - Apa setup/latar belakang yang diperlukan?
LANGKAH 3: Maju ke DEPAN untuk menemukan AKHIR LENGKAP
   - Apakah ada kesimpulan/reaksi?
   - Apakah cerita sudah selesai?

📤 OUTPUT (JSON array saja):
[
  {{
    "start": "MM:SS.s",
    "end": "MM:SS.s",
    "title": "Judul singkat menarik (max 50 karakter)",
    "reason": "Kenapa clip ini menarik + konfirmasi konteks lengkap"
  }}
]

⚠️ CHECKLIST WAJIB untuk setiap clip:
✅ Penonton langsung paham siapa/apa/mengapa?
✅ Tidak ada kalimat yang terpotong?
✅ Ada kesimpulan/ending yang jelas?
✅ Bisa dipahami TANPA konteks dari luar clip?"""

        user_prompt = f"""📹 VIDEO DURASI: {self._format_duration(video_duration)}

💡 DURASI CLIP: {min_clip_duration}-{max_clip_duration} detik (fleksibel, PRIORITASKAN KONTEKS LENGKAP)

📝 TRANSKRIP:
{transcript}

📋 INSTRUKSI PENTING:

1. BACA KESELURUHAN transkrip untuk memahami konteks

2. IDENTIFIKASI momen yang menarik/viral

3. Untuk setiap momen, TENTUKAN START dengan mundur ke belakang:
   - Kapan topik/cerita ini MULAI dibahas?
   - Apa latar belakang yang DIPERLUKAN penonton?

4. TENTUKAN END dengan maju ke depan:
   - Di mana KESIMPULAN/ending dari cerita ini?
   - Apakah ada REAKSI yang perlu disertakan?

5. VERIFIKASI setiap clip:
   - Jika penonton HANYA melihat clip ini, apakah mereka PAHAM?
   - Apakah ada kalimat yang TERPOTONG?
   - Apakah cerita LENGKAP?

⚠️ JANGAN buat clip yang dimulai dengan "jadi", "tapi", "terus" tanpa konteks sebelumnya!

Output JSON array saja."""

        if progress_callback:
            progress_callback(30, "Sending to AI...")
        
        try:
            response = self._call_api(system_prompt, user_prompt)
            
            if progress_callback:
                progress_callback(70, "Parsing AI response...")
            
            clips = self._parse_response(response, video_duration)
            
            if progress_callback:
                progress_callback(100, f"Found {len(clips)} interesting clips!")
            
            return clips
            
        except Exception as e:
            if progress_callback:
                progress_callback(100, f"Error: {str(e)}")
            raise
    
    def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        """Call the AI API with retry logic. Supports AI A (Anthropic) and AI B (Gemini raw)"""
        import time

        # Build headers and payload based on AI type
        if self.ai_type == "B":
            # AI B: Gemini raw format (OpenAI-style endpoint)
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "raw_response": True
            }
        else:
            # AI A: Anthropic format - requires max_tokens
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01"
            }
            payload = {
                "model": self.model,
                "max_tokens": 50000,
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": user_prompt}
                ]
            }

        # Log request size for debugging
        payload_size = len(json.dumps(payload))
        system_tokens = len(system_prompt) // 4
        user_tokens = len(user_prompt) // 4
        total_tokens = system_tokens + user_tokens
        print(f"[ANALYZER API-{self.ai_type}] Sending request:")
        print(f"[ANALYZER API-{self.ai_type}]   - Model: {self.model}")
        print(f"[ANALYZER API-{self.ai_type}]   - URL: {self.api_url}")
        print(f"[ANALYZER API-{self.ai_type}]   - System prompt: {system_tokens:,} tokens")
        print(f"[ANALYZER API-{self.ai_type}]   - User prompt: {user_tokens:,} tokens")
        print(f"[ANALYZER API-{self.ai_type}]   - Total: {total_tokens:,} tokens (~{payload_size:,} bytes)")

        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=180  # Increased timeout for long transcripts
                )

                if response.status_code == 200:
                    result = response.json()
                    content = self._extract_content(result)
                    print(f"[ANALYZER API-{self.ai_type}] Response length: {len(content)} chars")
                    return content
                elif response.status_code >= 500:
                    # Server error - log full response and retry with backoff
                    wait_time = (2 ** attempt) * 5  # 5s, 10s, 20s
                    response_preview = response.text[:1000] if response.text else "No response body"
                    print(f"[ANALYZER API-{self.ai_type}] Server error {response.status_code}")
                    print(f"[ANALYZER API-{self.ai_type}] Response body: {response_preview}")
                    print(f"[ANALYZER API-{self.ai_type}] Retrying in {wait_time}s... (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                    last_error = f"API error: {response.status_code} - {response_preview[:200]}"
                else:
                    # Client error - don't retry
                    print(f"[ANALYZER API-{self.ai_type} ERROR] Status: {response.status_code}, Response: {response.text[:500]}")
                    raise Exception(f"API error: {response.status_code} - {response.text[:200]}")

            except requests.exceptions.Timeout:
                wait_time = (2 ** attempt) * 5
                print(f"[ANALYZER API-{self.ai_type}] Timeout, retrying in {wait_time}s... (attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
                last_error = "API timeout"
            except requests.exceptions.RequestException as e:
                wait_time = (2 ** attempt) * 5
                print(f"[ANALYZER API-{self.ai_type}] Request error: {e}, retrying in {wait_time}s... (attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
                last_error = str(e)

        raise Exception(f"API failed after {max_retries} attempts: {last_error}")

    def _extract_content(self, result: Dict[str, Any]) -> str:
        """Extract text content from API response based on AI type"""
        if self.ai_type == "B":
            # AI B: Gemini raw format
            # Response: {"candidates": [{"content": {"parts": [{"text": "..."}]}}]}
            candidates = result.get('candidates', [])
            if candidates:
                content = candidates[0].get('content', {})
                parts = content.get('parts', [])
                if parts:
                    return parts[0].get('text', '')
            return ""
        else:
            # AI A: Anthropic format
            # Response: {"content": [{"type": "text", "text": "..."}]}
            content_blocks = result.get('content', [])
            for block in content_blocks:
                if block.get('type') == 'text':
                    return block.get('text', '')
            return ""

    def _parse_response(self, response: str, video_duration: float) -> List[Dict[str, Any]]:
        """Parse AI response to extract clips with precise cut points"""
        # Remove markdown code blocks if present
        clean_response = response
        if "```json" in response:
            clean_response = re.sub(r'```json\s*', '', response)
            clean_response = re.sub(r'```\s*', '', clean_response)
        elif "```" in response:
            clean_response = re.sub(r'```\s*', '', response)

        # Try to find JSON array in the response
        json_match = re.search(r'\[[\s\S]*\]', clean_response)

        if not json_match:
            print(f"[ANALYZER PARSE] No JSON array found. Response preview: {response[:500]}")
            raise Exception("Could not find JSON array in AI response")

        try:
            clips_data = json.loads(json_match.group())
            print(f"[ANALYZER PARSE] Parsed {len(clips_data)} clips from JSON")
        except json.JSONDecodeError as e:
            print(f"[ANALYZER PARSE] JSON decode error: {e}. JSON preview: {json_match.group()[:300]}")
            raise Exception(f"Failed to parse JSON: {e}")

        # Get duration limits from settings
        from backend.core.state import app_state
        min_duration = app_state.settings.get('min_clip_duration', 15)
        max_duration = app_state.settings.get('max_clip_duration', 60)

        clips = []
        rejected_count = 0
        for clip in clips_data:
            start_seconds = self._parse_timestamp(clip.get('start', '00:00'))
            end_seconds = self._parse_timestamp(clip.get('end', '00:00'))

            # Validate timestamps
            if start_seconds >= end_seconds:
                continue
            if start_seconds < 0 or end_seconds > video_duration:
                # Clamp to video duration
                start_seconds = max(0, start_seconds)
                end_seconds = min(video_duration, end_seconds)

            # Adjust cut points to speech gaps if transcription data available
            if self._transcription_data:
                start_seconds, end_seconds = self._adjust_cut_points(
                    start_seconds, end_seconds
                )

            duration = end_seconds - start_seconds

            # Duration validation - soft limits, prioritize context
            # Only reject if extremely short (< 10s) or extremely long (> 180s)
            if duration < 10:
                print(f"[ANALYZER] Rejected clip '{clip.get('title', 'Untitled')}': {duration:.1f}s too short (< 10s)")
                rejected_count += 1
                continue
            if duration > 180:
                print(f"[ANALYZER] Rejected clip '{clip.get('title', 'Untitled')}': {duration:.1f}s too long (> 180s)")
                rejected_count += 1
                continue

            # Log if outside suggested range but still accept
            if duration < min_duration:
                print(f"[ANALYZER] Note: Clip '{clip.get('title', 'Untitled')}' is {duration:.1f}s (below suggested {min_duration}s, but accepting for context)")
            elif duration > max_duration:
                print(f"[ANALYZER] Note: Clip '{clip.get('title', 'Untitled')}' is {duration:.1f}s (above suggested {max_duration}s, but accepting for context)")

            clips.append({
                'start': start_seconds,
                'end': end_seconds,
                'start_formatted': self._format_duration(start_seconds),
                'end_formatted': self._format_duration(end_seconds),
                'duration': duration,
                'title': clip.get('title', 'Untitled Clip'),
                'reason': clip.get('reason', 'Interesting content')
            })

        if rejected_count > 0:
            print(f"[ANALYZER] Rejected {rejected_count} clips due to duration limits ({min_duration}-{max_duration}s)")

        return clips

    def _adjust_cut_points(
        self,
        start_time: float,
        end_time: float,
        start_search_range: float = 3.0,  # Increased range for better context
        end_search_range: float = 3.0
    ) -> tuple:
        """
        Adjust cut points to sentence boundaries for better context.

        Strategy:
        - START: Find a natural sentence start (after pause or punctuation)
        - END: Find a natural sentence end (before pause or punctuation)

        This ensures clips don't start/end in the middle of sentences.
        """
        if not self._transcription_data:
            return start_time, end_time

        words = self._get_word_timestamps()
        if not words:
            return start_time, end_time

        # Buffers
        START_BUFFER = 0.2  # 200ms before first word
        END_BUFFER = 0.3    # 300ms after last word

        # === FIND BEST START POINT ===
        # Look for a word that starts after a pause (>0.5s gap) or at sentence start
        best_start_word = None
        best_start_score = -999

        for i, word in enumerate(words):
            # Only consider words within search range of AI's chosen start
            if abs(word['start'] - start_time) > start_search_range:
                continue

            score = 0
            word_text = word.get('word', '').strip().lower()

            # Prefer words AFTER a pause (natural sentence boundary)
            if i > 0:
                gap = word['start'] - words[i-1]['end']
                if gap > 0.5:  # Significant pause
                    score += 50
                elif gap > 0.3:
                    score += 30

            # Prefer words that don't start with conjunctions
            bad_starts = ['dan', 'tapi', 'jadi', 'karena', 'terus', 'lalu', 'makanya', 'soalnya']
            if word_text in bad_starts:
                score -= 100  # Strong penalty

            # Prefer capitalized words or question words (sentence starters)
            good_starts = ['apa', 'siapa', 'kenapa', 'gimana', 'bagaimana', 'mengapa', 'kapan', 'dimana']
            if word_text in good_starts:
                score += 20

            # Prefer closer to AI's chosen time
            time_diff = abs(word['start'] - start_time)
            score -= time_diff * 10  # Penalty for being far from chosen time

            if score > best_start_score:
                best_start_score = score
                best_start_word = word

        # Use the best word's start time
        if best_start_word:
            adjusted_start = max(0, best_start_word['start'] - START_BUFFER)
        else:
            adjusted_start = max(0, start_time - START_BUFFER)

        # === FIND BEST END POINT ===
        # Look for a word that ends before a pause or at sentence end
        best_end_word = None
        best_end_score = -999

        for i, word in enumerate(words):
            # Only consider words within search range of AI's chosen end
            if abs(word['end'] - end_time) > end_search_range:
                continue

            score = 0
            word_text = word.get('word', '').strip()

            # Prefer words BEFORE a pause (natural sentence boundary)
            if i < len(words) - 1:
                gap = words[i+1]['start'] - word['end']
                if gap > 0.5:  # Significant pause
                    score += 50
                elif gap > 0.3:
                    score += 30

            # Prefer words that end sentences (punctuation in Whisper)
            if word_text.endswith(('.', '?', '!')):
                score += 40

            # Avoid ending on conjunctions or incomplete phrases
            bad_ends = ['dan', 'tapi', 'yang', 'ini', 'itu', 'ke', 'di', 'untuk']
            if word_text.lower().rstrip('.,?!') in bad_ends:
                score -= 50

            # Prefer closer to AI's chosen time
            time_diff = abs(word['end'] - end_time)
            score -= time_diff * 10

            if score > best_end_score:
                best_end_score = score
                best_end_word = word

        # Use the best word's end time
        if best_end_word:
            adjusted_end = best_end_word['end'] + END_BUFFER
        else:
            adjusted_end = end_time + END_BUFFER

        # Ensure valid duration
        if adjusted_end <= adjusted_start:
            return start_time, end_time + END_BUFFER

        return adjusted_start, adjusted_end

    def _find_speech_gaps(self, min_gap: float = 0.3) -> List[Dict]:
        """Find gaps/pauses in speech from stored transcription"""
        words = self._get_word_timestamps()
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
                    'midpoint': (gap_start + gap_end) / 2
                })

        return gaps

    def _get_word_timestamps(self) -> List[Dict]:
        """Extract word timestamps from stored transcription"""
        if not self._transcription_data:
            return []

        all_words = []
        for segment in self._transcription_data.get('segments', []):
            words = segment.get('words', [])
            for word in words:
                if 'start' in word and 'end' in word:
                    all_words.append({
                        'word': word.get('word', '').strip(),
                        'start': word.get('start', 0),
                        'end': word.get('end', 0)
                    })
        return all_words
    
    def _parse_timestamp(self, timestamp: str) -> float:
        """Parse MM:SS or HH:MM:SS to seconds"""
        parts = timestamp.strip().split(':')
        
        if len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)
        elif len(parts) == 3:
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        else:
            return 0
    
    def _format_duration(self, seconds: float) -> str:
        """Format seconds to MM:SS"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"


if __name__ == "__main__":
    # Test the analyzer
    analyzer = ClipAnalyzer()
    
    test_transcript = """
[00:00 - 00:15] Halo semuanya, hari ini kita akan bahas topik yang sangat kontroversial
[00:15 - 00:30] Jadi ternyata, fakta mengejutkan tentang ini adalah...
[00:30 - 00:45] Tidak banyak orang yang tahu bahwa sebenarnya...
[00:45 - 01:00] Dan ini adalah rahasia yang selama ini disembunyikan
[01:00 - 01:15] Tapi yang paling gila adalah kejadian selanjutnya
    """
    
    def on_progress(percent, status):
        print(f"{status} - {percent:.1f}%")
    
    # clips = analyzer.analyze(test_transcript, 75, progress_callback=on_progress)
    # for clip in clips:
    #     print(f"\n{clip['title']}")
    #     print(f"  {clip['start_formatted']} - {clip['end_formatted']}")
    #     print(f"  {clip['reason']}")
