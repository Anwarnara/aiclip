"""
Tracking Analyzer Module - Self-Learning AI Supervisor
Autonomous agent that learns from experience and consults AI for novel problems.
Uses dual storage: .pt for fast tensor ops and .json for AI reporting.
Implements Context Tagging and Failure Analysis Loop.
"""

import numpy as np
import os
import json
import time
import torch
from typing import List, Dict, Any, Optional, Tuple
from collections import deque
from datetime import datetime

from backend.modules.analyzer import ClipAnalyzer  # Use existing LLM bridge
from backend.core.state import app_state
from backend.core.config import settings

# Paths for knowledge base
KB_TENSOR_FILE = os.path.join(settings.MODELS_DIR, "tracking_analyzer.pt")
KB_META_FILE = os.path.join(settings.MODELS_DIR, "tracking_analyzer.json")

class TrackingAnalyzer:
    """
    Intelligent Supervisor for Video Tracking.

    Capabilities:
    1. Monitor: Real-time stability analysis with Context Tagging.
    2. Recall: Check local knowledge base (.pt) for similar past cases using vector similarity.
    3. Consult: Ask AI (Gemini) if no local solution exists OR if a local solution failed.
    4. Learn: Save successful solutions to knowledge base (dual storage).
    5. Adapt: Retry with feedback if a solution fails (Failure Loop).
    """

    def __init__(self):
        self.history = deque(maxlen=300)  # ~10 seconds history @ 30fps
        self.ai_agent = ClipAnalyzer()

        # Knowledge Base in memory
        self.kb_states: torch.Tensor = torch.empty(0, 4) # [instability, face_var, avg_conf, mode]
        self.kb_actions: List[Dict] = []

        self._load_knowledge_base()

        # Active intervention state
        self.current_intervention = None
        self.last_consultation_time = 0
        self.cooldown = 5.0 # Reduced cooldown for faster reactions (was 10.0)

        # Log throttling
        self.last_log_times = {}

    def _log_throttled(self, key: str, message: str, level: str = "info", interval: float = 2.0):
        """Log a message only if enough time has passed since the last log for this key"""
        now = time.time()
        if now - self.last_log_times.get(key, 0) > interval:
            app_state.add_log(message, level)
            self.last_log_times[key] = now

    def _load_knowledge_base(self):
        """Load knowledge base from disk (pt + json)"""
        # Load Tensor Data
        if os.path.exists(KB_TENSOR_FILE):
            try:
                self.kb_states = torch.load(KB_TENSOR_FILE)
            except Exception as e:
                print(f"Error loading KB tensor: {e}")
                self.kb_states = torch.empty(0, 4)

        # Load Metadata
        if os.path.exists(KB_META_FILE):
            try:
                with open(KB_META_FILE, 'r') as f:
                    self.kb_actions = json.load(f)
            except Exception as e:
                print(f"Error loading KB metadata: {e}")
                self.kb_actions = []

        # Validate consistency
        if len(self.kb_states) != len(self.kb_actions):
            print("Warning: KB mismatch between tensor and metadata. Resetting.")
            self.kb_states = torch.empty(0, 4)
            self.kb_actions = []
        else:
            print(f"TrackingAnalyzer: Loaded {len(self.kb_actions)} learned patterns.")

    def _save_knowledge_base(self):
        """Save knowledge base to disk"""
        try:
            os.makedirs(os.path.dirname(KB_TENSOR_FILE), exist_ok=True)

            # Save Tensor
            torch.save(self.kb_states, KB_TENSOR_FILE)

            # Save Metadata
            with open(KB_META_FILE, 'w') as f:
                json.dump(self.kb_actions, f, indent=2)

        except Exception as e:
            print(f"Error saving knowledge base: {e}")

    def _learn_new_pattern(self, state: Dict, action: Dict, context_tags: List[str] = None):
        """Add new successful strategy to knowledge base"""
        # Convert state dict to tensor
        new_state_vector = torch.tensor([[
            state['instability'],
            state['face_variance'],
            state['avg_confidence'],
            state['current_mode']
        ]], dtype=torch.float32)

        # Update Memory
        self.kb_states = torch.cat((self.kb_states, new_state_vector), 0)

        new_meta = {
            "action": action,
            "timestamp": datetime.now().isoformat(),
            "confidence": 1.0,
            "context_tags": context_tags or [],
            # Store raw state in json too for debugging/reporting
            "state_snapshot": state
        }
        self.kb_actions.append(new_meta)

        self._save_knowledge_base()
        app_state.add_log("Tracking Analyzer: Learned new strategy (Saved to .pt/.json)!")

    def analyze_frame_stability(self, frame_data: Dict) -> Dict[str, Any]:
        """
        Real-time analysis. Returns corrective action if needed.
        IMPORTANT: When prescan is enabled, this should be PASSIVE - only monitor, not override.
        Only suggest actions when there are clear stability issues.
        """
        self.history.append(frame_data)

        # 1. Check if we are currently monitoring a fix
        if self.current_intervention:
            self._monitor_intervention_result(frame_data)
            return {"status": "MONITORING"}

        # 2. Analyze current state
        state_vector = self._extract_state_vector()
        instability_score = state_vector['instability']

        # Higher threshold for triggering actions (was 0.1)
        # Only intervene for significant instability
        if instability_score < 0.25:
            return {"status": "OK"}  # Stable enough

        # 3. Problem Detected - but check if prescan is handling this
        context_tags = self._generate_context_tags(state_vector)

        # Check if this is just normal mode switching (prescan doing its job)
        recent_30 = list(self.history)[-30:] if len(self.history) >= 30 else list(self.history)
        mode_switches = sum(1 for i in range(1, len(recent_30)) if recent_30[i].get('mode') != recent_30[i-1].get('mode'))

        # If we have smooth mode transitions (1-2 switches in 30 frames), prescan is working
        if mode_switches <= 2 and instability_score < 0.4:
            self._log_throttled("prescan_working", f"Tracking Analyzer: Prescan handling mode transition smoothly", "info")
            return {"status": "OK"}

        # Log instability
        if 0.25 <= instability_score < 0.45:
            self._log_throttled("moderate_instability", f"Tracking Analyzer: Moderate instability ({instability_score:.2f})", "warning")
        else:
            self._log_throttled("high_instability", f"Tracking Analyzer: High Instability ({instability_score:.2f}) - {', '.join(context_tags)}")

        # 4. Check Local Memory (Can I solve this myself?)
        solution = self._find_similar_solution(state_vector)

        if solution:
            action = solution['action']

            # Sanity Check: Don't force split if we mostly see 1 face
            if action['type'] == 'force_split':
                recent_faces = [f.get('faces_count', 0) for f in list(self.history)[-45:]]
                avg_faces = sum(recent_faces) / len(recent_faces) if recent_faces else 0
                if avg_faces < 1.85:  # Need nearly 2 faces consistently
                    self._log_throttled("block_split", f"Tracking Analyzer: Blocking 'force_split' (Avg faces: {avg_faces:.2f} < 1.85)", "warning")
                    return {"status": "OK"}

            # Don't force single if we see 2 faces
            if action['type'] == 'force_single':
                recent_faces = [f.get('faces_count', 0) for f in list(self.history)[-45:]]
                avg_faces = sum(recent_faces) / len(recent_faces) if recent_faces else 0
                if avg_faces >= 1.7:
                    self._log_throttled("block_single", f"Tracking Analyzer: Blocking 'force_single' (Avg faces: {avg_faces:.2f} >= 1.7)", "warning")
                    return {"status": "OK"}

            confidence = solution['confidence']
            app_state.add_log(f"Tracking Analyzer: Recalling local solution (Conf: {confidence:.2f}) -> {action['type']}")

            self._start_intervention(state_vector, action, source="LOCAL")
            return {
                "status": "ACTION",
                "issue": "instability",
                "suggestion": action['type'],
                "params": action.get('params', {})
            }

        # 5. Ask AI (I don't know what to do) - but with higher cooldown
        now = time.time()
        if now - self.last_consultation_time > self.cooldown:
            self.last_consultation_time = now
            app_state.add_log("Tracking Analyzer: Unknown pattern. Consulting AI...")

            ai_action = self._consult_ai_for_fix(list(self.history)[-60:], context_tags)

            # Sanity Check for AI - stricter validation
            if ai_action['type'] == 'force_split':
                recent_faces = [f.get('faces_count', 0) for f in list(self.history)[-45:]]
                avg_faces = sum(recent_faces) / len(recent_faces) if recent_faces else 0
                if avg_faces < 1.85:
                    app_state.add_log(f"Tracking Analyzer: Blocking AI 'force_split' (Avg faces: {avg_faces:.2f} < 1.85)")
                    ai_action = {"type": "increase_smoothing", "params": {}}

            if ai_action['type'] == 'force_single':
                recent_faces = [f.get('faces_count', 0) for f in list(self.history)[-45:]]
                avg_faces = sum(recent_faces) / len(recent_faces) if recent_faces else 0
                if avg_faces >= 1.7:
                    app_state.add_log(f"Tracking Analyzer: Blocking AI 'force_single' (Avg faces: {avg_faces:.2f} >= 1.7)")
                    ai_action = {"type": "increase_smoothing", "params": {}}

            self._start_intervention(state_vector, ai_action, source="AI")

            return {
                "status": "ACTION",
                "issue": "instability",
                "suggestion": ai_action['type'],
                "params": ai_action.get('params', {})
            }

        return {"status": "COOLDOWN"}

    def _generate_context_tags(self, state: Dict) -> List[str]:
        """Translate numerical state to human/AI-readable tags"""
        tags = []
        if state['instability'] > 0.6: tags.append("HIGH_INSTABILITY")
        elif state['instability'] > 0.3: tags.append("MODERATE_INSTABILITY")

        if state['face_variance'] > 0.5: tags.append("HIGH_FACE_FLICKER")

        if state['avg_confidence'] < 0.4: tags.append("LOW_CONFIDENCE")
        elif state['avg_confidence'] > 0.8: tags.append("HIGH_CONFIDENCE")

        if state['current_mode'] > 0.5: tags.append("SPLIT_MODE")
        else: tags.append("SINGLE_MODE")

        return tags

    def _extract_state_vector(self) -> Dict[str, float]:
        """
        Convert recent history into a numerical state vector for similarity matching.
        """
        # Reduced buffer requirement for faster reaction (was 30)
        history_window = 15
        if len(self.history) < history_window:
            return {"instability": 0.0, "face_variance": 0.0, "avg_confidence": 1.0, "current_mode": 0.0}

        recent = list(self.history)[-history_window:]

        # Calculate mode switches
        switches = 0
        current_mode = recent[0].get('mode')
        for f in recent[1:]:
            if f.get('mode') != current_mode:
                switches += 1
                current_mode = f.get('mode')

        # Face count variance
        counts = [f.get('faces_count', 0) for f in recent]
        face_variance = np.var(counts) if counts else 0

        # Micro-jitter detection (Variance of confidence)
        # High variance in confidence often means flickering faces
        confs = [f.get('avg_confidence', 0) for f in recent]
        avg_conf = np.mean(confs) if confs else 0
        conf_variance = np.var(confs) if confs else 0

        # Instability score calculation
        # Weighted combination: Switches + Face Count Variance + Confidence Jitter
        # Normalized roughly to 0-1
        switch_score = min(1.0, switches / 3.0) # 3 switches in 15 frames is huge
        jitter_score = min(1.0, conf_variance * 10.0)

        total_instability = (switch_score * 0.5) + (jitter_score * 0.3) + (min(1.0, face_variance) * 0.2)

        # State Vector
        return {
            "instability": float(total_instability),
            "face_variance": float(face_variance),
            "avg_confidence": float(avg_conf),
            "current_mode": 1.0 if recent[-1].get('mode') == 'split' else 0.0
        }

    def _find_similar_solution(self, current_state: Dict) -> Optional[Dict]:
        """Find best matching solution using Vector similarity (PyTorch)"""
        if len(self.kb_states) == 0:
            return None

        # Convert current state to tensor
        query = torch.tensor([[
            current_state['instability'],
            current_state['face_variance'],
            current_state['avg_confidence'],
            current_state['current_mode']
        ]], dtype=torch.float32)

        # Calculate Euclidean distances
        dists = torch.norm(self.kb_states - query, dim=1)

        # Find closest
        min_dist, idx = torch.min(dists, 0)

        best_dist = min_dist.item()
        best_idx = idx.item()

        threshold = 0.25 # Similarity threshold

        if best_dist < threshold:
            return {
                "action": self.kb_actions[best_idx]['action'],
                "confidence": 1.0 - (best_dist / threshold)
            }

        return None

    def _start_intervention(self, state: Dict, action: Dict, source: str):
        """Start monitoring a fix"""
        self.current_intervention = {
            "start_time": time.time(),
            "state_snapshot": state,
            "action": action,
            "source": source,
            "history_after": [],
            "context_tags": self._generate_context_tags(state)
        }

    def _monitor_intervention_result(self, frame_data: Dict):
        """Check if the applied fix is working"""
        if not self.current_intervention:
            return

        self.current_intervention['history_after'].append(frame_data)

        # Evaluate after 3 seconds (90 frames)
        if len(self.current_intervention['history_after']) >= 90:
            success = self._evaluate_success(self.current_intervention['history_after'])

            action_type = self.current_intervention['action']['type']
            source = self.current_intervention['source']
            tags = self.current_intervention['context_tags']

            if success:
                app_state.add_log(f"Tracking Analyzer: Fix '{action_type}' SUCCESS.")

                # LEARN: Save to knowledge base if it was from AI
                if source == "AI":
                    self._learn_new_pattern(
                        self.current_intervention['state_snapshot'],
                        self.current_intervention['action'],
                        tags
                    )
            else:
                app_state.add_log(f"Tracking Analyzer: Fix '{action_type}' FAILED.", "warning")

                # FAILURE ANALYSIS LOOP
                # If a LOCAL solution failed, we must ask AI why and get a new solution
                if source == "LOCAL":
                    app_state.add_log("Tracking Analyzer: Local memory failed. Escalating to AI for Failure Analysis...")

                    # Prepare rich failure context
                    failure_context = {
                        "failed_action": self.current_intervention['action'],
                        "original_tags": tags,
                        "outcome": "Still unstable after 3 seconds"
                    }

                    # Ask AI for a BETTER solution
                    ai_action = self._consult_ai_for_fix(
                        list(self.history)[-60:],
                        tags,
                        failure_context
                    )

                    # Note: We don't apply it immediately here to avoid loop complexity,
                    # but the main loop will catch the continued instability and likely pick up
                    # this new AI suggestion if we clear the cooldown.
                    self.last_consultation_time = 0 # Reset cooldown to allow immediate AI retry

            self.current_intervention = None # Reset

    def _evaluate_success(self, post_history: List[Dict]) -> bool:
        """Did the fix stabilize the video?"""
        switches = 0
        current_mode = post_history[0].get('mode')
        for f in post_history[1:]:
            if f.get('mode') != current_mode:
                switches += 1
                current_mode = f.get('mode')

        # Success if stable (few switches)
        return switches <= 1

    def _consult_ai_for_fix(self, recent_history: List[Dict], context_tags: List[str], failure_info: Dict = None) -> Dict[str, Any]:
        """
        Ask LLM (Gemini) what to do.
        """
        # Summarize data
        summary = {
            "context": context_tags,
            "avg_faces": np.mean([f.get('faces_count', 0) for f in recent_history]),
            "mode_switches": sum(1 for i in range(1, len(recent_history)) if recent_history[i]['mode'] != recent_history[i-1]['mode']),
            "avg_confidence": np.mean([f.get('avg_confidence', 0) for f in recent_history]),
            "duration_sec": len(recent_history) / 30.0
        }

        system_prompt = """
        You are an Autonomous Video Tracking Supervisor.
        The tracking system is unstable. Analyze the metrics and command a fix.

        Available Actions:
        1. "force_split" -> Use if 2 people likely present but detection flickers.
        2. "force_single" -> Use if 2nd face is ghost/noise.
        3. "increase_smoothing" -> Use if movement is jittery.
        4. "lower_threshold" -> Use if detection confidence is low.
        5. "request_feature" -> Use if code change is needed (e.g. "Add kalman filter").

        Return JSON ONLY: {"type": "force_split", "params": {}}
        """

        if failure_info:
            user_prompt = f"""
            CRITICAL: Previous attempt failed!
            Tried: {failure_info['failed_action']}
            Context: {failure_info['original_tags']}
            Outcome: {failure_info['outcome']}

            Current Metrics: {json.dumps(summary)}

            Please provide a DIFFERENT, more aggressive solution.
            If software limitations prevent fixing this, use "request_feature" with description in params.
            """
        else:
            user_prompt = f"Metrics: {json.dumps(summary)}"

        try:
            response = self.ai_agent._call_api(system_prompt, user_prompt)

            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                result = json.loads(json_match.group())

                # Special handling for feature requests
                if result.get("type") == "request_feature":
                    feature_desc = result.get("params", {}).get("description", "Unknown feature")
                    app_state.add_log(f"AI FEATURE REQUEST: {feature_desc}")
                    # Don't return this action to the processor as it can't execute it
                    return {"type": "increase_smoothing", "params": {}} # Fallback action

                return result

        except Exception as e:
            app_state.add_log(f"AI Error: {e}")

        return {"type": "increase_smoothing", "params": {}} # Safe fallback

    def analyze_segment_timeline(self, segments: List[Dict], fps: float = 30.0) -> List[Dict]:
        """
        Analyze and stabilize pre-scan timeline.
        This function RESPECTS prescan decisions and makes SMART merge decisions:

        Rules:
        1. Group consecutive short segments (<2.0 seconds) together
        2. When merging a group, pick the MODE that has more total duration (DOMINANT mode)
        3. Combine adjacent segments of the same mode
        4. Never force entire video to one mode - trust prescan's per-segment decisions
        5. Limit maximum segments to prevent excessive mode switching

        Args:
            segments: List of segments with 'start', 'end', 'mode'
            fps: Frames per second for duration calculation

        Returns:
            Stabilized segments list
        """
        if not segments:
            app_state.add_log("Tracking Analyzer: No segments to analyze")
            return []

        if len(segments) == 1:
            seg = segments[0]
            duration = (seg['end'] - seg['start']) / fps
            app_state.add_log(f"Tracking Analyzer: Single segment - {seg['mode'].upper()} ({duration:.1f}s)")
            return segments

        # Calculate statistics
        total_frames = segments[-1]['end']
        split_frames = sum(seg['end'] - seg['start'] for seg in segments if seg['mode'] == 'split')
        single_frames = total_frames - split_frames

        split_ratio = split_frames / total_frames if total_frames > 0 else 0
        single_ratio = single_frames / total_frames if total_frames > 0 else 0

        app_state.add_log(f"━━━ TRACKING ANALYZER REPORT ━━━")
        app_state.add_log(f"Input: {len(segments)} segments from prescan")
        app_state.add_log(f"Split: {split_frames/fps:.1f}s ({split_ratio*100:.1f}%)")
        app_state.add_log(f"Single: {single_frames/fps:.1f}s ({single_ratio*100:.1f}%)")

        # Step 1: Identify groups of consecutive short segments and merge them intelligently
        min_segment_frames = int(fps * 2.0)  # 2.0 seconds minimum
        stabilized = []

        i = 0
        while i < len(segments):
            seg = segments[i]
            seg_duration = seg['end'] - seg['start']

            # Check if this starts a group of short segments
            if seg_duration < min_segment_frames:
                # Collect all consecutive short segments
                group_start = seg['start']
                group_end = seg['end']
                split_duration_in_group = 0
                single_duration_in_group = 0

                j = i
                while j < len(segments):
                    s = segments[j]
                    s_dur = s['end'] - s['start']

                    if s_dur < min_segment_frames or j == i:
                        # Include this short segment in the group
                        group_end = s['end']
                        if s['mode'] == 'split':
                            split_duration_in_group += s_dur
                        else:
                            single_duration_in_group += s_dur
                        j += 1
                    else:
                        # Long segment encountered, stop grouping
                        break

                # Determine dominant mode in the group
                if split_duration_in_group > single_duration_in_group:
                    dominant_mode = 'split'
                else:
                    dominant_mode = 'single'

                group_duration = (group_end - group_start) / fps

                # If the group is still short after combining, merge with neighbor
                if (group_end - group_start) < min_segment_frames and stabilized:
                    # Merge with previous segment
                    prev_seg = stabilized[-1]
                    prev_seg['end'] = group_end
                    app_state.add_log(f"  Merged short group ({group_duration:.1f}s) -> previous {prev_seg['mode']}")
                else:
                    # Create a single segment for this group with dominant mode
                    if j - i > 1:  # Multiple segments were grouped
                        app_state.add_log(f"  Grouped {j-i} short segments ({group_duration:.1f}s) -> {dominant_mode.upper()}")

                    new_seg = {'start': group_start, 'end': group_end, 'mode': dominant_mode}

                    # Try to merge with previous if same mode
                    if stabilized and stabilized[-1]['mode'] == dominant_mode:
                        stabilized[-1]['end'] = group_end
                    else:
                        stabilized.append(new_seg)

                i = j  # Skip all processed segments in the group
            else:
                # Long segment - check if can merge with previous
                if stabilized and stabilized[-1]['mode'] == seg['mode']:
                    stabilized[-1]['end'] = seg['end']
                else:
                    stabilized.append(seg.copy())
                i += 1

        # Step 2: Final pass - merge any remaining adjacent same-mode segments
        merged = []
        for seg in stabilized:
            if merged and merged[-1]['mode'] == seg['mode']:
                merged[-1]['end'] = seg['end']
            else:
                merged.append(seg)

        # Step 3: If still too many segments (>6), iteratively merge shortest
        max_segments = 6
        while len(merged) > max_segments:
            # Find shortest segment (not first or last)
            shortest_idx = -1
            shortest_duration = float('inf')

            for i in range(1, len(merged) - 1):
                dur = merged[i]['end'] - merged[i]['start']
                if dur < shortest_duration:
                    shortest_duration = dur
                    shortest_idx = i

            if shortest_idx == -1:
                for i in [0, len(merged) - 1]:
                    dur = merged[i]['end'] - merged[i]['start']
                    if dur < shortest_duration:
                        shortest_duration = dur
                        shortest_idx = i

            if shortest_idx == -1:
                break

            # Merge with neighbor that has same mode, or pick based on duration
            if shortest_idx > 0:
                merged[shortest_idx - 1]['end'] = merged[shortest_idx]['end']
                app_state.add_log(f"  Consolidated segment {shortest_idx} ({shortest_duration/fps:.1f}s) -> prev")
                merged.pop(shortest_idx)
            elif shortest_idx < len(merged) - 1:
                merged[shortest_idx + 1]['start'] = merged[shortest_idx]['start']
                merged.pop(shortest_idx)

            # Re-merge same mode adjacents
            temp = []
            for seg in merged:
                if temp and temp[-1]['mode'] == seg['mode']:
                    temp[-1]['end'] = seg['end']
                else:
                    temp.append(seg)
            merged = temp

        # Log final result
        app_state.add_log(f"RESULT: {len(merged)} segments (from {len(segments)} input)")
        for i, seg in enumerate(merged):
            start_sec = seg['start'] / fps
            end_sec = seg['end'] / fps
            duration = end_sec - start_sec
            app_state.add_log(f"  [{i+1}] {seg['mode'].upper()}: {start_sec:.1f}s - {end_sec:.1f}s ({duration:.1f}s)")
        app_state.add_log(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        return merged
