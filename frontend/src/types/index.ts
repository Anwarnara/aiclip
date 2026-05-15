/**
 * TypeScript Types for Auto Clip Maker
 */

export interface VideoInfo {
  title: string;
  duration: number;
  thumbnail?: string;
  uploader?: string;
  is_local: boolean;
}

export interface Clip {
  id: number;
  start: number;
  end: number;
  start_formatted: string;
  end_formatted: string;
  duration: number;
  title: string;
  reason: string;
}

export interface Progress {
  download: number;
  transcribe: number;
  analyze: number;
  export: number;
}

export interface ProgressStatus {
  download: string;
  transcribe: string;
  analyze: string;
  export: string;
}

export interface Settings {
  tracking_method: string;
  yolo_model: string;  // Selected YOLO model file
  use_prescan: boolean;
  face_classifier: boolean;
  cinematic_mode: boolean;
  dynamic_tracking: boolean;
  dynamic_focus: boolean;  // Auto-zoom to active speaker
  tracking_analyzer: boolean;
  auto_process: boolean;
  auto_clip_count: boolean;
  smoothing: number;
  tracking_speed: number;
  deadzone: number;
  confidence: number;
  single_zoom: number;
  split_zoom: number;
  split_screen: boolean;
  min_clip_duration: number;
  max_clip_duration: number;
  clips_to_find: number;
  // Subtitle settings
  subtitle_enabled: boolean;
  subtitle_font_size: number;
  subtitle_font_path: string;
  subtitle_max_words: number;
  subtitle_position: number; // 0-100 percentage from top
  subtitle_style: 'uppercase' | 'bold';
  subtitle_color: string;
  subtitle_highlight_color: string;
  subtitle_bg_enabled: boolean;
  subtitle_bg_color: string;
  subtitle_bg_opacity: number;
  // AI API settings
  ai_selected: string; // "A" = Anthropic API, "B" = Raw Response
  ai_auto_chunk: boolean; // Auto calculate optimal chunk size
  ai_chunk_tokens: number; // Max tokens per chunk (0 = send all as one)
  ai_chunk_cooldown: number; // Cooldown in seconds between chunk requests
  // Optical Flow Poster Filter
  optical_flow_enabled: boolean;
  optical_flow_threshold: number;  // Min flow difference (px)
  optical_flow_min_samples: number;  // Min frames before deciding
  optical_flow_consistency: number;  // Ratio for independent motion (0-1)
  optical_flow_dense: boolean;  // True = Farneback (accurate), False = Lucas-Kanade (fast)
  // Debug settings
  debug_mode: boolean; // Enable basic tracking logs in terminal (minimalist)
  debug_mode_advanced: boolean; // Enable detailed/verbose tracking logs in terminal
}

export interface LogEntry {
  timestamp: string;
  message: string;
  level: 'info' | 'warning' | 'error';
}

export interface GPUStatus {
  cuda_available: boolean;
  gpu_name?: string;
  gpu_memory?: string;
  device: string;
}

export interface ProcessingStatus {
  is_processing: boolean;
  current_stage: string;
  video_title?: string;
  video_duration: number;
  progress: Progress;
  progress_status: ProgressStatus;
  clips_count: number;
}

export interface WebSocketMessage {
  event: string;
  data: unknown;
}

export interface ExportProgress {
  current: number;
  total: number;
  clip_title: string;
  status: string;
  percent: number;
}
