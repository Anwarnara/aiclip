/**
 * Zustand Store for Application State
 * Dengan localStorage persistence untuk settings
 */

import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { VideoInfo, Clip, Settings, LogEntry, Progress, ProgressStatus, GPUStatus } from '@/types'

interface AppState {
  // Video state
  videoInfo: VideoInfo | null
  isProcessing: boolean
  currentStage: string

  // Progress
  progress: Progress
  progressStatus: ProgressStatus

  // Clips
  clips: Clip[]
  selectedClips: Set<number>

  // Settings
  settings: Settings

  // Logs
  logs: LogEntry[]

  // GPU Status
  gpuStatus: GPUStatus | null

  // UI state
  activeTab: 'youtube' | 'local'
  inputValue: string
  sidebarCollapsed: boolean

  // Actions
  setVideoInfo: (info: VideoInfo | null) => void
  setProcessing: (processing: boolean) => void
  setCurrentStage: (stage: string) => void
  updateProgress: (stage: keyof Progress, value: number, status: string) => void
  resetProgress: () => void
  setClips: (clips: Clip[]) => void
  toggleClip: (id: number) => void
  selectAllClips: () => void
  deselectAllClips: () => void
  setSettings: (settings: Partial<Settings>) => void
  addLog: (entry: LogEntry) => void
  clearLogs: () => void
  setGPUStatus: (status: GPUStatus) => void
  setActiveTab: (tab: 'youtube' | 'local') => void
  setInputValue: (value: string) => void
  setSidebarCollapsed: (collapsed: boolean) => void
  reset: () => void
}

const defaultProgress: Progress = {
  download: 0,
  transcribe: 0,
  analyze: 0,
  export: 0
}

const defaultProgressStatus: ProgressStatus = {
  download: 'Waiting...',
  transcribe: 'Waiting...',
  analyze: 'Waiting...',
  export: 'Waiting...'
}

const defaultSettings: Settings = {
  tracking_method: 'yolo', // yolo or dlib - used for both prescan and tracking
  yolo_model: 'yolov8n-face.pt', // Selected YOLO model file
  use_prescan: true,
  face_classifier: true,
  cinematic_mode: false,
  dynamic_tracking: true,
  dynamic_focus: false, // Auto-zoom to active speaker
  tracking_analyzer: true,
  auto_process: false,
  auto_clip_count: false,
  smoothing: 0.2,
  tracking_speed: 0.5,
  deadzone: 40,
  confidence: 0.5,
  single_zoom: 1.0,
  split_zoom: 1.0,
  split_screen: true,
  min_clip_duration: 15,
  max_clip_duration: 60,
  clips_to_find: 5,
  // Subtitle settings
  subtitle_enabled: true,
  subtitle_font_size: 48,
  subtitle_font_path: '',
  subtitle_max_words: 5,
  subtitle_position: 85, // 0-100 percentage from top (85 = bottom area)
  subtitle_style: 'uppercase',
  subtitle_color: '#FFFFFF',
  subtitle_highlight_color: '#FFFF00',
  subtitle_bg_enabled: true,
  subtitle_bg_color: '#000000',
  subtitle_bg_opacity: 0.5,
  // AI API settings
  ai_selected: 'A', // "A" = Anthropic API, "B" = Raw Response
  ai_auto_chunk: true, // Auto calculate optimal chunk size
  ai_chunk_tokens: 0, // Max tokens per chunk (0 = send all as one)
  ai_chunk_cooldown: 2, // Cooldown in seconds between chunk requests
  // Debug settings
  debug_mode: false, // Enable basic tracking logs in terminal (minimalist)
  debug_mode_advanced: false // Enable detailed/verbose tracking logs in terminal
}

// Custom serializer for Set
const customStorage = {
  getItem: (name: string) => {
    const str = localStorage.getItem(name)
    if (!str) return null
    const data = JSON.parse(str)
    // Convert selectedClips array back to Set
    if (data.state.selectedClips) {
      data.state.selectedClips = new Set(data.state.selectedClips)
    }
    // Merge with defaults to handle new settings
    if (data.state.settings) {
      data.state.settings = { ...defaultSettings, ...data.state.settings }
    }
    return data
  },
  setItem: (name: string, value: any) => {
    // Convert Set to array for serialization
    const toStore = {
      ...value,
      state: {
        ...value.state,
        selectedClips: value.state.selectedClips ? Array.from(value.state.selectedClips) : []
      }
    }
    localStorage.setItem(name, JSON.stringify(toStore))
  },
  removeItem: (name: string) => localStorage.removeItem(name)
}

export const useAppStore = create<AppState>()(
  persist(
    (set, _get) => ({
      // Initial state
      videoInfo: null,
      isProcessing: false,
      currentStage: 'idle',
      progress: { ...defaultProgress },
      progressStatus: { ...defaultProgressStatus },
      clips: [],
      selectedClips: new Set(),
      settings: { ...defaultSettings },
      logs: [],
      gpuStatus: null,
      activeTab: 'youtube',
      inputValue: '',
      sidebarCollapsed: false,

      // Actions
      setVideoInfo: (info) => set({ videoInfo: info }),

      setProcessing: (processing) => set({ isProcessing: processing }),

      setCurrentStage: (stage) => set({ currentStage: stage }),

      updateProgress: (stage, value, status) => set((state) => ({
        progress: { ...state.progress, [stage]: value },
        progressStatus: { ...state.progressStatus, [stage]: status }
      })),

      resetProgress: () => set({
        progress: { ...defaultProgress },
        progressStatus: { ...defaultProgressStatus }
      }),

      setClips: (clips) => {
        const selected = new Set(clips.map(c => c.id))
        set({ clips, selectedClips: selected })
      },

      toggleClip: (id) => set((state) => {
        const newSelected = new Set(state.selectedClips)
        if (newSelected.has(id)) {
          newSelected.delete(id)
        } else {
          newSelected.add(id)
        }
        return { selectedClips: newSelected }
      }),

      selectAllClips: () => set((state) => ({
        selectedClips: new Set(state.clips.map(c => c.id))
      })),

      deselectAllClips: () => set({ selectedClips: new Set() }),

      setSettings: (newSettings) => set((state) => ({
        settings: { ...state.settings, ...newSettings }
      })),

      addLog: (entry) => set((state) => {
        // Prevent duplicates (simple check based on message and recent timestamp)
        const lastLog = state.logs[state.logs.length - 1]
        if (lastLog && lastLog.message === entry.message) {
          // If message is identical and timestamp is close (within 1 sec), ignore
          return {}
        }

        const logs = [...state.logs, entry]
        // Keep only last 200 logs
        if (logs.length > 200) {
          logs.shift()
        }
        return { logs }
      }),

      clearLogs: () => set({ logs: [] }),

      setGPUStatus: (status) => set({ gpuStatus: status }),

      setActiveTab: (tab) => set({ activeTab: tab }),

      setInputValue: (value) => set({ inputValue: value }),

      setSidebarCollapsed: (collapsed) => set({ sidebarCollapsed: collapsed }),

      reset: () => set({
        videoInfo: null,
        isProcessing: false,
        currentStage: 'idle',
        progress: { ...defaultProgress },
        progressStatus: { ...defaultProgressStatus },
        clips: [],
        selectedClips: new Set(),
        inputValue: ''
      })
    }),
    {
      name: 'autoclip-storage',
      storage: customStorage,
      partialize: (state) => ({
        settings: state.settings,
        activeTab: state.activeTab,
        sidebarCollapsed: state.sidebarCollapsed
      })
    }
  )
)
