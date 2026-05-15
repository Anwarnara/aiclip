/**
 * Sidebar Component
 * Left sidebar with settings and navigation
 */

import { useEffect, useState } from 'react'
import { Settings, ChevronLeft, ChevronRight, Sliders, Video, Info, Bug, Bot } from 'lucide-react'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { Slider } from '@/components/ui/slider'
import { Separator } from '@/components/ui/separator'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { useAppStore } from '@/stores/appStore'
import { useApi } from '@/hooks/useApi'
import { cn } from '@/lib/utils'

interface YoloModel {
  name: string
  path: string
}

export function Sidebar() {
  const { settings, setSettings, isProcessing, gpuStatus, sidebarCollapsed, setSidebarCollapsed } = useAppStore()
  const { saveSettings } = useApi()
  const [yoloModels, setYoloModels] = useState<YoloModel[]>([])

  // Load available YOLO models on mount
  useEffect(() => {
    fetch('/api/settings/yolo-models')
      .then(res => res.json())
      .then(data => {
        if (data.models) {
          setYoloModels(data.models)
        }
      })
      .catch(err => console.error('Failed to load YOLO models:', err))
  }, [])

  const handleSettingChange = (key: string, value: unknown) => {
    const newSettings = { ...settings, [key]: value }
    setSettings(newSettings)
    saveSettings({ [key]: value })
  }

  if (sidebarCollapsed) {
    return (
      <div className="w-12 bg-card border-r flex flex-col items-center py-4 gap-4">
        <Button
          variant="ghost"
          size="icon"
          onClick={() => setSidebarCollapsed(false)}
          className="h-8 w-8"
        >
          <ChevronRight className="h-4 w-4" />
        </Button>
        <Separator />
        <Settings className="h-4 w-4 text-muted-foreground" />
        <Sliders className="h-4 w-4 text-muted-foreground" />
        <Video className="h-4 w-4 text-muted-foreground" />
      </div>
    )
  }

  return (
    <div className="w-72 bg-card border-r flex flex-col">
      {/* Header */}
      <div className="p-3 border-b flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Settings className="h-4 w-4" />
          <span className="font-semibold text-sm">Settings</span>
        </div>
        <Button
          variant="ghost"
          size="icon"
          onClick={() => setSidebarCollapsed(true)}
          className="h-7 w-7"
        >
          <ChevronLeft className="h-4 w-4" />
        </Button>
      </div>

      <ScrollArea className="flex-1">
        <div className="p-4 space-y-5">
          {/* GPU Status */}
          {gpuStatus && (
            <>
              <div className="space-y-2">
                <div className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
                  <Info className="h-3 w-3" />
                  GPU Status
                </div>
                <div className="text-xs bg-muted/50 rounded p-2 space-y-1">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Device:</span>
                    <span className={cn(
                      "font-medium",
                      gpuStatus.cuda_available ? "text-green-500" : "text-yellow-500"
                    )}>
                      {gpuStatus.cuda_available ? "CUDA" : "CPU"}
                    </span>
                  </div>
                  {gpuStatus.gpu_name && (
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">GPU:</span>
                      <span className="font-medium truncate max-w-[120px]" title={gpuStatus.gpu_name}>
                        {gpuStatus.gpu_name.replace('NVIDIA GeForce ', '')}
                      </span>
                    </div>
                  )}
                  {gpuStatus.gpu_memory && (
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">VRAM:</span>
                      <span className="font-medium">{gpuStatus.gpu_memory}</span>
                    </div>
                  )}
                </div>
              </div>
              <Separator />
            </>
          )}

          {/* Clip Settings */}
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
              <Sliders className="h-3 w-3" />
              Clip Settings
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="text-xs">Min Duration</Label>
                <span className="text-xs font-medium">{settings.min_clip_duration}s</span>
              </div>
              <Slider
                value={[settings.min_clip_duration]}
                onValueChange={([v]) => handleSettingChange('min_clip_duration', v)}
                min={5}
                max={60}
                step={5}
                disabled={isProcessing}
              />
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="text-xs">Max Duration</Label>
                <span className="text-xs font-medium">{settings.max_clip_duration}s</span>
              </div>
              <Slider
                value={[settings.max_clip_duration]}
                onValueChange={([v]) => handleSettingChange('max_clip_duration', v)}
                min={15}
                max={600}
                step={5}
                disabled={isProcessing}
              />
            </div>

            {/* Clips to Find */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Label className="text-xs">Clips to Find</Label>
                  <div className="flex items-center gap-1">
                    <Label className="text-[10px] text-muted-foreground">Auto</Label>
                    <Switch
                      checked={settings.auto_clip_count ?? false}
                      onCheckedChange={(v) => handleSettingChange('auto_clip_count', v)}
                      className="h-4 w-7"
                      disabled={isProcessing}
                    />
                  </div>
                </div>
                <span className="text-xs font-medium">
                  {settings.auto_clip_count ? "Auto" : settings.clips_to_find}
                </span>
              </div>
              {!settings.auto_clip_count && (
                <Slider
                  value={[settings.clips_to_find]}
                  onValueChange={([v]) => handleSettingChange('clips_to_find', v)}
                  min={1}
                  max={100}
                  step={1}
                  disabled={isProcessing}
                />
              )}
            </div>

            {/* AI Chunk Settings */}
            <div className="space-y-2">
              <Label className="text-xs">AI Chunk Settings</Label>

              {/* Auto Chunk Toggle */}
              <div className="flex items-center justify-between">
                <Label className="text-[10px] text-muted-foreground">Auto Chunk (max 230k tokens)</Label>
                <Switch
                  checked={settings.ai_auto_chunk ?? true}
                  onCheckedChange={(v) => handleSettingChange('ai_auto_chunk', v)}
                  disabled={isProcessing}
                />
              </div>

              {/* Manual Settings - only show when auto is off */}
              {!(settings.ai_auto_chunk ?? true) && (
                <div className="flex gap-2">
                  <div className="flex-1">
                    <Label className="text-[10px] text-muted-foreground">Tokens</Label>
                    <input
                      type="number"
                      value={settings.ai_chunk_tokens ?? 0}
                      onChange={(e) => handleSettingChange('ai_chunk_tokens', parseInt(e.target.value) || 0)}
                      placeholder="0"
                      min={0}
                      disabled={isProcessing}
                      className="w-full h-8 px-2 text-xs rounded-md border bg-background disabled:opacity-50"
                    />
                  </div>
                  <div className="w-20">
                    <Label className="text-[10px] text-muted-foreground">Cooldown (s)</Label>
                    <input
                      type="number"
                      value={settings.ai_chunk_cooldown ?? 2}
                      onChange={(e) => handleSettingChange('ai_chunk_cooldown', parseInt(e.target.value) || 0)}
                      placeholder="2"
                      min={0}
                      disabled={isProcessing}
                      className="w-full h-8 px-2 text-xs rounded-md border bg-background disabled:opacity-50"
                    />
                  </div>
                </div>
              )}
              <p className="text-[10px] text-muted-foreground">
                {(settings.ai_auto_chunk ?? true)
                  ? "Auto: Sends max 230k tokens per request for best context"
                  : "Manual: 0 = send all as one. Cooldown: delay between chunks."
                }
              </p>
            </div>
          </div>

          <Separator />

          {/* Automation */}
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
              <Sliders className="h-3 w-3" />
              Automation
            </div>

            <div className="flex items-center justify-between">
              <div>
                <Label className="text-xs">Auto Process</Label>
                <p className="text-[10px] text-muted-foreground">Export found clips immediately</p>
              </div>
              <Switch
                checked={settings.auto_process ?? false}
                onCheckedChange={(v) => handleSettingChange('auto_process', v)}
                disabled={isProcessing}
              />
            </div>
          </div>

          <Separator />

          {/* Video Processing */}
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
              <Video className="h-3 w-3" />
              Video Processing
            </div>

            {/* Detection Method (used for both prescan and tracking) */}
            <div className="space-y-2">
              <Label className="text-xs">Detection Method</Label>
              <div className="flex gap-1">
                <button
                  onClick={() => handleSettingChange('tracking_method', 'yolo')}
                  disabled={isProcessing}
                  className={cn(
                    "flex-1 px-2 py-1 text-[10px] rounded border transition-colors",
                    (settings.tracking_method ?? 'yolo') === 'yolo'
                      ? "bg-primary text-primary-foreground border-primary"
                      : "bg-background hover:bg-muted border-input",
                    "disabled:opacity-50"
                  )}
                >
                  YOLO
                </button>
                <button
                  onClick={() => handleSettingChange('tracking_method', 'dlib')}
                  disabled={isProcessing}
                  className={cn(
                    "flex-1 px-2 py-1 text-[10px] rounded border transition-colors",
                    (settings.tracking_method ?? 'yolo') === 'dlib'
                      ? "bg-primary text-primary-foreground border-primary"
                      : "bg-background hover:bg-muted border-input",
                    "disabled:opacity-50"
                  )}
                >
                  dlib
                </button>
              </div>
              <p className="text-[10px] text-muted-foreground">YOLO: faster. dlib: embedding identity</p>
            </div>

            {/* YOLO Model Selection */}
            <div className="space-y-2">
              <Label className="text-xs">YOLO Model</Label>
              <select
                value={settings.yolo_model || 'yolov8n-face.pt'}
                onChange={(e) => handleSettingChange('yolo_model', e.target.value)}
                disabled={isProcessing || (settings.tracking_method ?? 'yolo') !== 'yolo'}
                className={cn(
                  "w-full px-2 py-1.5 text-xs rounded-md border bg-background",
                  (settings.tracking_method ?? 'yolo') === 'yolo'
                    ? "hover:bg-muted border-input"
                    : "opacity-50 border-input"
                )}
              >
                {yoloModels.length > 0 ? (
                  yoloModels.map((model) => (
                    <option key={model.name} value={model.name}>
                      {model.name}
                    </option>
                  ))
                ) : (
                  <option value="yolov8n-face.pt">yolov8n-face.pt</option>
                )}
              </select>
              {(settings.tracking_method ?? 'yolo') !== 'yolo' && (
                <p className="text-[10px] text-muted-foreground">Pilih YOLO untuk mengaktifkan</p>
              )}
            </div>

            {/* AI Model Selection */}
            <div className="space-y-2">
              <Label className="text-xs flex items-center gap-1">
                <Bot className="h-3 w-3" />
                AI Model
              </Label>
              <div className="grid grid-cols-2 gap-2">
                <button
                  onClick={() => handleSettingChange('ai_selected', 'A')}
                  disabled={isProcessing}
                  className={cn(
                    "px-2 py-2 text-xs rounded-md border transition-colors",
                    (settings.ai_selected ?? 'A') === 'A'
                      ? "bg-primary text-primary-foreground border-primary"
                      : "bg-background hover:bg-muted border-input",
                    "disabled:opacity-50"
                  )}
                >
                  <div className="font-medium">AI A</div>
                  <div className="text-[10px] opacity-70">Gemini 3 Pro</div>
                </button>
                <button
                  onClick={() => handleSettingChange('ai_selected', 'B')}
                  disabled={isProcessing}
                  className={cn(
                    "px-2 py-2 text-xs rounded-md border transition-colors",
                    settings.ai_selected === 'B'
                      ? "bg-primary text-primary-foreground border-primary"
                      : "bg-background hover:bg-muted border-input",
                    "disabled:opacity-50"
                  )}
                >
                  <div className="font-medium">AI B</div>
                  <div className="text-[10px] opacity-70">Gemini 2.5 Flash</div>
                </button>
              </div>
            </div>

            {/* Pre-scan Tracking */}
            <div className="flex items-center justify-between">
              <div>
                <Label className="text-xs">Pre-scan Mode</Label>
                <p className="text-[10px] text-muted-foreground">Scan first (stable)</p>
              </div>
              <Switch
                checked={settings.use_prescan ?? true}
                onCheckedChange={(v) => handleSettingChange('use_prescan', v)}
                disabled={isProcessing}
              />
            </div>

            {/* Face Classifier */}
            <div className="flex items-center justify-between">
              <div>
                <Label className="text-xs">Face Classifier</Label>
                <p className="text-[10px] text-muted-foreground">Classify humans</p>
              </div>
              <Switch
                checked={settings.face_classifier ?? true}
                onCheckedChange={(v) => handleSettingChange('face_classifier', v)}
                disabled={isProcessing}
              />
            </div>

            {/* Optical Flow Poster Filter */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <div>
                  <Label className="text-xs">Optical Flow Filter</Label>
                  <p className="text-[10px] text-muted-foreground">Detect posters by motion</p>
                </div>
                <Switch
                  checked={settings.optical_flow_enabled ?? true}
                  onCheckedChange={(v) => handleSettingChange('optical_flow_enabled', v)}
                  disabled={isProcessing}
                />
              </div>

              {/* Advanced settings - only show when enabled */}
              {(settings.optical_flow_enabled ?? true) && (
                <div className="pl-2 border-l-2 border-muted space-y-2">
                  <div className="space-y-1">
                    <div className="flex items-center justify-between">
                      <Label className="text-[10px] text-muted-foreground">Threshold</Label>
                      <span className="text-[10px] font-medium">{(settings.optical_flow_threshold ?? 2.0).toFixed(1)}px</span>
                    </div>
                    <Slider
                      value={[settings.optical_flow_threshold ?? 2.0]}
                      onValueChange={([v]) => handleSettingChange('optical_flow_threshold', v)}
                      min={0.5}
                      max={10}
                      step={0.5}
                      disabled={isProcessing}
                    />
                    <p className="text-[9px] text-muted-foreground">Lower = more sensitive (detects more posters)</p>
                  </div>

                  <div className="space-y-1">
                    <div className="flex items-center justify-between">
                      <Label className="text-[10px] text-muted-foreground">Min Samples</Label>
                      <span className="text-[10px] font-medium">{settings.optical_flow_min_samples ?? 5}</span>
                    </div>
                    <Slider
                      value={[settings.optical_flow_min_samples ?? 5]}
                      onValueChange={([v]) => handleSettingChange('optical_flow_min_samples', v)}
                      min={3}
                      max={20}
                      step={1}
                      disabled={isProcessing}
                    />
                    <p className="text-[9px] text-muted-foreground">Frames needed before deciding</p>
                  </div>

                  <div className="space-y-1">
                    <div className="flex items-center justify-between">
                      <Label className="text-[10px] text-muted-foreground">Consistency</Label>
                      <span className="text-[10px] font-medium">{((settings.optical_flow_consistency ?? 0.7) * 100).toFixed(0)}%</span>
                    </div>
                    <Slider
                      value={[settings.optical_flow_consistency ?? 0.7]}
                      onValueChange={([v]) => handleSettingChange('optical_flow_consistency', v)}
                      min={0.3}
                      max={1.0}
                      step={0.05}
                      disabled={isProcessing}
                    />
                    <p className="text-[9px] text-muted-foreground">Higher = stricter (fewer false positives)</p>
                  </div>

                  <div className="flex items-center justify-between">
                    <div>
                      <Label className="text-[10px] text-muted-foreground">Dense Flow</Label>
                      <p className="text-[9px] text-muted-foreground">Accurate but slower</p>
                    </div>
                    <Switch
                      checked={settings.optical_flow_dense ?? false}
                      onCheckedChange={(v) => handleSettingChange('optical_flow_dense', v)}
                      disabled={isProcessing}
                      className="h-4 w-7"
                    />
                  </div>
                </div>
              )}
            </div>

            {/* Cinematic Mode (AE Style) */}
            <div className="flex items-center justify-between">
              <div>
                <Label className="text-xs">Cinematic Motion</Label>
                <p className="text-[10px] text-muted-foreground">AE-style smooth tracking</p>
              </div>
              <Switch
                checked={settings.cinematic_mode ?? false}
                onCheckedChange={(v) => handleSettingChange('cinematic_mode', v)}
                disabled={isProcessing}
              />
            </div>

            {/* Dynamic Tracking */}
            <div className="flex items-center justify-between">
              <div>
                <Label className="text-xs">Dynamic Tracking</Label>
                <p className="text-[10px] text-muted-foreground">Auto-adjust to scene changes</p>
              </div>
              <Switch
                checked={settings.dynamic_tracking ?? true}
                onCheckedChange={(v) => handleSettingChange('dynamic_tracking', v)}
                disabled={isProcessing}
              />
            </div>

            {/* Dynamic Focus */}
            <div className="flex items-center justify-between">
              <div>
                <Label className="text-xs">Dynamic Focus</Label>
                <p className="text-[10px] text-muted-foreground">Auto-zoom to active speaker</p>
              </div>
              <Switch
                checked={settings.dynamic_focus ?? false}
                onCheckedChange={(v) => handleSettingChange('dynamic_focus', v)}
                disabled={isProcessing}
              />
            </div>

            {/* Tracking Analyzer */}
            <div className="flex items-center justify-between">
              <div>
                <Label className="text-xs">Tracking Analyzer</Label>
                <p className="text-[10px] text-muted-foreground">AI supervisor to fix glitches</p>
              </div>
              <Switch
                checked={settings.tracking_analyzer ?? true}
                onCheckedChange={(v) => handleSettingChange('tracking_analyzer', v)}
                disabled={isProcessing}
              />
            </div>

            <div className="flex items-center justify-between">
              <Label className="text-xs">Split Screen Mode</Label>
              <Switch
                checked={settings.split_screen}
                onCheckedChange={(v) => handleSettingChange('split_screen', v)}
                disabled={isProcessing}
              />
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="text-xs">Face Confidence</Label>
                <span className="text-xs font-medium">{(settings.confidence * 100).toFixed(0)}%</span>
              </div>
              <Slider
                value={[settings.confidence]}
                onValueChange={([v]) => handleSettingChange('confidence', v)}
                min={0.1}
                max={1}
                step={0.05}
                disabled={isProcessing}
              />
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="text-xs">Single Zoom</Label>
                <span className="text-xs font-medium">{settings.single_zoom.toFixed(1)}x</span>
              </div>
              <Slider
                value={[settings.single_zoom]}
                onValueChange={([v]) => handleSettingChange('single_zoom', v)}
                min={0.5}
                max={1.5}
                step={0.1}
                disabled={isProcessing}
              />
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="text-xs">Split Zoom</Label>
                <span className="text-xs font-medium">{settings.split_zoom.toFixed(1)}x</span>
              </div>
              <Slider
                value={[settings.split_zoom]}
                onValueChange={([v]) => handleSettingChange('split_zoom', v)}
                min={0.5}
                max={1.5}
                step={0.1}
                disabled={isProcessing}
              />
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="text-xs">Smoothing</Label>
                <span className="text-xs font-medium">{(settings.smoothing * 100).toFixed(0)}%</span>
              </div>
              <Slider
                value={[settings.smoothing]}
                onValueChange={([v]) => handleSettingChange('smoothing', v)}
                min={0}
                max={1}
                step={0.05}
                disabled={isProcessing}
              />
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="text-xs">Tracking Speed</Label>
                <span className="text-xs font-medium">{((settings.tracking_speed ?? 0.5) * 100).toFixed(0)}%</span>
              </div>
              <Slider
                value={[settings.tracking_speed ?? 0.5]}
                onValueChange={([v]) => handleSettingChange('tracking_speed', v)}
                min={0}
                max={1}
                step={0.05}
                disabled={isProcessing}
              />
            </div>
          </div>

          <Separator />

          {/* Debug Settings */}
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
              <Bug className="h-3 w-3" />
              Debug
            </div>

            <div className="flex items-center justify-between">
              <div>
                <Label className="text-xs">Debug Normal</Label>
                <p className="text-[10px] text-muted-foreground">Basic info per frame</p>
              </div>
              <Switch
                checked={settings.debug_mode ?? false}
                onCheckedChange={(v) => handleSettingChange('debug_mode', v)}
                disabled={isProcessing}
              />
            </div>

            <div className="flex items-center justify-between">
              <div>
                <Label className="text-xs">Debug Advanced</Label>
                <p className="text-[10px] text-muted-foreground">Detailed verbose logs</p>
              </div>
              <Switch
                checked={settings.debug_mode_advanced ?? false}
                onCheckedChange={(v) => handleSettingChange('debug_mode_advanced', v)}
                disabled={isProcessing}
              />
            </div>
          </div>
        </div>
      </ScrollArea>
    </div>
  )
}
