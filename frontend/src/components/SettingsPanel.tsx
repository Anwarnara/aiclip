/**
 * SettingsPanel Component
 * Configure processing settings
 */

import { useEffect, useState } from 'react'
import { Settings, Bot, Database, Trash2, AlertTriangle } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { Slider } from '@/components/ui/slider'
import { Separator } from '@/components/ui/separator'
import { Button } from '@/components/ui/button'
import { useAppStore } from '@/stores/appStore'
import { useApi } from '@/hooks/useApi'

interface YoloModel {
  name: string
  path: string
}

interface CacheInfo {
  has_cache: boolean
  video_title?: string
  video_filename?: string
  saved_at?: string
  segment_count?: number
  language?: string
}

export function SettingsPanel() {
  const { settings, setSettings, isProcessing } = useAppStore()
  const { loadSettings, saveSettings } = useApi()
  const [yoloModels, setYoloModels] = useState<YoloModel[]>([])
  const [cacheInfo, setCacheInfo] = useState<CacheInfo>({ has_cache: false })
  const [isLoadingCache, setIsLoadingCache] = useState(false)

  const loadCacheInfo = async () => {
    try {
      const res = await fetch('/api/video/cache')
      const data = await res.json()
      setCacheInfo(data)
    } catch (err) {
      console.error('Failed to load cache info:', err)
    }
  }

  const clearCache = async () => {
    if (!confirm('Hapus data transcription cache?')) return
    setIsLoadingCache(true)
    try {
      await fetch('/api/video/cache', { method: 'DELETE' })
      setCacheInfo({ has_cache: false })
    } catch (err) {
      console.error('Failed to clear cache:', err)
    } finally {
      setIsLoadingCache(false)
    }
  }

  useEffect(() => {
    loadSettings()
    loadCacheInfo()
    // Load available YOLO models
    fetch('/api/settings/yolo-models')
      .then(res => res.json())
      .then(data => {
        if (data.models) {
          setYoloModels(data.models)
        }
      })
      .catch(err => console.error('Failed to load YOLO models:', err))
  }, [loadSettings])

  const handleSettingChange = (key: string, value: unknown) => {
    const newSettings = { ...settings, [key]: value }
    setSettings(newSettings)
    saveSettings({ [key]: value })
  }

  return (
    <Card className="border-2 border-red-500">
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          <Settings className="h-4 w-4" />
          Settings (DEBUG MODE)
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Clip Duration */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label className="text-xs">Min Clip Duration</Label>
            <span className="text-xs text-muted-foreground">{settings.min_clip_duration}s</span>
          </div>
          <Slider
            value={[settings.min_clip_duration]}
            onValueChange={([v]) => handleSettingChange('min_clip_duration', v)}
            min={5}
            max={60}
            step={1}
            disabled={isProcessing}
          />
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label className="text-xs">Max Clip Duration</Label>
            <span className="text-xs text-muted-foreground">{settings.max_clip_duration}s</span>
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

        {/* Clips to Find with AUTO toggle */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label className="text-xs">Clips to Find</Label>
            <div className="flex items-center gap-2">
              <span className="text-xs text-muted-foreground">
                {settings.auto_clip_count ? 'AUTO' : settings.clips_to_find}
              </span>
              <Switch
                checked={settings.auto_clip_count ?? false}
                onCheckedChange={(v) => handleSettingChange('auto_clip_count', v)}
                disabled={isProcessing}
              />
              <span className="text-[10px] text-muted-foreground">AUTO</span>
            </div>
          </div>
          <Slider
            value={[settings.clips_to_find]}
            onValueChange={([v]) => handleSettingChange('clips_to_find', v)}
            min={1}
            max={100}
            step={1}
            disabled={isProcessing || settings.auto_clip_count}
            className={settings.auto_clip_count ? 'opacity-50' : ''}
          />
        </div>

        <Separator />

        {/* Tracking Method */}
        <div className="space-y-2">
          <Label className="text-xs">Tracking Method</Label>
          <div className="flex gap-2">
            <button
              onClick={() => handleSettingChange('tracking_method', 'dlib')}
              disabled={isProcessing}
              className={`flex-1 px-3 py-1.5 text-xs rounded-md border transition-colors ${
                settings.tracking_method === 'dlib'
                  ? 'bg-primary text-primary-foreground border-primary'
                  : 'bg-background hover:bg-muted border-input'
              } disabled:opacity-50`}
            >
              dlib (Accurate)
            </button>
            <button
              onClick={() => handleSettingChange('tracking_method', 'yolo')}
              disabled={isProcessing}
              className={`flex-1 px-3 py-1.5 text-xs rounded-md border transition-colors ${
                settings.tracking_method === 'yolo'
                  ? 'bg-primary text-primary-foreground border-primary'
                  : 'bg-background hover:bg-muted border-input'
              } disabled:opacity-50`}
            >
              YOLO (Fast)
            </button>
          </div>
        </div>

        {/* YOLO Model Selection - Always visible, disabled when not using YOLO */}
        <div className="space-y-2">
          <Label className="text-xs">YOLO Model</Label>
          <select
            value={settings.yolo_model || 'yolov8n-face.pt'}
            onChange={(e) => handleSettingChange('yolo_model', e.target.value)}
            disabled={isProcessing || settings.tracking_method !== 'yolo'}
            className={`w-full px-3 py-1.5 text-xs rounded-md border bg-background border-input ${
              settings.tracking_method === 'yolo' ? 'hover:bg-muted' : 'opacity-50'
            }`}
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
          {settings.tracking_method !== 'yolo' && (
            <p className="text-[10px] text-muted-foreground">
              Pilih YOLO tracking method untuk mengaktifkan
            </p>
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
              className={`px-3 py-2 text-xs rounded-md border transition-colors ${
                (settings.ai_selected ?? 'A') === 'A'
                  ? 'bg-primary text-primary-foreground border-primary'
                  : 'bg-background hover:bg-muted border-input'
              } disabled:opacity-50`}
            >
              <div className="font-medium">AI A</div>
              <div className="text-[10px] opacity-70">Gemini 3 Pro</div>
            </button>
            <button
              onClick={() => handleSettingChange('ai_selected', 'B')}
              disabled={isProcessing}
              className={`px-3 py-2 text-xs rounded-md border transition-colors ${
                settings.ai_selected === 'B'
                  ? 'bg-primary text-primary-foreground border-primary'
                  : 'bg-background hover:bg-muted border-input'
              } disabled:opacity-50`}
            >
              <div className="font-medium">AI B</div>
              <div className="text-[10px] opacity-70">Gemini 2.5 Flash</div>
            </button>
          </div>
        </div>

        {/* Pre-scan Tracking */}
        <div className="flex items-center justify-between">
          <div>
            <Label className="text-xs">Pre-scan Tracking</Label>
            <p className="text-[10px] text-muted-foreground">Scan video first (stable)</p>
          </div>
          <Switch
            checked={settings.use_prescan ?? true}
            onCheckedChange={(v) => handleSettingChange('use_prescan', v)}
            disabled={isProcessing}
          />
        </div>

        {/* Face Classifier - DISABLED, using motion detection */}
        {/*
        <div className="flex items-center justify-between opacity-50">
          <div>
            <Label className="text-xs">Face Classifier</Label>
            <p className="text-[10px] text-muted-foreground">Disabled - using motion detection</p>
          </div>
          <Switch
            checked={false}
            disabled={true}
          />
        </div>
        */}

        {/* Split Screen */}
        <div className="flex items-center justify-between">
          <div>
            <Label className="text-xs">Split Screen Mode</Label>
            <p className="text-[10px] text-muted-foreground">Show 2 faces in split view</p>
          </div>
          <Switch
            checked={settings.split_screen}
            onCheckedChange={(v) => handleSettingChange('split_screen', v)}
            disabled={isProcessing}
          />
        </div>

        {/* Auto Process */}
        <div className="flex items-center justify-between">
          <div>
            <Label className="text-xs">Auto Process</Label>
            <p className="text-[10px] text-muted-foreground">Auto export detected clips</p>
          </div>
          <Switch
            checked={settings.auto_process ?? false}
            onCheckedChange={(v) => handleSettingChange('auto_process', v)}
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

        {/* Tracking Analyzer - DISABLED for simpler pipeline */}
        {/*
        <div className="flex items-center justify-between opacity-50">
          <div>
            <Label className="text-xs">Tracking Analyzer</Label>
            <p className="text-[10px] text-muted-foreground">Disabled - using simpler pipeline</p>
          </div>
          <Switch
            checked={false}
            disabled={true}
          />
        </div>
        */}

        <Separator />

        {/* Face Detection Confidence */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label className="text-xs">Face Confidence</Label>
            <span className="text-xs text-muted-foreground">{(settings.confidence * 100).toFixed(0)}%</span>
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

        {/* Smoothing */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-xs">Smoothing</Label>
              <p className="text-[10px] text-muted-foreground">Higher = smoother movement</p>
            </div>
            <span className="text-xs text-muted-foreground">{(settings.smoothing * 100).toFixed(0)}%</span>
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

        {/* Tracking Speed */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-xs">Tracking Speed</Label>
              <p className="text-[10px] text-muted-foreground">Higher = faster follow</p>
            </div>
            <span className="text-xs text-muted-foreground">{((settings.tracking_speed ?? 0.5) * 100).toFixed(0)}%</span>
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

        {/* Zoom Settings */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label className="text-xs">Single Zoom</Label>
            <span className="text-xs text-muted-foreground">{settings.single_zoom.toFixed(1)}x</span>
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
            <span className="text-xs text-muted-foreground">{settings.split_zoom.toFixed(1)}x</span>
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

        <Separator />

        {/* Transcription Cache */}
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Database className="h-4 w-4" />
            <Label className="text-xs font-medium">Transcription Cache</Label>
          </div>

          {cacheInfo.has_cache ? (
            <div className="p-3 rounded-md bg-muted/50 border space-y-2">
              <div className="flex items-start justify-between gap-2">
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-medium truncate" title={cacheInfo.video_title}>
                    {cacheInfo.video_title}
                  </p>
                  <p className="text-[10px] text-muted-foreground">
                    {cacheInfo.segment_count} segments • {cacheInfo.language?.toUpperCase()}
                  </p>
                  {cacheInfo.saved_at && (
                    <p className="text-[10px] text-muted-foreground">
                      Saved: {new Date(cacheInfo.saved_at).toLocaleString()}
                    </p>
                  )}
                </div>
                <Button
                  variant="destructive"
                  size="sm"
                  onClick={clearCache}
                  disabled={isProcessing || isLoadingCache}
                  className="h-7 px-2"
                >
                  <Trash2 className="h-3 w-3 mr-1" />
                  Hapus
                </Button>
              </div>
              <div className="flex items-center gap-1 text-[10px] text-yellow-600 dark:text-yellow-500">
                <AlertTriangle className="h-3 w-3" />
                <span>Jika video berbeda, transcription akan di-replace</span>
              </div>
            </div>
          ) : (
            <div className="p-3 rounded-md bg-muted/30 border border-dashed">
              <p className="text-xs text-muted-foreground text-center">
                Tidak ada cache transcription
              </p>
              <p className="text-[10px] text-muted-foreground text-center mt-1">
                Cache akan dibuat saat video diproses
              </p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
