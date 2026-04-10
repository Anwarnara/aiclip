/**
 * SubtitlePanel Component
 * Configure subtitle/karaoke settings with visual preview and drag-to-position
 */

import { useEffect, useState, useRef, useCallback } from 'react'
import { Type, Palette, Monitor, Move } from 'lucide-react'
import { Card, CardHeader, CardTitle } from '@/components/ui/card'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { Slider } from '@/components/ui/slider'
import { Separator } from '@/components/ui/separator'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { useAppStore } from '@/stores/appStore'
import { useApi } from '@/hooks/useApi'
import { cn } from '@/lib/utils'

interface FontInfo {
  name: string
  path: string
  source: string
}

export function SubtitlePanel() {
  const { settings, setSettings, isProcessing } = useAppStore()
  const { saveSettings } = useApi()
  const [fonts, setFonts] = useState<FontInfo[]>([])
  const [isDragging, setIsDragging] = useState(false)
  const previewRef = useRef<HTMLDivElement>(null)

  // Load available fonts
  useEffect(() => {
    fetch('/api/settings/fonts')
      .then(res => res.json())
      .then(data => {
        if (data.fonts) {
          setFonts(data.fonts)
        }
      })
      .catch(() => {
        // Fallback fonts
        setFonts([
          { name: 'Arial Bold', path: 'C:/Windows/Fonts/arialbd.ttf', source: 'system' },
          { name: 'Impact', path: 'C:/Windows/Fonts/impact.ttf', source: 'system' }
        ])
      })
  }, [])

  const handleSettingChange = (key: string, value: unknown) => {
    const newSettings = { ...settings, [key]: value }
    setSettings(newSettings)
    saveSettings({ [key]: value })
  }

  // Calculate preview dimensions (9:16 aspect ratio)
  const previewWidth = 180
  const previewHeight = 320

  // Handle drag to position subtitle
  const handleDrag = useCallback((e: React.MouseEvent | MouseEvent) => {
    if (!previewRef.current || isProcessing) return

    const rect = previewRef.current.getBoundingClientRect()
    const y = e.clientY - rect.top

    // Calculate percentage (0-100)
    let percent = Math.round((y / rect.height) * 100)
    // Clamp between 10 and 90
    percent = Math.max(10, Math.min(90, percent))

    handleSettingChange('subtitle_position', percent)
  }, [isProcessing])

  const handleMouseDown = (e: React.MouseEvent) => {
    if (isProcessing) return
    setIsDragging(true)
    handleDrag(e)
  }

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (isDragging) {
      handleDrag(e)
    }
  }, [isDragging, handleDrag])

  const handleMouseUp = useCallback(() => {
    setIsDragging(false)
  }, [])

  // Add/remove mouse event listeners for dragging
  useEffect(() => {
    if (isDragging) {
      window.addEventListener('mousemove', handleMouseMove)
      window.addEventListener('mouseup', handleMouseUp)
    }
    return () => {
      window.removeEventListener('mousemove', handleMouseMove)
      window.removeEventListener('mouseup', handleMouseUp)
    }
  }, [isDragging, handleMouseMove, handleMouseUp])

  // Generate sample words based on max_words setting
  const maxWords = settings.subtitle_max_words ?? 5
  const sampleWords1 = ['apakah', 'saya', 'akan', 'makan', 'semangka'].slice(0, Math.min(maxWords, 5))
  const sampleWords2 = maxWords > 3 ? ['hari', 'ini'].slice(0, maxWords - 3) : []

  return (
    <Card className="flex flex-col overflow-hidden" style={{ maxHeight: '500px' }}>
      <CardHeader className="pb-3 flex-shrink-0">
        <CardTitle className="text-base flex items-center gap-2">
          <Type className="h-4 w-4" />
          Subtitle (Karaoke)
        </CardTitle>
      </CardHeader>
      <div className="flex-1 overflow-y-auto px-6 pb-4">
        <div className="space-y-4">
          {/* Enable Subtitle */}
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-xs">Enable Subtitle</Label>
              <p className="text-[10px] text-muted-foreground">TikTok/CapCut style captions</p>
            </div>
            <Switch
              checked={settings.subtitle_enabled ?? true}
              onCheckedChange={(v) => handleSettingChange('subtitle_enabled', v)}
              disabled={isProcessing}
            />
          </div>

          {settings.subtitle_enabled && (
            <>
              <Separator />

              {/* Visual Preview - 9:16 Frame with Drag Support */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
                    <Monitor className="h-3 w-3" />
                    Preview (9:16)
                  </div>
                  <div className="flex items-center gap-1 text-[10px] text-muted-foreground">
                    <Move className="h-3 w-3" />
                    Drag to position
                  </div>
                </div>
                <div className="flex justify-center">
                  <div
                    ref={previewRef}
                    className={cn(
                      "relative rounded-lg overflow-hidden border-2 select-none",
                      isDragging ? "border-primary cursor-grabbing" : "border-muted cursor-grab",
                      isProcessing && "opacity-50 cursor-not-allowed"
                    )}
                    style={{
                      width: previewWidth,
                      height: previewHeight,
                      background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)'
                    }}
                    onMouseDown={handleMouseDown}
                  >
                    {/* Simulated face area */}
                    <div className="absolute top-1/4 left-1/2 -translate-x-1/2 w-12 h-12 rounded-full bg-gray-600/30" />

                    {/* Subtitle container - positioned by percentage */}
                    <div
                      className="absolute left-2 right-2 flex flex-col items-center justify-center transition-all"
                      style={{
                        top: `${settings.subtitle_position ?? 85}%`,
                        transform: 'translateY(-50%)'
                      }}
                    >
                      {/* Position indicator line */}
                      <div
                        className={cn(
                          "absolute left-0 right-0 h-0.5 -top-3",
                          isDragging ? "bg-primary" : "bg-primary/30"
                        )}
                      />

                      <div
                        className="px-2 py-1 rounded text-center"
                        style={{
                          backgroundColor: settings.subtitle_bg_enabled
                            ? `${settings.subtitle_bg_color ?? '#000000'}${Math.round((settings.subtitle_bg_opacity ?? 0.5) * 255).toString(16).padStart(2, '0')}`
                            : 'transparent',
                          fontSize: `${Math.max(8, (settings.subtitle_font_size ?? 48) / 6)}px`,
                          lineHeight: 1.3
                        }}
                      >
                        {/* Line 1 - show words based on max_words setting */}
                        <div className="flex flex-wrap justify-center gap-0.5">
                          {sampleWords1.map((word, idx) => (
                            <span
                              key={idx}
                              style={{
                                color: idx === 1
                                  ? (settings.subtitle_highlight_color ?? '#FFFF00')
                                  : (settings.subtitle_color ?? '#FFFFFF'),
                                fontWeight: idx === 1 && settings.subtitle_style === 'bold' ? 'bold' : 'normal'
                              }}
                            >
                              {idx === 1 && settings.subtitle_style === 'uppercase'
                                ? word.toUpperCase()
                                : word}
                            </span>
                          ))}
                        </div>
                        {/* Line 2 - if words overflow */}
                        {sampleWords2.length > 0 && (
                          <div className="flex flex-wrap justify-center gap-0.5">
                            {sampleWords2.map((word, idx) => (
                              <span key={idx} style={{ color: settings.subtitle_color ?? '#FFFFFF' }}>
                                {word}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Info overlay */}
                    <div className="absolute top-1 left-1 text-[8px] text-white/50">
                      1080×1920
                    </div>
                    <div className="absolute bottom-1 right-1 text-[8px] text-white/50">
                      {settings.subtitle_position ?? 85}%
                    </div>
                  </div>
                </div>
              </div>

              <Separator />

              {/* Position Slider */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label className="text-xs">Vertical Position</Label>
                  <span className="text-xs font-medium">{settings.subtitle_position ?? 85}%</span>
                </div>
                <Slider
                  value={[settings.subtitle_position ?? 85]}
                  onValueChange={([v]) => handleSettingChange('subtitle_position', v)}
                  min={10}
                  max={90}
                  step={1}
                  disabled={isProcessing}
                />
                <p className="text-[10px] text-muted-foreground">
                  10% = Top, 50% = Center, 90% = Bottom
                </p>
              </div>

              {/* Font Selection */}
              <div className="space-y-2">
                <Label className="text-xs">Font</Label>
                <Select
                  value={settings.subtitle_font_path || 'auto'}
                  onValueChange={(v) => handleSettingChange('subtitle_font_path', v === 'auto' ? '' : v)}
                  disabled={isProcessing}
                >
                  <SelectTrigger className="h-8 text-xs">
                    <SelectValue placeholder="Auto (system font)" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="auto">Auto (system font)</SelectItem>
                    {fonts.map((font) => (
                      <SelectItem key={font.path} value={font.path}>
                        {font.source === 'custom' ? `★ ${font.name}` : font.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <p className="text-[10px] text-muted-foreground">
                  ★ = Custom fonts dari folder /font
                </p>
              </div>

              {/* Font Size */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label className="text-xs">Font Size</Label>
                  <span className="text-xs font-medium">{settings.subtitle_font_size ?? 48}px</span>
                </div>
                <Slider
                  value={[settings.subtitle_font_size ?? 48]}
                  onValueChange={([v]) => handleSettingChange('subtitle_font_size', v)}
                  min={24}
                  max={96}
                  step={4}
                  disabled={isProcessing}
                />
              </div>

              {/* Max Words Per Line */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label className="text-xs">Words Per Line</Label>
                  <span className="text-xs font-medium">{settings.subtitle_max_words ?? 5}</span>
                </div>
                <Slider
                  value={[settings.subtitle_max_words ?? 5]}
                  onValueChange={([v]) => handleSettingChange('subtitle_max_words', v)}
                  min={3}
                  max={8}
                  step={1}
                  disabled={isProcessing}
                />
                <p className="text-[10px] text-muted-foreground">
                  Maksimum kata per baris subtitle
                </p>
              </div>

              {/* Highlight Style */}
              <div className="space-y-2">
                <Label className="text-xs">Highlight Style</Label>
                <div className="flex gap-2">
                  <button
                    onClick={() => handleSettingChange('subtitle_style', 'uppercase')}
                    disabled={isProcessing}
                    className={cn(
                      "flex-1 px-3 py-1.5 text-xs rounded-md border transition-colors",
                      (settings.subtitle_style ?? 'uppercase') === 'uppercase'
                        ? "bg-primary text-primary-foreground border-primary"
                        : "bg-background hover:bg-muted border-input",
                      "disabled:opacity-50"
                    )}
                  >
                    UPPERCASE
                  </button>
                  <button
                    onClick={() => handleSettingChange('subtitle_style', 'bold')}
                    disabled={isProcessing}
                    className={cn(
                      "flex-1 px-3 py-1.5 text-xs rounded-md border transition-colors",
                      settings.subtitle_style === 'bold'
                        ? "bg-primary text-primary-foreground border-primary"
                        : "bg-background hover:bg-muted border-input",
                      "disabled:opacity-50"
                    )}
                  >
                    Bold Color
                  </button>
                </div>
              </div>

              <Separator />

              {/* Colors */}
              <div className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
                <Palette className="h-3 w-3" />
                Colors
              </div>

              <div className="grid grid-cols-2 gap-3">
                {/* Normal Color */}
                <div className="space-y-1">
                  <Label className="text-xs">Text</Label>
                  <div className="flex items-center gap-2">
                    <input
                      type="color"
                      value={settings.subtitle_color ?? '#FFFFFF'}
                      onChange={(e) => handleSettingChange('subtitle_color', e.target.value)}
                      disabled={isProcessing}
                      className="w-8 h-8 rounded cursor-pointer disabled:opacity-50"
                    />
                    <span className="text-[10px] text-muted-foreground">{settings.subtitle_color ?? '#FFFFFF'}</span>
                  </div>
                </div>

                {/* Highlight Color */}
                <div className="space-y-1">
                  <Label className="text-xs">Highlight</Label>
                  <div className="flex items-center gap-2">
                    <input
                      type="color"
                      value={settings.subtitle_highlight_color ?? '#FFFF00'}
                      onChange={(e) => handleSettingChange('subtitle_highlight_color', e.target.value)}
                      disabled={isProcessing}
                      className="w-8 h-8 rounded cursor-pointer disabled:opacity-50"
                    />
                    <span className="text-[10px] text-muted-foreground">{settings.subtitle_highlight_color ?? '#FFFF00'}</span>
                  </div>
                </div>
              </div>

              <Separator />

              {/* Background */}
              <div className="flex items-center justify-between">
                <div>
                  <Label className="text-xs">Text Background</Label>
                  <p className="text-[10px] text-muted-foreground">Semi-transparent box</p>
                </div>
                <Switch
                  checked={settings.subtitle_bg_enabled ?? true}
                  onCheckedChange={(v) => handleSettingChange('subtitle_bg_enabled', v)}
                  disabled={isProcessing}
                />
              </div>

              {settings.subtitle_bg_enabled && (
                <>
                  <div className="flex items-center gap-3">
                    <div className="space-y-1">
                      <Label className="text-xs">BG Color</Label>
                      <input
                        type="color"
                        value={settings.subtitle_bg_color ?? '#000000'}
                        onChange={(e) => handleSettingChange('subtitle_bg_color', e.target.value)}
                        disabled={isProcessing}
                        className="w-8 h-8 rounded cursor-pointer disabled:opacity-50"
                      />
                    </div>
                    <div className="flex-1 space-y-2">
                      <div className="flex items-center justify-between">
                        <Label className="text-xs">Opacity</Label>
                        <span className="text-xs font-medium">{((settings.subtitle_bg_opacity ?? 0.5) * 100).toFixed(0)}%</span>
                      </div>
                      <Slider
                        value={[settings.subtitle_bg_opacity ?? 0.5]}
                        onValueChange={([v]) => handleSettingChange('subtitle_bg_opacity', v)}
                        min={0}
                        max={1}
                        step={0.1}
                        disabled={isProcessing}
                      />
                    </div>
                  </div>
                </>
              )}
            </>
          )}
        </div>
      </div>
    </Card>
  )
}
