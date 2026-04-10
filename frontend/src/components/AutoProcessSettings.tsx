/**
 * Auto Process Settings Panel
 * Configure automatic video processing settings
 */

import { useState, useEffect } from 'react'
import {
  Settings,
  X,
  Clock,
  Download,
  Filter,
  Globe,
  Sparkles,
  TrendingUp,
  Eye,
  Shuffle,
  ThumbsUp,
  Save,
  RotateCcw,
  Trash2,
  CheckCircle2,
  Loader2,
  Play,
  Search,
  ListPlus,
  Square,
  CheckSquare
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from '@/components/ui/dialog'

interface Category {
  id: string
  name: string
}

interface Region {
  code: string
  name: string
}

interface Video {
  id: string
  title: string
  channel: string
  thumbnail: string
  duration: string
  views: string
  url: string
  engagement_rate: number
}

interface Settings {
  enabled: boolean
  resolution: string
  check_interval_hours: number
  active_hours_start: number
  active_hours_end: number
  processing_priority: string
  genres: string[]
  duration_filter: string
  region_code: string
  max_videos_per_run: number
  max_downloads_per_scan: number
  search_query: string
  skip_processed: boolean
}

interface AutoProcessSettingsProps {
  isOpen: boolean
  onClose: () => void
}

export function AutoProcessSettings({ isOpen, onClose }: AutoProcessSettingsProps) {
  const [settings, setSettings] = useState<Settings>({
    enabled: false,
    resolution: '1080',
    check_interval_hours: 6,
    active_hours_start: 8,
    active_hours_end: 22,
    processing_priority: 'ai_recommendation',
    genres: ['0'],
    duration_filter: 'medium',
    region_code: 'ID',
    max_videos_per_run: 5,
    max_downloads_per_scan: 10,
    search_query: '',
    skip_processed: true
  })

  const [categories, setCategories] = useState<Category[]>([])
  const [regions, setRegions] = useState<Region[]>([])
  const [processedCount, setProcessedCount] = useState(0)
  const [showClearDialog, setShowClearDialog] = useState(false)

  const [isLoading, setIsLoading] = useState(true)
  const [isSaving, setIsSaving] = useState(false)
  const [saveSuccess, setSaveSuccess] = useState(false)

  // Manual scan state
  const [isScanning, setIsScanning] = useState(false)
  const [scanResults, setScanResults] = useState<Video[]>([])
  const [selectedVideos, setSelectedVideos] = useState<Set<string>>(new Set())
  const [showScanResults, setShowScanResults] = useState(false)
  const [isAddingToQueue, setIsAddingToQueue] = useState(false)
  const [scanError, setScanError] = useState<string | null>(null)
  const [aiRecommendations, setAiRecommendations] = useState<{index: number, reason: string, score: number}[]>([])
  const [isLoadingAI, setIsLoadingAI] = useState(false)
  const [aiSummary, setAiSummary] = useState('')

  // Load data on open
  useEffect(() => {
    if (isOpen) {
      loadAllData()
    }
  }, [isOpen])

  const loadAllData = async () => {
    setIsLoading(true)
    await Promise.all([
      loadSettings(),
      loadCategories(),
      loadRegions(),
      loadProcessedVideos()
    ])
    setIsLoading(false)
  }

  const loadSettings = async () => {
    try {
      const response = await fetch('/api/auto-process/settings')
      const data = await response.json()
      setSettings(data)
    } catch (err) {
      console.error('Error loading settings:', err)
    }
  }

  const loadCategories = async () => {
    try {
      const response = await fetch('/api/youtube/categories')
      const data = await response.json()
      setCategories(data.categories || [])
    } catch {
      setCategories([{ id: '0', name: 'All' }])
    }
  }

  const loadRegions = async () => {
    try {
      const response = await fetch('/api/youtube/regions')
      const data = await response.json()
      setRegions(data.regions || [])
    } catch {
      setRegions([{ code: 'ID', name: 'Indonesia' }])
    }
  }

  const loadProcessedVideos = async () => {
    try {
      const response = await fetch('/api/auto-process/processed')
      const data = await response.json()
      setProcessedCount(data.count || 0)
    } catch {
      setProcessedCount(0)
    }
  }

  const saveSettings = async () => {
    setIsSaving(true)
    setSaveSuccess(false)

    try {
      const response = await fetch('/api/auto-process/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings)
      })

      if (response.ok) {
        setSaveSuccess(true)
        setTimeout(() => setSaveSuccess(false), 2000)
      }
    } catch (err) {
      console.error('Error saving settings:', err)
    } finally {
      setIsSaving(false)
    }
  }

  const resetSettings = async () => {
    try {
      const response = await fetch('/api/auto-process/settings/reset', {
        method: 'POST'
      })
      const data = await response.json()
      setSettings(data)
    } catch (err) {
      console.error('Error resetting settings:', err)
    }
  }

  const clearProcessedVideos = async () => {
    try {
      await fetch('/api/auto-process/clear-processed', {
        method: 'POST'
      })
      setProcessedCount(0)
    } catch (err) {
      console.error('Error clearing processed videos:', err)
    }
  }

  const updateSetting = <K extends keyof Settings>(key: K, value: Settings[K]) => {
    setSettings(prev => ({ ...prev, [key]: value }))
  }

  // Manual scan function
  const handleManualScan = async () => {
    if (!settings.search_query.trim()) {
      alert('Please enter a search query first')
      return
    }

    setIsScanning(true)
    setScanResults([])
    setSelectedVideos(new Set())
    setScanError(null)
    setAiRecommendations([])
    setAiSummary('')
    setShowScanResults(true) // Show dialog immediately with loading state

    try {
      // Build search request using current settings
      const response = await fetch('/api/youtube/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: settings.search_query.trim(),
          max_results: settings.max_downloads_per_scan,
          duration_filter: settings.duration_filter,
          order: 'relevance',
          category_id: settings.genres.includes('0') ? null : settings.genres[0],
          region_code: settings.region_code
        })
      })

      const data = await response.json()

      if (data.error) {
        setScanError(data.error)
        return
      }

      const videos = data.videos || []
      setScanResults(videos)

      // Get AI recommendations if we have videos
      if (videos.length > 0) {
        getAIRecommendations(videos)
      }
    } catch (err) {
      console.error('Error scanning:', err)
      setScanError('Failed to scan. Check your connection and API key.')
    } finally {
      setIsScanning(false)
    }
  }

  // Get AI recommendations for scan results
  const getAIRecommendations = async (videos: Video[]) => {
    setIsLoadingAI(true)
    try {
      const response = await fetch('/api/youtube/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          videos: videos,
          purpose: 'viral clips'
        })
      })
      const data = await response.json()
      setAiRecommendations(data.recommendations || [])
      setAiSummary(data.summary || '')
    } catch (err) {
      console.error('Error getting AI recommendations:', err)
    } finally {
      setIsLoadingAI(false)
    }
  }

  // Get recommendation for specific video
  const getRecommendationForVideo = (videoId: string): {index: number, reason: string, score: number} | undefined => {
    const videoIndex = scanResults.findIndex(v => v.id === videoId)
    return aiRecommendations.find(r => r.index === videoIndex + 1)
  }

  // Toggle video selection
  const toggleVideoSelection = (videoId: string) => {
    setSelectedVideos(prev => {
      const newSet = new Set(prev)
      if (newSet.has(videoId)) {
        newSet.delete(videoId)
      } else {
        newSet.add(videoId)
      }
      return newSet
    })
  }

  // Select all videos
  const selectAllVideos = () => {
    if (selectedVideos.size === scanResults.length) {
      setSelectedVideos(new Set())
    } else {
      setSelectedVideos(new Set(scanResults.map(v => v.id)))
    }
  }

  // Add selected videos to queue
  const addSelectedToQueue = async () => {
    if (selectedVideos.size === 0) return

    setIsAddingToQueue(true)
    let added = 0

    try {
      for (const video of scanResults) {
        if (selectedVideos.has(video.id)) {
          await fetch('/api/queue/add', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              video_id: video.id,
              video_url: video.url,
              title: video.title,
              channel: video.channel,
              thumbnail: video.thumbnail,
              resolution: settings.resolution
            })
          })
          added++
        }
      }

      alert(`Added ${added} videos to queue!`)
      setShowScanResults(false)
      setScanResults([])
      setSelectedVideos(new Set())
    } catch (err) {
      console.error('Error adding to queue:', err)
    } finally {
      setIsAddingToQueue(false)
    }
  }

  // Generate hour options
  const hourOptions = Array.from({ length: 24 }, (_, i) => ({
    value: i.toString(),
    label: i.toString().padStart(2, '0') + ':00'
  }))

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 bg-background/80 backdrop-blur-sm">
      <div className="fixed inset-4 z-50 bg-card border rounded-lg shadow-lg flex flex-col max-w-2xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b">
          <div className="flex items-center gap-2">
            <Settings className="h-5 w-5 text-primary" />
            <h2 className="text-lg font-semibold">Auto Process Settings</h2>
            {settings.enabled && (
              <Badge variant="default" className="bg-green-500 text-xs">Active</Badge>
            )}
          </div>
          <Button variant="ghost" size="icon" onClick={onClose}>
            <X className="h-4 w-4" />
          </Button>
        </div>

        {isLoading ? (
          <div className="flex-1 flex items-center justify-center">
            <Loader2 className="h-8 w-8 animate-spin" />
          </div>
        ) : (
          <ScrollArea className="flex-1 p-4">
            <div className="space-y-6">
              {/* Enable Toggle */}
              <div className="flex items-center justify-between p-4 bg-muted/50 rounded-lg">
                <div className="flex items-center gap-3">
                  <Play className="h-5 w-5 text-green-500" />
                  <div>
                    <p className="font-medium">Enable Auto Process</p>
                    <p className="text-xs text-muted-foreground">
                      Automatically scrape and process videos
                    </p>
                  </div>
                </div>
                <Switch
                  checked={settings.enabled}
                  onCheckedChange={(checked) => updateSetting('enabled', checked)}
                />
              </div>

              <Separator />

              {/* Search Query */}
              <div className="space-y-2">
                <Label className="flex items-center gap-2">
                  <Filter className="h-4 w-4" />
                  Search Query
                </Label>
                <div className="flex gap-2">
                  <Input
                    placeholder="Enter search keywords..."
                    value={settings.search_query}
                    onChange={(e) => updateSetting('search_query', e.target.value)}
                    className="flex-1"
                  />
                  <Button
                    onClick={handleManualScan}
                    disabled={isScanning || !settings.search_query.trim()}
                    variant="outline"
                  >
                    {isScanning ? (
                      <Loader2 className="h-4 w-4 animate-spin mr-2" />
                    ) : (
                      <Search className="h-4 w-4 mr-2" />
                    )}
                    Scan Now
                  </Button>
                </div>
                <p className="text-xs text-muted-foreground">
                  Keywords to search for. Click "Scan Now" to manually search and select videos.
                </p>
              </div>

              {/* Download Resolution */}
              <div className="space-y-2">
                <Label className="flex items-center gap-2">
                  <Download className="h-4 w-4" />
                  Download Resolution
                </Label>
                <Select
                  value={settings.resolution}
                  onValueChange={(v) => updateSetting('resolution', v)}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="720">720p (HD)</SelectItem>
                    <SelectItem value="1080">1080p (Full HD)</SelectItem>
                    <SelectItem value="1440">1440p (2K)</SelectItem>
                    <SelectItem value="2160">2160p (4K)</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Schedule Settings */}
              <div className="space-y-3">
                <Label className="flex items-center gap-2">
                  <Clock className="h-4 w-4" />
                  Schedule
                </Label>

                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-1">
                    <p className="text-xs text-muted-foreground">Check every</p>
                    <Select
                      value={settings.check_interval_hours.toString()}
                      onValueChange={(v) => updateSetting('check_interval_hours', parseInt(v))}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="1">1 hour</SelectItem>
                        <SelectItem value="2">2 hours</SelectItem>
                        <SelectItem value="4">4 hours</SelectItem>
                        <SelectItem value="6">6 hours</SelectItem>
                        <SelectItem value="12">12 hours</SelectItem>
                        <SelectItem value="24">24 hours</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-1">
                    <p className="text-xs text-muted-foreground">Max videos per run</p>
                    <Select
                      value={settings.max_videos_per_run.toString()}
                      onValueChange={(v) => updateSetting('max_videos_per_run', parseInt(v))}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="1">1 video</SelectItem>
                        <SelectItem value="3">3 videos</SelectItem>
                        <SelectItem value="5">5 videos</SelectItem>
                        <SelectItem value="10">10 videos</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-1">
                    <p className="text-xs text-muted-foreground">Active from</p>
                    <Select
                      value={settings.active_hours_start.toString()}
                      onValueChange={(v) => updateSetting('active_hours_start', parseInt(v))}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {hourOptions.map(h => (
                          <SelectItem key={h.value} value={h.value}>{h.label}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-1">
                    <p className="text-xs text-muted-foreground">Active until</p>
                    <Select
                      value={settings.active_hours_end.toString()}
                      onValueChange={(v) => updateSetting('active_hours_end', parseInt(v))}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {hourOptions.map(h => (
                          <SelectItem key={h.value} value={h.value}>{h.label}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </div>

              <Separator />

              {/* Processing Priority */}
              <div className="space-y-2">
                <Label className="flex items-center gap-2">
                  <Sparkles className="h-4 w-4" />
                  Processing Priority
                </Label>
                <Select
                  value={settings.processing_priority}
                  onValueChange={(v) => updateSetting('processing_priority', v)}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="ai_recommendation">
                      <span className="flex items-center gap-2">
                        <Sparkles className="h-3 w-3 text-purple-500" />
                        AI Recommendation First
                      </span>
                    </SelectItem>
                    <SelectItem value="engagement">
                      <span className="flex items-center gap-2">
                        <ThumbsUp className="h-3 w-3 text-blue-500" />
                        Highest Engagement First
                      </span>
                    </SelectItem>
                    <SelectItem value="viral_potential">
                      <span className="flex items-center gap-2">
                        <TrendingUp className="h-3 w-3 text-orange-500" />
                        Viral Potential First
                      </span>
                    </SelectItem>
                    <SelectItem value="most_views">
                      <span className="flex items-center gap-2">
                        <Eye className="h-3 w-3 text-green-500" />
                        Most Views First
                      </span>
                    </SelectItem>
                    <SelectItem value="random">
                      <span className="flex items-center gap-2">
                        <Shuffle className="h-3 w-3 text-gray-500" />
                        Random
                      </span>
                    </SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Filters */}
              <div className="space-y-3">
                <Label>Filters</Label>

                {/* Genre Multi-Select */}
                <div className="space-y-2">
                  <p className="text-xs text-muted-foreground">Genres (select multiple)</p>
                  <div className="grid grid-cols-4 gap-2 p-3 border rounded-md max-h-[200px] overflow-y-auto">
                    {categories.map(cat => (
                      <label
                        key={cat.id}
                        className={`flex items-center gap-2 p-2 rounded cursor-pointer text-xs hover:bg-muted/50 ${
                          settings.genres.includes(cat.id) ? 'bg-primary/10 border border-primary/50' : 'bg-muted/30'
                        }`}
                      >
                        <input
                          type="checkbox"
                          checked={settings.genres.includes(cat.id)}
                          onChange={(e) => {
                            if (cat.id === '0') {
                              // "All" - clear other selections
                              updateSetting('genres', ['0'])
                            } else {
                              let newGenres = [...settings.genres]
                              // Remove "All" if selecting specific genre
                              newGenres = newGenres.filter(g => g !== '0')

                              if (e.target.checked) {
                                newGenres.push(cat.id)
                              } else {
                                newGenres = newGenres.filter(g => g !== cat.id)
                              }
                              // If empty, default to "All"
                              if (newGenres.length === 0) {
                                newGenres = ['0']
                              }
                              updateSetting('genres', newGenres)
                            }
                          }}
                          className="h-3 w-3"
                        />
                        <span className="truncate">{cat.name}</span>
                      </label>
                    ))}
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Selected: {settings.genres.includes('0') ? 'All' : settings.genres.length + ' genres'}
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-1">
                    <p className="text-xs text-muted-foreground">Duration</p>
                    <Select
                      value={settings.duration_filter}
                      onValueChange={(v) => updateSetting('duration_filter', v)}
                    >
                      <SelectTrigger className="h-8 text-xs">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="any">Any</SelectItem>
                        <SelectItem value="short">&lt; 4 min</SelectItem>
                        <SelectItem value="medium">4-20 min</SelectItem>
                        <SelectItem value="long">&gt; 20 min</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-1">
                    <p className="text-xs text-muted-foreground">Region</p>
                    <Select
                      value={settings.region_code}
                      onValueChange={(v) => updateSetting('region_code', v)}
                    >
                      <SelectTrigger className="h-8 text-xs">
                        <Globe className="h-3 w-3 mr-1" />
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {regions.map(r => (
                          <SelectItem key={r.code} value={r.code}>{r.name}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                {/* Max Downloads Per Scan */}
                <div className="space-y-1">
                  <p className="text-xs text-muted-foreground">Max downloads per scan</p>
                  <div className="flex items-center gap-2">
                    <Download className="h-4 w-4 text-muted-foreground" />
                    <Input
                      type="number"
                      min={1}
                      max={100}
                      value={settings.max_downloads_per_scan}
                      onChange={(e) => {
                        const val = parseInt(e.target.value) || 1
                        updateSetting('max_downloads_per_scan', Math.min(100, Math.max(1, val)))
                      }}
                      className="h-8 w-24 text-xs"
                    />
                    <span className="text-xs text-muted-foreground">videos</span>
                  </div>
                </div>
              </div>

              <Separator />

              {/* Skip Processed */}
              <div className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
                <div>
                  <p className="font-medium text-sm">Skip Processed Videos</p>
                  <p className="text-xs text-muted-foreground">
                    Don't process videos that have been processed before
                  </p>
                </div>
                <Switch
                  checked={settings.skip_processed}
                  onCheckedChange={(checked) => updateSetting('skip_processed', checked)}
                />
              </div>

              {/* Processed Videos Count */}
              <div className="p-4 bg-muted/30 rounded-lg">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium text-sm">Processed Videos</p>
                    <p className="text-xs text-muted-foreground">
                      {processedCount} videos have been processed
                    </p>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={processedCount === 0}
                    onClick={() => setShowClearDialog(true)}
                  >
                    <Trash2 className="h-3 w-3 mr-1" />
                    Clear History
                  </Button>
                </div>
              </div>

              {/* Clear Confirmation Dialog */}
              <Dialog open={showClearDialog} onOpenChange={setShowClearDialog}>
                <DialogContent>
                  <DialogHeader>
                    <DialogTitle>Clear Processed History?</DialogTitle>
                  </DialogHeader>
                  <p className="text-sm text-muted-foreground">
                    This will remove all {processedCount} videos from the processed list.
                    Videos will be eligible for processing again.
                  </p>
                  <DialogFooter>
                    <Button variant="outline" onClick={() => setShowClearDialog(false)}>
                      Cancel
                    </Button>
                    <Button
                      variant="destructive"
                      onClick={() => {
                        clearProcessedVideos()
                        setShowClearDialog(false)
                      }}
                    >
                      Clear All
                    </Button>
                  </DialogFooter>
                </DialogContent>
              </Dialog>
            </div>
          </ScrollArea>
        )}

        {/* Footer */}
        <div className="flex items-center justify-between p-4 border-t">
          <Button variant="outline" onClick={resetSettings}>
            <RotateCcw className="h-4 w-4 mr-2" />
            Reset to Default
          </Button>

          <div className="flex items-center gap-2">
            {saveSuccess && (
              <span className="text-green-500 text-sm flex items-center gap-1">
                <CheckCircle2 className="h-4 w-4" />
                Saved!
              </span>
            )}
            <Button onClick={saveSettings} disabled={isSaving}>
              {isSaving ? (
                <Loader2 className="h-4 w-4 animate-spin mr-2" />
              ) : (
                <Save className="h-4 w-4 mr-2" />
              )}
              Save Settings
            </Button>
          </div>
        </div>
      </div>

      {/* Scan Results Dialog */}
      <Dialog open={showScanResults} onOpenChange={setShowScanResults}>
        <DialogContent className="max-w-4xl max-h-[80vh] flex flex-col">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Search className="h-4 w-4" />
              Scan Results
              {!isScanning && <Badge variant="secondary">{scanResults.length} videos</Badge>}
              {isScanning && <Loader2 className="h-4 w-4 animate-spin" />}
            </DialogTitle>
          </DialogHeader>

          {/* Loading State */}
          {isScanning && (
            <div className="flex-1 flex flex-col items-center justify-center py-12">
              <Loader2 className="h-8 w-8 animate-spin mb-4" />
              <p className="text-muted-foreground">Scanning YouTube...</p>
              <p className="text-xs text-muted-foreground mt-1">Query: {settings.search_query}</p>
            </div>
          )}

          {/* Error State */}
          {!isScanning && scanError && (
            <div className="flex-1 flex flex-col items-center justify-center py-12 text-red-500">
              <p className="font-medium">{scanError}</p>
              <p className="text-xs mt-2">Check your YouTube API key in data/yt_api_key.txt</p>
            </div>
          )}

          {/* Empty State */}
          {!isScanning && !scanError && scanResults.length === 0 && (
            <div className="flex-1 flex flex-col items-center justify-center py-12 text-muted-foreground">
              <p>No videos found</p>
              <p className="text-xs mt-1">Try different search keywords</p>
            </div>
          )}

          {/* Results */}
          {!isScanning && !scanError && scanResults.length > 0 && (
            <>
              {/* AI Recommendations Banner */}
              <div className="p-3 bg-gradient-to-r from-purple-500/10 to-pink-500/10 rounded-lg border border-purple-500/20 mb-2">
                <div className="flex items-center gap-2 mb-1">
                  <Sparkles className="h-4 w-4 text-purple-500" />
                  <span className="font-medium text-sm">AI Recommendations</span>
                  {isLoadingAI && <Loader2 className="h-3 w-3 animate-spin text-purple-500" />}
                </div>
                {aiSummary && <p className="text-xs text-muted-foreground">{aiSummary}</p>}
                {aiRecommendations.length > 0 && (
                  <div className="flex flex-wrap gap-1 mt-2">
                    {aiRecommendations.map((rec) => (
                      <Badge
                        key={rec.index}
                        variant="secondary"
                        className="text-xs bg-purple-500/20 hover:bg-purple-500/30 cursor-pointer"
                        onClick={() => {
                          const video = scanResults[rec.index - 1]
                          if (video) toggleVideoSelection(video.id)
                        }}
                      >
                        <Sparkles className="h-3 w-3 mr-1 text-yellow-500" />
                        #{rec.index} Score: {rec.score}
                      </Badge>
                    ))}
                  </div>
                )}
              </div>

              {/* Select All / Add to Queue Bar */}
              <div className="flex items-center justify-between py-2 border-b">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={selectAllVideos}
                  className="text-xs"
                  disabled={scanResults.length === 0}
                >
                  {selectedVideos.size === scanResults.length && scanResults.length > 0 ? (
                    <>
                      <CheckSquare className="h-4 w-4 mr-1" />
                      Deselect All
                    </>
                  ) : (
                    <>
                      <Square className="h-4 w-4 mr-1" />
                      Select All ({scanResults.length})
                    </>
                  )}
                </Button>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-muted-foreground">
                    {selectedVideos.size} selected
                  </span>
                  <Button
                    onClick={addSelectedToQueue}
                    disabled={selectedVideos.size === 0 || isAddingToQueue}
                    size="sm"
                  >
                    {isAddingToQueue ? (
                      <Loader2 className="h-4 w-4 animate-spin mr-1" />
                    ) : (
                      <ListPlus className="h-4 w-4 mr-1" />
                    )}
                    Add to Queue
                  </Button>
                </div>
              </div>

              {/* Video List */}
              <div className="flex-1 overflow-y-auto">
                <div className="grid grid-cols-1 gap-2 p-2">
                  {scanResults.map((video) => {
                    const rec = getRecommendationForVideo(video.id)
                    return (
                      <div
                        key={video.id}
                        className={`flex items-center gap-3 p-2 rounded-lg border cursor-pointer transition-colors ${
                          selectedVideos.has(video.id)
                            ? 'bg-primary/10 border-primary'
                            : rec
                              ? 'bg-purple-500/5 border-purple-500/30 hover:bg-purple-500/10'
                              : 'hover:bg-muted/50'
                        }`}
                        onClick={() => toggleVideoSelection(video.id)}
                      >
                        {/* Checkbox */}
                        <div className="shrink-0">
                          {selectedVideos.has(video.id) ? (
                            <CheckSquare className="h-5 w-5 text-primary" />
                          ) : (
                            <Square className="h-5 w-5 text-muted-foreground" />
                          )}
                        </div>

                        {/* Thumbnail */}
                        <div className="w-24 h-14 rounded overflow-hidden shrink-0 bg-muted relative">
                          {video.thumbnail && (
                            <img
                              src={video.thumbnail}
                              alt={video.title}
                              className="w-full h-full object-cover"
                            />
                          )}
                          {rec && (
                            <Badge className="absolute top-0 left-0 bg-purple-500 text-white text-[10px] px-1">
                              AI #{rec.index}
                            </Badge>
                          )}
                        </div>

                        {/* Info */}
                        <div className="flex-1 min-w-0">
                          <h4 className="font-medium text-sm line-clamp-1">{video.title}</h4>
                          <p className="text-xs text-muted-foreground">{video.channel}</p>
                          <div className="flex items-center gap-2 mt-1 text-xs text-muted-foreground">
                            <span>{video.duration}</span>
                            <span>•</span>
                            <span>{video.views} views</span>
                            {video.engagement_rate > 0 && (
                              <>
                                <span>•</span>
                                <span className={video.engagement_rate > 5 ? 'text-green-500' : ''}>
                                  {video.engagement_rate.toFixed(1)}% eng
                                </span>
                              </>
                            )}
                            {rec && (
                              <>
                                <span>•</span>
                                <span className="text-purple-500">Score: {rec.score}</span>
                              </>
                            )}
                          </div>
                          {rec && (
                            <p className="text-xs text-purple-400 mt-1 line-clamp-1">{rec.reason}</p>
                          )}
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            </>
          )}

          <DialogFooter>
            <Button variant="outline" onClick={() => setShowScanResults(false)}>
              Close
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
