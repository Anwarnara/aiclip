/**
 * VideoInput Component
 * YouTube URL or Local file input with tabs
 * Supports drag & drop for local files
 */

import { useState, useCallback, useRef, useEffect } from 'react'
import { Youtube, FolderOpen, Play, Square, Upload, File as FileIcon, Download, ListPlus, Database, Trash2, RefreshCw } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { useAppStore } from '@/stores/appStore'
import { useApi } from '@/hooks/useApi'
import { cn } from '@/lib/utils'
import { api } from '@/lib/api'

interface CacheInfo {
  has_cache: boolean
  video_title?: string
  video_filename?: string
  saved_at?: string
  segment_count?: number
  language?: string
}

export function VideoInput() {
  const { activeTab, setActiveTab, inputValue, setInputValue, isProcessing, videoInfo, setProcessing, setCurrentStage, addLog, clips, selectedClips } = useAppStore()
  const { loading, error, setError, startProcessing, cancelProcessing, exportClips } = useApi()
  const [isDragging, setIsDragging] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [cacheInfo, setCacheInfo] = useState<CacheInfo>({ has_cache: false })
  const [loadingCache, setLoadingCache] = useState(false)

  const loadCacheInfo = async () => {
    try {
      const res = await fetch('/api/video/cache')
      if (res.ok) {
        const data = await res.json()
        setCacheInfo(data)
      }
    } catch (err) {
      console.error('Failed to load cache info:', err)
    }
  }

  const clearCache = async () => {
    if (!confirm('Hapus data transcription cache?')) return
    setLoadingCache(true)
    try {
      await fetch('/api/video/cache', { method: 'DELETE' })
      setCacheInfo({ has_cache: false })
      addLog({ timestamp: new Date().toLocaleTimeString(), message: 'Transcription cache cleared', level: 'info' })
    } catch (err) {
      console.error('Failed to clear cache:', err)
    } finally {
      setLoadingCache(false)
    }
  }

  const reanalyzeWithCache = async () => {
    if (!cacheInfo.has_cache) return
    setLoadingCache(true)
    try {
      const res = await fetch('/api/video/reanalyze', { method: 'POST' })
      if (res.ok) {
        setProcessing(true)
        setCurrentStage('analyzing')
        addLog({ timestamp: new Date().toLocaleTimeString(), message: `Re-analyzing with cached transcription: ${cacheInfo.video_title}`, level: 'info' })
      } else {
        const data = await res.json()
        setError(data.detail || 'Failed to reanalyze')
      }
    } catch (err) {
      console.error('Failed to reanalyze:', err)
    } finally {
      setLoadingCache(false)
    }
  }

  useEffect(() => {
    loadCacheInfo()
    // Reload cache info periodically
    const interval = setInterval(loadCacheInfo, 10000)
    return () => clearInterval(interval)
  }, [])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    // If already processing, add to queue instead
    if (isProcessing) {
      await addToQueue()
      return
    }

    // If we have a selected file, upload it
    if (activeTab === 'local' && selectedFile) {
      await handleFileUpload(selectedFile)
    } else {
      startProcessing()
    }
  }

  const addToQueue = async () => {
    try {
      // For local files, we need to upload first or use path
      const videoPath = activeTab === 'local'
        ? (selectedFile ? selectedFile.name : inputValue)
        : inputValue

      // Add to queue via API
      const response = await fetch('/api/queue/add', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          video_id: `local_${Date.now()}`,
          video_url: videoPath,
          title: selectedFile?.name || inputValue.split('/').pop() || 'Video',
          channel: 'Local',
          resolution: '1080'
        })
      })

      if (response.ok) {
        addLog({
          timestamp: new Date().toLocaleTimeString(),
          message: `Added to queue: ${selectedFile?.name || inputValue}`,
          level: 'info'
        })
        clearSelectedFile()
      }
    } catch (err) {
      console.error('Error adding to queue:', err)
    }
  }

  const handleExport = () => {
    const clipIds = Array.from(selectedClips)
    if (clipIds.length > 0) {
      exportClips(clipIds)
    }
  }

  const handleFileUpload = async (file: File) => {
    setUploading(true)
    setError(null)

    try {
      addLog({ timestamp: new Date().toLocaleTimeString(), message: `Uploading ${file.name}...`, level: 'info' })
      await api.uploadVideo(file)
      setProcessing(true)
      setCurrentStage('transcribing')
    } catch (e) {
      const message = e instanceof Error ? e.message : 'Failed to upload file'
      setError(message)
      addLog({ timestamp: new Date().toLocaleTimeString(), message, level: 'error' })
    } finally {
      setUploading(false)
    }
  }

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (!isProcessing && !uploading) {
      setIsDragging(true)
      // Auto-switch to local tab when dragging
      if (activeTab !== 'local') {
        setActiveTab('local')
      }
    }
  }, [isProcessing, uploading, activeTab, setActiveTab])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)

    if (isProcessing || uploading) return

    // Switch to local tab
    setActiveTab('local')

    // Get the file from the dropped files
    const files = e.dataTransfer.files
    if (files.length > 0) {
      const file = files[0]
      // Check if it's a video file
      if (file.type.startsWith('video/') || /\.(mp4|mkv|avi|mov|webm|flv|wmv)$/i.test(file.name)) {
        setSelectedFile(file)
        setInputValue(file.name)
      } else {
        setError('Please drop a video file (mp4, mkv, avi, mov, webm, flv, wmv)')
      }
    }
  }, [isProcessing, uploading, setActiveTab, setInputValue, setError])

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (files && files.length > 0) {
      const file = files[0]
      setSelectedFile(file)
      setInputValue(file.name)
    }
  }

  const handleBrowseClick = () => {
    fileInputRef.current?.click()
  }

  const clearSelectedFile = () => {
    setSelectedFile(null)
    setInputValue('')
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const hasSelection = selectedClips.size > 0

  return (
    <Card
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className={cn(
        "transition-all duration-200",
        isDragging && "ring-2 ring-primary ring-offset-2 ring-offset-background"
      )}
    >
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex justify-between items-center">
          <span>Video Input</span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs value={activeTab} onValueChange={(v) => { setActiveTab(v as 'youtube' | 'local'); clearSelectedFile(); }}>
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="youtube" className="gap-2">
              <Youtube className="h-4 w-4" />
              YouTube
            </TabsTrigger>
            <TabsTrigger value="local" className="gap-2">
              <FolderOpen className="h-4 w-4" />
              Local File
            </TabsTrigger>
          </TabsList>

          <form onSubmit={handleSubmit} className="mt-4">
            <TabsContent value="youtube" className="mt-0">
              <Input
                placeholder="Paste YouTube URL here..."
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                disabled={isProcessing}
              />
            </TabsContent>

            <TabsContent value="local" className="mt-0 space-y-3">
              {/* Hidden file input */}
              <input
                ref={fileInputRef}
                type="file"
                accept="video/*,.mp4,.mkv,.avi,.mov,.webm,.flv,.wmv"
                onChange={handleFileSelect}
                className="hidden"
              />

              {/* Drop zone */}
              <div
                className={cn(
                  "border-2 border-dashed rounded-lg p-6 text-center transition-all cursor-pointer",
                  isDragging
                    ? "border-primary bg-primary/10"
                    : "border-muted-foreground/25 hover:border-muted-foreground/50",
                  (isProcessing || uploading) && "opacity-50 cursor-not-allowed"
                )}
                onClick={() => {
                  if (!isProcessing && !uploading) {
                    handleBrowseClick()
                  }
                }}
              >
                {selectedFile ? (
                  <>
                    <FileIcon className="h-8 w-8 mx-auto mb-2 text-primary" />
                    <p className="text-sm font-medium text-foreground truncate">
                      {selectedFile.name}
                    </p>
                    <p className="text-xs text-muted-foreground mt-1">
                      {(selectedFile.size / 1024 / 1024).toFixed(1)} MB
                    </p>
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      className="mt-2"
                      onClick={(e) => {
                        e.stopPropagation()
                        clearSelectedFile()
                      }}
                    >
                      Change file
                    </Button>
                  </>
                ) : (
                  <>
                    <Upload className={cn(
                      "h-8 w-8 mx-auto mb-2",
                      isDragging ? "text-primary" : "text-muted-foreground"
                    )} />
                    <p className="text-sm text-muted-foreground">
                      {isDragging
                        ? "Drop video file here..."
                        : "Drag & drop video file here"}
                    </p>
                    <p className="text-xs text-muted-foreground/70 mt-1">
                      or click to browse
                    </p>
                  </>
                )}
              </div>

              {/* Manual path input */}
              <div className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground">or enter path:</span>
              </div>
              <Input
                id="local-file-input"
                placeholder="C:\Videos\video.mp4"
                value={selectedFile ? '' : inputValue}
                onChange={(e) => {
                  setSelectedFile(null)
                  setInputValue(e.target.value)
                }}
                disabled={isProcessing || uploading || !!selectedFile}
              />
            </TabsContent>

            {error && (
              <p className="mt-2 text-sm text-destructive">{error}</p>
            )}

            {videoInfo && (
              <div className="mt-3 p-2 bg-muted rounded-md">
                <p className="text-sm font-medium truncate">{videoInfo.title}</p>
                <p className="text-xs text-muted-foreground">
                  Duration: {Math.floor(videoInfo.duration / 60)}:{String(Math.floor(videoInfo.duration % 60)).padStart(2, '0')}
                </p>
              </div>
            )}

            {/* Transcription Cache Info */}
            {cacheInfo.has_cache && (
              <div className="mt-3 p-3 bg-blue-500/10 border border-blue-500/30 rounded-md">
                <div className="flex items-center gap-2 mb-2">
                  <Database className="h-4 w-4 text-blue-500" />
                  <span className="text-xs font-medium text-blue-500">Transcription Cache</span>
                </div>
                <p className="text-sm font-medium truncate" title={cacheInfo.video_title}>
                  {cacheInfo.video_title}
                </p>
                <p className="text-xs text-muted-foreground">
                  {cacheInfo.segment_count} segments • {cacheInfo.language?.toUpperCase()}
                </p>
                <div className="flex gap-2 mt-2">
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={reanalyzeWithCache}
                    disabled={isProcessing || loadingCache}
                    className="flex-1 h-7 text-xs gap-1"
                  >
                    <RefreshCw className="h-3 w-3" />
                    Re-analyze
                  </Button>
                  <Button
                    type="button"
                    variant="destructive"
                    size="sm"
                    onClick={clearCache}
                    disabled={isProcessing || loadingCache}
                    className="h-7 text-xs gap-1"
                  >
                    <Trash2 className="h-3 w-3" />
                    Hapus
                  </Button>
                </div>
              </div>
            )}

            <div className="mt-4 flex gap-2">
              {isProcessing ? (
                <>
                  {/* Add to Queue button when processing */}
                  <Button
                    type="submit"
                    variant="secondary"
                    className="flex-1 gap-2"
                    disabled={loading || uploading || (!inputValue.trim() && !selectedFile)}
                  >
                    <ListPlus className="h-4 w-4" />
                    Add to Queue
                  </Button>
                  <Button
                    type="button"
                    variant="destructive"
                    className="gap-2"
                    onClick={cancelProcessing}
                  >
                    <Square className="h-4 w-4" />
                    Cancel
                  </Button>
                </>
              ) : (
                <>
                  <Button
                    type="submit"
                    className="flex-1 gap-2"
                    disabled={loading || uploading || (!inputValue.trim() && !selectedFile)}
                  >
                    {uploading ? (
                      <>
                        <Upload className="h-4 w-4 animate-pulse" />
                        Uploading...
                      </>
                    ) : (
                      <>
                        <Play className="h-4 w-4" />
                        Start Processing
                      </>
                    )}
                  </Button>

                  {clips.length > 0 && (
                    <Button
                      type="button"
                      variant="secondary"
                      onClick={handleExport}
                      disabled={!hasSelection || loading}
                      className="flex-1 gap-2 bg-green-600 text-white hover:bg-green-700"
                    >
                      <Download className="h-4 w-4" />
                      Export Selected ({selectedClips.size})
                    </Button>
                  )}
                </>
              )}
            </div>
          </form>
        </Tabs>
      </CardContent>
    </Card>
  )
}
