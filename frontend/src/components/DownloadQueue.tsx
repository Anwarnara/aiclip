/**
 * Download Queue Panel
 * Shows download and processing queue status
 */

import { useState, useEffect, useCallback } from 'react'
import {
  Download,
  X,
  Loader2,
  CheckCircle2,
  AlertCircle,
  Trash2,
  RefreshCw,
  Clock,
  Film,
  RotateCcw,
  Pause,
  Play
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Progress } from '@/components/ui/progress'

interface QueueItem {
  id: string
  video_id: string
  video_url: string
  title: string
  channel: string
  thumbnail: string
  resolution: string
  status: string
  created_at: string
  download_progress: number
  process_progress: number
  downloaded_path: string | null
  output_path: string | null
  error: string | null
  clips_count: number
}

interface QueueStats {
  pending: number
  downloading: number
  downloaded: number
  processing: number
  completed: number
  failed: number
  total: number
  paused?: number
}

interface DownloadQueueProps {
  isOpen: boolean
  onClose: () => void
}

export function DownloadQueue({ isOpen, onClose }: DownloadQueueProps) {
  const [items, setItems] = useState<QueueItem[]>([])
  const [stats, setStats] = useState<QueueStats>({
    pending: 0,
    downloading: 0,
    downloaded: 0,
    processing: 0,
    completed: 0,
    failed: 0,
    total: 0,
    paused: 0
  })
  const [isLoading, setIsLoading] = useState(true)
  const [isPaused, setIsPaused] = useState(false)

  const loadQueue = useCallback(async () => {
    try {
      const [queueRes, pausedRes] = await Promise.all([
        fetch('/api/queue'),
        fetch('/api/queue/paused')
      ])
      const data = await queueRes.json()
      const pausedData = await pausedRes.json()
      setItems(data.items || [])
      setStats(data.stats || stats)
      setIsPaused(pausedData.paused || false)
    } catch (err) {
      console.error('Error loading queue:', err)
    } finally {
      setIsLoading(false)
    }
  }, [])

  // Load queue and poll for updates
  useEffect(() => {
    if (isOpen) {
      loadQueue()
      // Poll every 5 seconds when panel is open
      const interval = setInterval(loadQueue, 5000)
      return () => clearInterval(interval)
    }
  }, [isOpen, loadQueue])

  const removeItem = async (itemId: string, force: boolean = false) => {
    try {
      await fetch(`/api/queue/item/${itemId}?force=${force}`, { method: 'DELETE' })
      loadQueue()
    } catch (err) {
      console.error('Error removing item:', err)
    }
  }

  const retryItem = async (itemId: string) => {
    try {
      await fetch(`/api/queue/retry/${itemId}`, { method: 'POST' })
      loadQueue()
    } catch (err) {
      console.error('Error retrying item:', err)
    }
  }

  const clearCompleted = async () => {
    try {
      await fetch('/api/queue/clear-completed', { method: 'POST' })
      loadQueue()
    } catch (err) {
      console.error('Error clearing completed:', err)
    }
  }

  const togglePause = async () => {
    try {
      const endpoint = isPaused ? '/api/queue/resume' : '/api/queue/pause'
      await fetch(endpoint, { method: 'POST' })
      setIsPaused(!isPaused)
      loadQueue()
    } catch (err) {
      console.error('Error toggling pause:', err)
    }
  }

  const pauseItem = async (itemId: string) => {
    try {
      await fetch(`/api/queue/pause/${itemId}`, { method: 'POST' })
      loadQueue()
    } catch (err) {
      console.error('Error pausing item:', err)
    }
  }

  const resumeItem = async (itemId: string) => {
    try {
      await fetch(`/api/queue/resume/${itemId}`, { method: 'POST' })
      loadQueue()
    } catch (err) {
      console.error('Error resuming item:', err)
    }
  }

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'pending':
        return <Badge variant="secondary" className="text-xs"><Clock className="h-3 w-3 mr-1" />Pending</Badge>
      case 'downloading':
        return <Badge className="bg-blue-500 text-xs"><Download className="h-3 w-3 mr-1 animate-bounce" />Downloading</Badge>
      case 'downloaded':
        return <Badge className="bg-yellow-500 text-xs"><Clock className="h-3 w-3 mr-1" />Waiting</Badge>
      case 'processing':
        return <Badge className="bg-purple-500 text-xs"><Film className="h-3 w-3 mr-1 animate-spin" />Processing</Badge>
      case 'completed':
        return <Badge className="bg-green-500 text-xs"><CheckCircle2 className="h-3 w-3 mr-1" />Completed</Badge>
      case 'failed':
        return <Badge variant="destructive" className="text-xs"><AlertCircle className="h-3 w-3 mr-1" />Failed</Badge>
      case 'paused':
        return <Badge variant="outline" className="text-xs"><Pause className="h-3 w-3 mr-1" />Paused</Badge>
      default:
        return <Badge variant="secondary" className="text-xs">{status}</Badge>
    }
  }

  const getProgress = (item: QueueItem) => {
    if (item.status === 'downloading') {
      return item.download_progress
    } else if (item.status === 'processing') {
      return item.process_progress
    } else if (item.status === 'completed') {
      return 100
    }
    return 0
  }

  if (!isOpen) return null

  const activeCount = stats.downloading + stats.processing + stats.pending + stats.downloaded

  return (
    <div className="fixed inset-0 z-50 bg-background/80 backdrop-blur-sm">
      <div className="fixed inset-4 z-50 bg-card border rounded-lg shadow-lg flex flex-col max-w-2xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b">
          <div className="flex items-center gap-2">
            <Download className="h-5 w-5 text-primary" />
            <h2 className="text-lg font-semibold">Download Queue</h2>
            {activeCount > 0 && (
              <Badge variant="default" className="text-xs">{activeCount} active</Badge>
            )}
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant={isPaused ? "default" : "outline"}
              size="sm"
              onClick={togglePause}
              title={isPaused ? "Resume Queue" : "Pause Queue"}
              className="h-8"
            >
              {isPaused ? (
                <>
                  <Play className="h-4 w-4 mr-1" />
                  Resume
                </>
              ) : (
                <>
                  <Pause className="h-4 w-4 mr-1" />
                  Pause
                </>
              )}
            </Button>
            <Button variant="ghost" size="icon" onClick={loadQueue} title="Refresh">
              <RefreshCw className="h-4 w-4" />
            </Button>
            <Button variant="ghost" size="icon" onClick={onClose}>
              <X className="h-4 w-4" />
            </Button>
          </div>
        </div>

        {/* Stats Bar */}
        <div className="flex items-center gap-4 p-3 border-b bg-muted/30 text-xs">
          <span className="flex items-center gap-1">
            <Clock className="h-3 w-3" /> {stats.pending} Pending
          </span>
          <span className="flex items-center gap-1 text-blue-500">
            <Download className="h-3 w-3" /> {stats.downloading} Downloading
          </span>
          <span className="flex items-center gap-1 text-purple-500">
            <Film className="h-3 w-3" /> {stats.processing} Processing
          </span>
          <span className="flex items-center gap-1 text-green-500">
            <CheckCircle2 className="h-3 w-3" /> {stats.completed} Completed
          </span>
          {stats.failed > 0 && (
            <span className="flex items-center gap-1 text-red-500">
              <AlertCircle className="h-3 w-3" /> {stats.failed} Failed
            </span>
          )}
        </div>

        {/* Queue Items */}
        <ScrollArea className="flex-1">
          {isLoading ? (
            <div className="flex items-center justify-center py-16">
              <Loader2 className="h-8 w-8 animate-spin" />
            </div>
          ) : items.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-16 text-muted-foreground">
              <Download className="h-12 w-12 mb-4 opacity-50" />
              <p>Queue is empty</p>
              <p className="text-xs mt-1">Add videos from YT Scraper to start downloading</p>
            </div>
          ) : (
            <div className="p-4 space-y-3">
              {items.map((item) => (
                <div
                  key={item.id}
                  className={`border rounded-lg p-3 ${
                    item.status === 'failed' ? 'border-red-500/50 bg-red-500/5' :
                    item.status === 'completed' ? 'border-green-500/50 bg-green-500/5' :
                    item.status === 'processing' ? 'border-purple-500/50' :
                    item.status === 'downloading' ? 'border-blue-500/50' :
                    ''
                  }`}
                >
                  <div className="flex gap-3">
                    {/* Thumbnail */}
                    <div className="w-24 h-14 rounded overflow-hidden shrink-0 bg-muted">
                      {item.thumbnail && (
                        <img
                          src={item.thumbnail}
                          alt={item.title}
                          className="w-full h-full object-cover"
                        />
                      )}
                    </div>

                    {/* Info */}
                    <div className="flex-1 min-w-0">
                      <h4 className="font-medium text-sm line-clamp-1">{item.title}</h4>
                      <p className="text-xs text-muted-foreground">{item.channel}</p>

                      <div className="flex items-center gap-2 mt-1">
                        {getStatusBadge(item.status)}
                        <Badge variant="outline" className="text-xs">{item.resolution}p</Badge>
                        {item.clips_count > 0 && (
                          <Badge variant="secondary" className="text-xs">{item.clips_count} clips</Badge>
                        )}
                      </div>

                      {/* Progress Bar */}
                      {(item.status === 'downloading' || item.status === 'processing') && (
                        <div className="mt-2">
                          <Progress value={getProgress(item)} className="h-1" />
                          <p className="text-xs text-muted-foreground mt-1">
                            {item.status === 'downloading' ? 'Downloading' : 'Processing'}: {getProgress(item).toFixed(0)}%
                          </p>
                        </div>
                      )}

                      {/* Error Message */}
                      {item.error && (
                        <p className="text-xs text-red-500 mt-1">{item.error}</p>
                      )}
                    </div>

                    {/* Actions */}
                    <div className="flex flex-col gap-1">
                      {/* Cancel button for downloading/processing items */}
                      {(item.status === 'downloading' || item.status === 'processing') && (
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-7 w-7 text-red-500 hover:text-red-600"
                          onClick={() => removeItem(item.id, true)}
                          title="Cancel"
                        >
                          <X className="h-3 w-3" />
                        </Button>
                      )}
                      {/* Pause/Resume button for pending/downloaded/paused items */}
                      {(item.status === 'pending' || item.status === 'downloaded') && (
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-7 w-7"
                          onClick={() => pauseItem(item.id)}
                          title="Pause"
                        >
                          <Pause className="h-3 w-3" />
                        </Button>
                      )}
                      {item.status === 'paused' && (
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-7 w-7 text-green-500"
                          onClick={() => resumeItem(item.id)}
                          title="Resume"
                        >
                          <Play className="h-3 w-3" />
                        </Button>
                      )}
                      {item.status === 'failed' && (
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-7 w-7"
                          onClick={() => retryItem(item.id)}
                          title="Retry"
                        >
                          <RotateCcw className="h-3 w-3" />
                        </Button>
                      )}
                      {item.status !== 'downloading' && item.status !== 'processing' && (
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-7 w-7 text-muted-foreground hover:text-red-500"
                          onClick={() => removeItem(item.id)}
                          title="Remove"
                        >
                          <Trash2 className="h-3 w-3" />
                        </Button>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </ScrollArea>

        {/* Footer */}
        {(stats.completed > 0 || stats.failed > 0) && (
          <div className="flex items-center justify-end p-3 border-t">
            <Button variant="outline" size="sm" onClick={clearCompleted}>
              <Trash2 className="h-3 w-3 mr-1" />
              Clear Completed ({stats.completed + stats.failed})
            </Button>
          </div>
        )}
      </div>
    </div>
  )
}
