/**
 * Header Component
 * Shows logo, version, and Log button
 */

import { useEffect, useState, useRef } from 'react'
import { Zap, Cpu, Terminal, Trash2, Youtube, Settings, ListOrdered, FolderOpen, Upload } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog'
import { useAppStore } from '@/stores/appStore'
import { useApi } from '@/hooks/useApi'
import { cn } from '@/lib/utils'
import { YTScraper } from './YTScraper'
import { AutoProcessSettings } from './AutoProcessSettings'
import { DownloadQueue } from './DownloadQueue'
import { Gallery } from './Gallery'
import { UploadPanel } from './UploadPanel'

export function Header() {
  const { gpuStatus, logs, clearLogs, currentStage, isProcessing } = useAppStore()
  const { loadGPUStatus } = useApi()
  const [logOpen, setLogOpen] = useState(false)
  const [ytScraperOpen, setYtScraperOpen] = useState(false)
  const [autoProcessOpen, setAutoProcessOpen] = useState(false)
  const [queueOpen, setQueueOpen] = useState(false)
  const [galleryOpen, setGalleryOpen] = useState(false)
  const [uploadOpen, setUploadOpen] = useState(false)
  const [queueCount, setQueueCount] = useState(0)
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    loadGPUStatus()
    loadQueueStats()
    // Poll queue stats every 1 minute when queue panel is closed
    const interval = setInterval(loadQueueStats, 60000)
    return () => clearInterval(interval)
  }, [loadGPUStatus])

  const loadQueueStats = async () => {
    try {
      const response = await fetch('/api/queue/stats')
      const stats = await response.json()
      setQueueCount(stats.pending + stats.downloading + stats.downloaded + stats.processing)
    } catch {
      // Ignore errors
    }
  }

  // Auto-scroll to bottom when new logs are added
  useEffect(() => {
    if (scrollRef.current && logOpen) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [logs, logOpen])

  const getStageLabel = (stage: string) => {
    switch (stage) {
      case 'downloading': return 'Downloading'
      case 'transcribing': return 'Transcribing'
      case 'analyzing': return 'Analyzing'
      case 'exporting': return 'Exporting'
      default: return 'Idle'
    }
  }

  const getStageColor = (stage: string) => {
    switch (stage) {
      case 'downloading': return 'bg-blue-500'
      case 'transcribing': return 'bg-purple-500'
      case 'analyzing': return 'bg-orange-500'
      case 'exporting': return 'bg-green-500'
      default: return 'bg-gray-500'
    }
  }

  return (
    <header className="border-b bg-card shrink-0">
      <div className="flex h-12 items-center justify-between px-4">
        <div className="flex items-center gap-2">
          <Zap className="h-5 w-5 text-primary" />
          <h1 className="text-base font-semibold">Auto Clip Maker</h1>
          <Badge variant="secondary" className="text-xs">v2.0</Badge>

          {/* YT Scraper Button */}
          <Button
            variant="outline"
            size="sm"
            className="h-7 gap-1.5 text-xs ml-2"
            onClick={() => setYtScraperOpen(true)}
          >
            <Youtube className="h-3.5 w-3.5 text-red-500" />
            YT Scraper
          </Button>

          {/* Auto Process Button */}
          <Button
            variant="outline"
            size="sm"
            className="h-7 gap-1.5 text-xs"
            onClick={() => setAutoProcessOpen(true)}
          >
            <Settings className="h-3.5 w-3.5 text-primary" />
            Auto Process
          </Button>

          {/* Queue Button */}
          <Button
            variant="outline"
            size="sm"
            className="h-7 gap-1.5 text-xs"
            onClick={() => setQueueOpen(true)}
          >
            <ListOrdered className="h-3.5 w-3.5 text-blue-500" />
            Queue
            {queueCount > 0 && (
              <Badge variant="secondary" className="h-4 px-1 text-[10px]">
                {queueCount}
              </Badge>
            )}
          </Button>

          {/* Gallery Button */}
          <Button
            variant="outline"
            size="sm"
            className="h-7 gap-1.5 text-xs"
            onClick={() => setGalleryOpen(true)}
          >
            <FolderOpen className="h-3.5 w-3.5 text-green-500" />
            Gallery
          </Button>

          {/* Upload Button */}
          <Button
            variant="outline"
            size="sm"
            className="h-7 gap-1.5 text-xs"
            onClick={() => setUploadOpen(true)}
          >
            <Upload className="h-3.5 w-3.5 text-purple-500" />
            Upload
          </Button>
        </div>

        <div className="flex items-center gap-3 text-xs">
          {/* Log Button */}
          <Dialog open={logOpen} onOpenChange={setLogOpen}>
            <DialogTrigger asChild>
              <Button variant="outline" size="sm" className="h-7 gap-1.5 text-xs">
                <Terminal className="h-3.5 w-3.5" />
                Log
                {logs.length > 0 && (
                  <Badge variant="secondary" className="h-4 px-1 text-[10px]">
                    {logs.length}
                  </Badge>
                )}
              </Button>
            </DialogTrigger>
            <DialogContent className="max-w-2xl max-h-[80vh]">
              <DialogHeader>
                <div className="flex items-center justify-between pr-8">
                  <div className="flex items-center gap-3">
                    <DialogTitle className="flex items-center gap-2">
                      <Terminal className="h-4 w-4" />
                      Processing Logs
                    </DialogTitle>
                    {isProcessing && (
                      <Badge variant="outline" className={cn("text-xs", getStageColor(currentStage), "text-white border-0")}>
                        {getStageLabel(currentStage)}
                      </Badge>
                    )}
                    <Badge variant="secondary" className="text-xs">
                      {logs.length} entries
                    </Badge>
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-7 w-7"
                    onClick={clearLogs}
                    title="Clear logs"
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              </DialogHeader>
              <ScrollArea className="h-[400px] border rounded-md">
                <div ref={scrollRef} className="p-3 font-mono text-xs space-y-0.5">
                  {logs.length === 0 ? (
                    <p className="text-muted-foreground py-4 text-center">No logs yet. Start processing a video to see logs.</p>
                  ) : (
                    logs.map((log, index) => (
                      <div
                        key={index}
                        className={cn(
                          "py-1 px-2 rounded hover:bg-muted/50 flex items-start gap-2",
                          log.level === 'error' && "bg-red-500/10 text-red-400",
                          log.level === 'warning' && "bg-yellow-500/10 text-yellow-400",
                          log.level === 'info' && "bg-green-500/10 text-green-400"
                        )}
                      >
                        <span className="text-muted-foreground shrink-0">[{log.timestamp}]</span>
                        <span className="break-all">{log.message}</span>
                      </div>
                    ))
                  )}
                </div>
              </ScrollArea>
            </DialogContent>
          </Dialog>

          {/* GPU Status */}
          {gpuStatus && gpuStatus.cuda_available && (
            <div className="flex items-center gap-1.5 text-green-500">
              <Cpu className="h-3.5 w-3.5" />
              <span>{gpuStatus.gpu_name?.replace('NVIDIA GeForce ', '')}</span>
            </div>
          )}
        </div>
      </div>

      {/* YT Scraper Panel */}
      <YTScraper
        isOpen={ytScraperOpen}
        onClose={() => setYtScraperOpen(false)}
      />

      {/* Auto Process Settings Panel */}
      <AutoProcessSettings
        isOpen={autoProcessOpen}
        onClose={() => setAutoProcessOpen(false)}
      />

      {/* Download Queue Panel */}
      <DownloadQueue
        isOpen={queueOpen}
        onClose={() => setQueueOpen(false)}
      />

      {/* Gallery Panel */}
      <Gallery
        isOpen={galleryOpen}
        onClose={() => setGalleryOpen(false)}
      />

      {/* Upload Panel */}
      <UploadPanel
        isOpen={uploadOpen}
        onClose={() => setUploadOpen(false)}
      />
    </header>
  )
}
