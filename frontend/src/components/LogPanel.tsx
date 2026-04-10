/**
 * LogPanel Component
 * Shows processing logs with detailed information
 */

import { useRef, useEffect } from 'react'
import { Terminal, Trash2, ChevronDown, ChevronUp } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Badge } from '@/components/ui/badge'
import { useAppStore } from '@/stores/appStore'
import { cn } from '@/lib/utils'
import { useState } from 'react'

export function LogPanel() {
  const { logs, clearLogs, currentStage, isProcessing } = useAppStore()
  const scrollRef = useRef<HTMLDivElement>(null)
  const [expanded, setExpanded] = useState(true)

  // Auto-scroll to bottom when new logs are added
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [logs])

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
    <Card className="flex flex-col">
      <CardHeader className="pb-3 flex-shrink-0">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <CardTitle className="text-base flex items-center gap-2">
              <Terminal className="h-4 w-4" />
              Logs
            </CardTitle>
            {isProcessing && (
              <Badge variant="outline" className={cn("text-xs", getStageColor(currentStage), "text-white border-0")}>
                {getStageLabel(currentStage)}
              </Badge>
            )}
            <Badge variant="secondary" className="text-xs">
              {logs.length} entries
            </Badge>
          </div>
          <div className="flex items-center gap-1">
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7"
              onClick={() => setExpanded(!expanded)}
              title={expanded ? "Collapse" : "Expand"}
            >
              {expanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
            </Button>
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
        </div>
      </CardHeader>
      {expanded && (
        <CardContent className="flex-1 overflow-hidden p-0">
          <ScrollArea className="h-[250px]">
            <div ref={scrollRef} className="px-4 py-2 font-mono text-xs space-y-0.5">
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
        </CardContent>
      )}
    </Card>
  )
}
