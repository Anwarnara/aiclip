/**
 * ProgressPanel Component
 * Shows processing progress for each stage
 */

import { Download, Mic, Brain, Film } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { useAppStore } from '@/stores/appStore'

interface ProgressItemProps {
  icon: React.ReactNode
  label: string
  value: number
  status: string
  active: boolean
}

function ProgressItem({ icon, label, value, status, active }: ProgressItemProps) {
  return (
    <div className={`space-y-1 ${active ? '' : 'opacity-50'}`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {icon}
          <span className="text-sm font-medium">{label}</span>
        </div>
        <span className="text-xs text-muted-foreground">{Math.round(value)}%</span>
      </div>
      <Progress value={value} className="h-2" />
      <p className="text-xs text-muted-foreground truncate">{status}</p>
    </div>
  )
}

export function ProgressPanel() {
  const { progress, progressStatus, currentStage, isProcessing } = useAppStore()

  const stages = [
    { key: 'download', label: 'Download', icon: <Download className="h-4 w-4" />, stageId: 'downloading' },
    { key: 'transcribe', label: 'Transcribe', icon: <Mic className="h-4 w-4" />, stageId: 'transcribing' },
    { key: 'analyze', label: 'Analyze', icon: <Brain className="h-4 w-4" />, stageId: 'analyzing' },
    { key: 'export', label: 'Export', icon: <Film className="h-4 w-4" />, stageId: 'exporting' },
  ]

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Progress</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {stages.map(({ key, label, icon, stageId }) => (
          <ProgressItem
            key={key}
            icon={icon}
            label={label}
            value={progress[key as keyof typeof progress]}
            status={progressStatus[key as keyof typeof progressStatus]}
            active={isProcessing && (currentStage === stageId || progress[key as keyof typeof progress] > 0)}
          />
        ))}

        {!isProcessing && currentStage === 'idle' && progress.download === 0 && (
          <p className="text-sm text-muted-foreground text-center py-4">
            Enter a video URL or file path to start
          </p>
        )}
      </CardContent>
    </Card>
  )
}
