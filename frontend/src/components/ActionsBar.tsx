/**
 * ActionsBar Component
 * Select all, deselect all, and export buttons
 */

import { CheckSquare, Square, Download } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Separator } from '@/components/ui/separator'
import { useAppStore } from '@/stores/appStore'
import { useApi } from '@/hooks/useApi'

export function ActionsBar() {
  const { clips, selectedClips, selectAllClips, deselectAllClips, isProcessing } = useAppStore()
  const { exportClips, loading } = useApi()

  const handleExport = () => {
    const clipIds = Array.from(selectedClips)
    exportClips(clipIds)
  }

  const hasClips = clips.length > 0
  const hasSelection = selectedClips.size > 0

  return (
    <footer className="border-t bg-card">
      <div className="container flex h-14 items-center justify-between px-4">
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={selectAllClips}
            disabled={!hasClips || isProcessing}
            className="gap-2"
          >
            <CheckSquare className="h-4 w-4" />
            Select All
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={deselectAllClips}
            disabled={!hasClips || isProcessing}
            className="gap-2"
          >
            <Square className="h-4 w-4" />
            Deselect All
          </Button>

          <Separator orientation="vertical" className="h-6 mx-2" />

          <span className="text-sm text-muted-foreground">
            {selectedClips.size} of {clips.length} clips selected
          </span>
        </div>

        <Button
          onClick={handleExport}
          disabled={!hasSelection || isProcessing || loading}
          className="gap-2"
        >
          <Download className="h-4 w-4" />
          Export Selected Clips
        </Button>
      </div>
    </footer>
  )
}
