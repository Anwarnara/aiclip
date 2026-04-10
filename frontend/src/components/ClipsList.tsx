/**
 * ClipsList Component
 * Shows discovered clips with selection
 */

import { Film, Clock } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Checkbox } from '@/components/ui/checkbox'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import { useAppStore } from '@/stores/appStore'

export function ClipsList() {
  const { clips, selectedClips, toggleClip } = useAppStore()

  if (clips.length === 0) {
    return (
      <Card className="h-full">
        <CardHeader className="pb-3">
          <CardTitle className="text-base flex items-center gap-2">
            <Film className="h-4 w-4" />
            Discovered Clips
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground text-center py-8">
            No clips discovered yet. Process a video to find interesting clips.
          </p>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="h-full flex flex-col">
      <CardHeader className="pb-3 flex-shrink-0">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base flex items-center gap-2">
            <Film className="h-4 w-4" />
            Discovered Clips
          </CardTitle>
          <Badge variant="secondary">
            {selectedClips.size}/{clips.length} selected
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="flex-1 overflow-hidden p-0">
        <ScrollArea className="h-[300px] px-6">
          {clips.map((clip, index) => (
            <div key={clip.id}>
              <div
                className="py-3 flex items-start gap-3 cursor-pointer hover:bg-muted/50 rounded px-2 -mx-2"
                onClick={() => toggleClip(clip.id)}
              >
                <Checkbox
                  checked={selectedClips.has(clip.id)}
                  onCheckedChange={() => toggleClip(clip.id)}
                  className="mt-1"
                />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium truncate">{clip.title}</span>
                  </div>
                  <div className="flex items-center gap-2 mt-1">
                    <Badge variant="outline" className="text-xs">
                      <Clock className="h-3 w-3 mr-1" />
                      {clip.start_formatted} - {clip.end_formatted}
                    </Badge>
                    <span className="text-xs text-muted-foreground">
                      ({Math.round(clip.duration)}s)
                    </span>
                  </div>
                  <p className="text-xs text-muted-foreground mt-1 line-clamp-2">
                    {clip.reason}
                  </p>
                </div>
              </div>
              {index < clips.length - 1 && <Separator />}
            </div>
          ))}
        </ScrollArea>
      </CardContent>
    </Card>
  )
}
