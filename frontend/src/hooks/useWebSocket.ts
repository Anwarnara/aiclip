/**
 * WebSocket Hook for Real-time Updates
 */

import { useEffect, useCallback } from 'react'
import { wsClient } from '@/lib/websocket'
import { useAppStore } from '@/stores/appStore'
import type { WebSocketMessage, Clip, LogEntry, VideoInfo } from '@/types'

export function useWebSocket() {
  const {
    setVideoInfo,
    setProcessing,
    setCurrentStage,
    updateProgress,
    setClips,
    addLog,
    setSettings
  } = useAppStore()

  const handleMessage = useCallback((message: WebSocketMessage) => {
    const { event, data } = message

    switch (event) {
      case 'connected': {
        const connData = data as {
          status: { is_processing: boolean; current_stage: string };
          settings: Record<string, unknown>;
          clips: Clip[];
          logs: LogEntry[];
        }
        setProcessing(connData.status.is_processing)
        setCurrentStage(connData.status.current_stage)
        setSettings(connData.settings)
        if (connData.clips?.length) {
          setClips(connData.clips)
        }
        connData.logs?.forEach(log => addLog(log))
        break
      }

      case 'progress': {
        const progressData = data as { stage: string; value: number; status: string }
        updateProgress(
          progressData.stage as 'download' | 'transcribe' | 'analyze' | 'export',
          progressData.value,
          progressData.status
        )
        break
      }

      case 'log': {
        const logData = data as LogEntry
        addLog({
          timestamp: logData.timestamp || new Date().toLocaleTimeString(),
          message: logData.message,
          level: logData.level || 'info'
        })
        break
      }

      case 'video_info': {
        const infoData = data as VideoInfo
        setVideoInfo(infoData)
        break
      }

      case 'clips_ready': {
        const clipsData = data as { clips: Clip[] }
        setClips(clipsData.clips)
        setProcessing(false)
        setCurrentStage('idle')
        break
      }

      case 'training_progress': {
        const trainData = data as { percent: number; status: string }
        // Update export progress bar to show training status
        // We keep value low (indigo) or just show the text
        updateProgress('export', trainData.percent, `Training Model: ${trainData.status}`)
        break
      }

      case 'export_progress': {
        const exportData = data as { current: number; total: number; status: string; percent: number }
        updateProgress('export', (exportData.current - 1 + exportData.percent / 100) / exportData.total * 100, exportData.status)
        break
      }

      case 'export_complete': {
        setProcessing(false)
        setCurrentStage('idle')
        break
      }

      case 'error': {
        const errorData = data as { message: string }
        addLog({
          timestamp: new Date().toLocaleTimeString(),
          message: errorData.message,
          level: 'error'
        })
        break
      }

      case 'pong':
        // Ignore pong responses
        break

      default:
        console.log('Unknown WebSocket event:', event, data)
    }
  }, [setVideoInfo, setProcessing, setCurrentStage, updateProgress, setClips, addLog, setSettings])

  useEffect(() => {
    // Connect to WebSocket
    wsClient.connect()

    // Subscribe to messages
    const unsubscribe = wsClient.subscribe(handleMessage)

    // Cleanup
    return () => {
      unsubscribe()
    }
  }, [handleMessage])

  return {
    isConnected: wsClient.isConnected,
    send: wsClient.send.bind(wsClient)
  }
}
