/**
 * API Hook for making API calls
 */

import { useState, useCallback } from 'react'
import { api } from '@/lib/api'
import { useAppStore } from '@/stores/appStore'

export function useApi() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const {
    setProcessing,
    setCurrentStage,
    resetProgress,
    setClips,
    setSettings,
    setGPUStatus,
    addLog,
    activeTab,
    inputValue
  } = useAppStore()

  const startProcessing = useCallback(async () => {
    if (!inputValue.trim()) {
      setError('Please enter a URL or file path')
      return
    }

    setLoading(true)
    setError(null)
    resetProgress()

    try {
      if (activeTab === 'youtube') {
        await api.processYouTube(inputValue.trim())
      } else {
        await api.processLocal(inputValue.trim())
      }
      setProcessing(true)
      setCurrentStage('downloading')
    } catch (e) {
      const message = e instanceof Error ? e.message : 'Failed to start processing'
      setError(message)
      addLog({ timestamp: new Date().toLocaleTimeString(), message, level: 'error' })
    } finally {
      setLoading(false)
    }
  }, [activeTab, inputValue, resetProgress, setProcessing, setCurrentStage, addLog])

  const cancelProcessing = useCallback(async () => {
    try {
      await api.cancelProcessing()
    } catch (e) {
      console.error('Failed to cancel:', e)
    }
  }, [])

  const stopAllProcesses = useCallback(async () => {
    try {
      await api.stopAllProcesses()
      setProcessing(false)
      setCurrentStage('idle')
      resetProgress()
      addLog({ timestamp: new Date().toLocaleTimeString(), message: '⛔ All processes stopped', level: 'warning' })
    } catch (e) {
      console.error('Failed to stop all:', e)
    }
  }, [setProcessing, setCurrentStage, resetProgress, addLog])

  const exportClips = useCallback(async (clipIds: number[]) => {
    if (clipIds.length === 0) {
      setError('No clips selected')
      return
    }

    setLoading(true)
    setError(null)

    try {
      await api.exportClips(clipIds)
      setProcessing(true)
      setCurrentStage('exporting')
    } catch (e) {
      const message = e instanceof Error ? e.message : 'Failed to start export'
      setError(message)
    } finally {
      setLoading(false)
    }
  }, [setProcessing, setCurrentStage])

  const loadSettings = useCallback(async () => {
    try {
      const settings = await api.getSettings()
      setSettings(settings)
    } catch (e) {
      console.error('Failed to load settings:', e)
    }
  }, [setSettings])

  const saveSettings = useCallback(async (settings: Parameters<typeof api.updateSettings>[0]) => {
    try {
      const result = await api.updateSettings(settings)
      setSettings(result.settings)
    } catch (e) {
      console.error('Failed to save settings:', e)
    }
  }, [setSettings])

  const loadGPUStatus = useCallback(async () => {
    try {
      const status = await api.getGPUStatus()
      setGPUStatus(status)
    } catch (e) {
      console.error('Failed to load GPU status:', e)
    }
  }, [setGPUStatus])

  const loadClips = useCallback(async () => {
    try {
      const clips = await api.getClips()
      setClips(clips)
    } catch (e) {
      console.error('Failed to load clips:', e)
    }
  }, [setClips])

  return {
    loading,
    error,
    setError,
    startProcessing,
    cancelProcessing,
    stopAllProcesses,
    exportClips,
    loadSettings,
    saveSettings,
    loadGPUStatus,
    loadClips
  }
}
