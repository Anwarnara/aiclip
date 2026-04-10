/**
 * API Client for Auto Clip Maker Backend
 */

import type { VideoInfo, Clip, Settings, GPUStatus, ProcessingStatus } from '@/types'

const API_BASE = '/api'

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: 'Unknown error' }))
    throw new Error(error.detail || error.error || 'Request failed')
  }
  return response.json()
}

export const api = {
  // Video endpoints
  async processYouTube(url: string): Promise<{ status: string; url: string }> {
    const response = await fetch(`${API_BASE}/video/youtube`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url })
    })
    return handleResponse(response)
  },

  async processLocal(path: string): Promise<{ status: string; path: string }> {
    const response = await fetch(`${API_BASE}/video/local`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url: path })
    })
    return handleResponse(response)
  },

  async getVideoInfo(): Promise<VideoInfo> {
    const response = await fetch(`${API_BASE}/video/info`)
    return handleResponse(response)
  },

  async cancelProcessing(): Promise<{ status: string }> {
    const response = await fetch(`${API_BASE}/video/cancel`, { method: 'POST' })
    return handleResponse(response)
  },

  async getStatus(): Promise<ProcessingStatus> {
    const response = await fetch(`${API_BASE}/video/status`)
    return handleResponse(response)
  },

  async getGPUStatus(): Promise<GPUStatus> {
    const response = await fetch(`${API_BASE}/video/gpu`)
    return handleResponse(response)
  },

  async uploadVideo(file: File): Promise<{ status: string; filename: string; path: string }> {
    const formData = new FormData()
    formData.append('file', file)
    const response = await fetch(`${API_BASE}/video/upload`, {
      method: 'POST',
      body: formData
    })
    return handleResponse(response)
  },

  // Clips endpoints
  async getClips(): Promise<Clip[]> {
    const response = await fetch(`${API_BASE}/clips`)
    return handleResponse(response)
  },

  async exportClips(clipIds: number[]): Promise<{ status: string; count: number }> {
    const response = await fetch(`${API_BASE}/clips/export`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ clip_ids: clipIds })
    })
    return handleResponse(response)
  },

  async cancelExport(): Promise<{ status: string }> {
    const response = await fetch(`${API_BASE}/clips/export/cancel`, { method: 'POST' })
    return handleResponse(response)
  },

  // Settings endpoints
  async getSettings(): Promise<Settings> {
    const response = await fetch(`${API_BASE}/settings`)
    return handleResponse(response)
  },

  async updateSettings(settings: Partial<Settings>): Promise<{ status: string; settings: Settings }> {
    const response = await fetch(`${API_BASE}/settings`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(settings)
    })
    return handleResponse(response)
  },

  async resetSettings(): Promise<{ status: string; settings: Settings }> {
    const response = await fetch(`${API_BASE}/settings/reset`, { method: 'POST' })
    return handleResponse(response)
  },

  // Health check
  async healthCheck(): Promise<{ status: string; version: string }> {
    const response = await fetch(`${API_BASE}/health`)
    return handleResponse(response)
  }
}
