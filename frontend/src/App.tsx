/**
 * Auto Clip Maker - Main App Component
 */

import { useEffect } from 'react'
import { Header } from '@/components/Header'
import { Sidebar } from '@/components/Sidebar'
import { VideoInput } from '@/components/VideoInput'
import { ProgressPanel } from '@/components/ProgressPanel'
import { ClipsList } from '@/components/ClipsList'
import { ActionsBar } from '@/components/ActionsBar'
import { SubtitlePanel } from '@/components/SubtitlePanel'
import { useWebSocket } from '@/hooks/useWebSocket'
import { useApi } from '@/hooks/useApi'

function App() {
  // Initialize WebSocket connection
  useWebSocket()
  const { loadSettings } = useApi()

  // Load settings on mount
  useEffect(() => {
    loadSettings()
  }, [loadSettings])

  return (
    <div className="min-h-screen bg-background flex flex-col">
      <Header />

      <div className="flex-1 flex overflow-hidden">
        {/* Left Sidebar */}
        <Sidebar />

        {/* Main Content */}
        <main className="flex-1 overflow-auto p-4">
          <div className="max-w-6xl mx-auto space-y-4">
            {/* Top Row - Input and Progress */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <VideoInput />
              <ProgressPanel />
            </div>

            {/* Subtitle Settings and Clips */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
              <div className="lg:col-span-2">
                <ClipsList />
              </div>
              <div>
                <SubtitlePanel />
              </div>
            </div>

          </div>
        </main>
      </div>

      <ActionsBar />
    </div>
  )
}

export default App
