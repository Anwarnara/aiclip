/**
 * Upload Panel
 * Settings and controls for auto-uploading to social media
 */

import { useState, useEffect, useCallback } from 'react'
import {
  Upload as UploadIcon,
  X,
  Loader2,
  Youtube,
  Facebook,
  Settings,
  History,
  RefreshCw,
  FolderOpen,
  Save,
  Play,
  Clock,
  CheckCircle2,
  Video,
  LogIn,
  LogOut,
  ExternalLink,
  AlertCircle
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Switch } from '@/components/ui/switch'
import { Label } from '@/components/ui/label'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'

interface UploadSettings {
  youtube_enabled: boolean
  youtube_api_key: string
  youtube_api_key_set?: boolean
  youtube_channel_id: string
  youtube_privacy: string
  youtube_category: string
  youtube_tags: string[]
  facebook_enabled: boolean
  facebook_access_token: string
  facebook_access_token_set?: boolean
  facebook_page_id: string
  auto_upload_on_complete: boolean
  default_title_template: string
  default_description_template: string
  upload_delay_seconds: number
}

interface YouTubeOAuthStatus {
  client_configured: boolean
  client_id_exists: boolean
  client_secret_exists: boolean
  authenticated: boolean
  channel_name: string | null
  channel_id: string | null
  token_expires: string | null
}

interface UploadHistoryItem {
  id: number
  platform: string
  file_path: string
  title: string
  status: string
  created_at: string
  error?: string
}

interface VideoItem {
  name: string
  path: string
  size: number
  metadata?: {
    title?: string
    description?: string
    tags?: string[]
  }
}

interface OutputFolder {
  folder_name: string
  folder_path: string
  videos: VideoItem[]
  video_count: number
  has_metadata: boolean
}

interface UploadPanelProps {
  isOpen: boolean
  onClose: () => void
}

const YT_CATEGORIES = [
  { id: "1", name: "Film & Animation" },
  { id: "2", name: "Autos & Vehicles" },
  { id: "10", name: "Music" },
  { id: "15", name: "Pets & Animals" },
  { id: "17", name: "Sports" },
  { id: "20", name: "Gaming" },
  { id: "22", name: "People & Blogs" },
  { id: "23", name: "Comedy" },
  { id: "24", name: "Entertainment" },
  { id: "25", name: "News & Politics" },
  { id: "26", name: "Howto & Style" },
  { id: "27", name: "Education" },
  { id: "28", name: "Science & Technology" },
]

export function UploadPanel({ isOpen, onClose }: UploadPanelProps) {
  const [activeTab, setActiveTab] = useState<'settings' | 'folders' | 'history'>('settings')
  const [settings, setSettings] = useState<UploadSettings | null>(null)
  const [history, setHistory] = useState<UploadHistoryItem[]>([])
  const [folders, setFolders] = useState<OutputFolder[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [isSaving, setIsSaving] = useState(false)
  const [ytOAuthStatus, setYtOAuthStatus] = useState<YouTubeOAuthStatus | null>(null)
  const [selectedFolder, setSelectedFolder] = useState<string | null>(null)
  const [uploadingFolder, setUploadingFolder] = useState<string | null>(null)
  const [hasChanges, setHasChanges] = useState(false)
  const [isLoggingIn, setIsLoggingIn] = useState(false)

  const loadData = useCallback(async () => {
    setIsLoading(true)
    try {
      const [settingsRes, historyRes, foldersRes, oauthRes] = await Promise.all([
        fetch('/api/upload/settings'),
        fetch('/api/upload/history'),
        fetch('/api/upload/folders'),
        fetch('/api/upload/youtube/oauth-status')
      ])
      const settingsData = await settingsRes.json()
      const historyData = await historyRes.json()
      const foldersData = await foldersRes.json()
      const oauthData = await oauthRes.json()

      setSettings(settingsData)
      setHistory(historyData.history || [])
      setFolders(foldersData.folders || [])
      setYtOAuthStatus(oauthData)
      setHasChanges(false)
    } catch (err) {
      console.error('Error loading upload data:', err)
    } finally {
      setIsLoading(false)
    }
  }, [])

  const handleYouTubeLogin = async () => {
    setIsLoggingIn(true)
    try {
      const res = await fetch('/api/upload/youtube/auth-url')
      const data = await res.json()

      if (data.success && data.auth_url) {
        // Open auth URL in new window
        const authWindow = window.open(data.auth_url, 'YouTube Login', 'width=600,height=700')

        // Listen for callback
        const checkCallback = setInterval(async () => {
          try {
            if (authWindow?.closed) {
              clearInterval(checkCallback)
              setIsLoggingIn(false)
              // Reload OAuth status
              const statusRes = await fetch('/api/upload/youtube/oauth-status')
              const statusData = await statusRes.json()
              setYtOAuthStatus(statusData)
            }
          } catch {
            // Window still open or cross-origin
          }
        }, 1000)
      } else {
        alert(data.error || 'Failed to get auth URL')
        setIsLoggingIn(false)
      }
    } catch (err) {
      console.error('Error starting YouTube login:', err)
      setIsLoggingIn(false)
    }
  }

  const handleYouTubeLogout = async () => {
    if (!confirm('Logout from YouTube?')) return

    try {
      await fetch('/api/upload/youtube/logout', { method: 'POST' })
      setYtOAuthStatus(prev => prev ? { ...prev, authenticated: false, channel_name: null } : null)
    } catch (err) {
      console.error('Error logging out:', err)
    }
  }

  const checkYouTubeStatus = async () => {
    try {
      const res = await fetch('/api/upload/youtube/oauth-status')
      const data = await res.json()
      setYtOAuthStatus(data)
    } catch {
      // Ignore errors
    }
  }

  useEffect(() => {
    if (isOpen) {
      loadData()
    }
  }, [isOpen, loadData])

  const updateSettingLocal = (key: string, value: unknown) => {
    if (!settings) return
    setSettings({ ...settings, [key]: value })
    setHasChanges(true)
  }

  const saveSettings = async () => {
    if (!settings) return

    setIsSaving(true)
    try {
      await fetch('/api/upload/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings)
      })
      setHasChanges(false)
      checkYouTubeStatus()
    } catch (err) {
      console.error('Error saving settings:', err)
    } finally {
      setIsSaving(false)
    }
  }

  const clearHistory = async () => {
    if (!confirm('Clear all upload history?')) return

    try {
      await fetch('/api/upload/history', { method: 'DELETE' })
      setHistory([])
    } catch (err) {
      console.error('Error clearing history:', err)
    }
  }

  const uploadFolder = async (folderPath: string) => {
    // Check if authenticated for YouTube
    if (!ytOAuthStatus?.authenticated) {
      alert('Please login to YouTube first')
      return
    }

    setUploadingFolder(folderPath)
    try {
      const res = await fetch('/api/upload/youtube/upload-folder', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          folder_path: folderPath,
          platform: 'youtube'
        })
      })
      const data = await res.json()

      if (res.status === 401) {
        alert('Not authenticated. Please login to YouTube first.')
        return
      }

      if (data.success) {
        alert(`Started uploading ${data.video_count} videos. Delay: ${data.delay_seconds}s between uploads.`)
        loadData()
        setActiveTab('history')
      } else {
        alert(data.detail || 'Upload failed')
      }
    } catch (err) {
      console.error('Error uploading folder:', err)
    } finally {
      setUploadingFolder(null)
    }
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 bg-background/80 backdrop-blur-sm">
      <div className="fixed inset-4 z-50 bg-card border rounded-lg shadow-lg flex flex-col max-w-3xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b">
          <div className="flex items-center gap-2">
            <UploadIcon className="h-5 w-5 text-primary" />
            <h2 className="text-lg font-semibold">Upload Settings</h2>
            {isSaving && <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />}
            {hasChanges && <Badge variant="secondary" className="text-xs">Unsaved</Badge>}
          </div>
          <div className="flex items-center gap-2">
            {hasChanges && (
              <Button variant="default" size="sm" onClick={saveSettings} disabled={isSaving}>
                <Save className="h-4 w-4 mr-1" />
                Save
              </Button>
            )}
            <Button variant="ghost" size="icon" onClick={loadData} title="Refresh">
              <RefreshCw className="h-4 w-4" />
            </Button>
            <Button variant="ghost" size="icon" onClick={onClose}>
              <X className="h-4 w-4" />
            </Button>
          </div>
        </div>

        {/* Tabs */}
        <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as 'settings' | 'folders' | 'history')} className="flex-1 flex flex-col">
          <div className="border-b px-4">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="settings" className="flex items-center gap-2">
                <Settings className="h-4 w-4" />
                Settings
              </TabsTrigger>
              <TabsTrigger value="folders" className="flex items-center gap-2">
                <FolderOpen className="h-4 w-4" />
                Folders
                {folders.length > 0 && (
                  <Badge variant="secondary" className="ml-1 text-xs">{folders.length}</Badge>
                )}
              </TabsTrigger>
              <TabsTrigger value="history" className="flex items-center gap-2">
                <History className="h-4 w-4" />
                History
                {history.length > 0 && (
                  <Badge variant="secondary" className="ml-1 text-xs">{history.length}</Badge>
                )}
              </TabsTrigger>
            </TabsList>
          </div>

          {isLoading ? (
            <div className="flex items-center justify-center py-16">
              <Loader2 className="h-8 w-8 animate-spin" />
            </div>
          ) : (
            <>
              {/* Settings Tab */}
              <TabsContent value="settings" className="flex-1 m-0">
                <ScrollArea className="h-[calc(100vh-250px)]">
                  {settings && (
                    <div className="p-4 space-y-6">
                      {/* YouTube OAuth Section */}
                      <div className="space-y-4">
                        <div className="flex items-center gap-2">
                          <Youtube className="h-5 w-5 text-red-500" />
                          <h3 className="font-semibold">YouTube Upload</h3>
                        </div>

                        {/* OAuth Status Card */}
                        <div className={`p-4 rounded-lg border ${
                          ytOAuthStatus?.authenticated
                            ? 'bg-green-500/10 border-green-500/30'
                            : 'bg-muted border-muted-foreground/20'
                        }`}>
                          {ytOAuthStatus?.authenticated ? (
                            <div className="space-y-3">
                              <div className="flex items-center gap-2">
                                <CheckCircle2 className="h-5 w-5 text-green-500" />
                                <span className="font-medium">Connected to YouTube</span>
                              </div>
                              <div className="text-sm space-y-1">
                                <p><strong>Channel:</strong> {ytOAuthStatus.channel_name}</p>
                                <p className="text-muted-foreground text-xs">
                                  ID: {ytOAuthStatus.channel_id}
                                </p>
                              </div>
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={handleYouTubeLogout}
                                className="mt-2"
                              >
                                <LogOut className="h-4 w-4 mr-2" />
                                Logout
                              </Button>
                            </div>
                          ) : (
                            <div className="space-y-3">
                              <div className="flex items-center gap-2">
                                <AlertCircle className="h-5 w-5 text-yellow-500" />
                                <span className="font-medium">Not Connected</span>
                              </div>

                              {/* Client credentials status */}
                              <div className="text-sm space-y-1">
                                <p className="flex items-center gap-2">
                                  {ytOAuthStatus?.client_id_exists ? (
                                    <CheckCircle2 className="h-3 w-3 text-green-500" />
                                  ) : (
                                    <X className="h-3 w-3 text-red-500" />
                                  )}
                                  client_id.txt: {ytOAuthStatus?.client_id_exists ? 'Found' : 'Not found'}
                                </p>
                                <p className="flex items-center gap-2">
                                  {ytOAuthStatus?.client_secret_exists ? (
                                    <CheckCircle2 className="h-3 w-3 text-green-500" />
                                  ) : (
                                    <X className="h-3 w-3 text-red-500" />
                                  )}
                                  client_secret.txt: {ytOAuthStatus?.client_secret_exists ? 'Found' : 'Not found'}
                                </p>
                              </div>

                              {!ytOAuthStatus?.client_configured ? (
                                <div className="text-xs text-muted-foreground mt-2 p-2 bg-muted/50 rounded">
                                  <p className="font-medium mb-1">Setup required:</p>
                                  <ol className="list-decimal list-inside space-y-0.5">
                                    <li>Go to Google Cloud Console</li>
                                    <li>Create OAuth2 credentials</li>
                                    <li>Save client_id.txt and client_secret.txt to /data folder</li>
                                  </ol>
                                  <a
                                    href="https://console.cloud.google.com/apis/credentials"
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="inline-flex items-center gap-1 text-primary hover:underline mt-2"
                                  >
                                    Open Google Cloud Console
                                    <ExternalLink className="h-3 w-3" />
                                  </a>
                                </div>
                              ) : (
                                <Button
                                  onClick={handleYouTubeLogin}
                                  disabled={isLoggingIn}
                                  className="mt-2"
                                >
                                  {isLoggingIn ? (
                                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                                  ) : (
                                    <LogIn className="h-4 w-4 mr-2" />
                                  )}
                                  Login with Google
                                </Button>
                              )}
                            </div>
                          )}
                        </div>

                        {/* Upload Settings (only show if authenticated) */}
                        {ytOAuthStatus?.authenticated && (
                          <>
                            <div className="space-y-2">
                              <Label>Default Privacy</Label>
                              <Select
                                value={settings.youtube_privacy}
                                onValueChange={(v) => updateSettingLocal('youtube_privacy', v)}
                              >
                                <SelectTrigger>
                                  <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                  <SelectItem value="private">Private</SelectItem>
                                  <SelectItem value="unlisted">Unlisted</SelectItem>
                                  <SelectItem value="public">Public</SelectItem>
                                </SelectContent>
                              </Select>
                            </div>

                            <div className="space-y-2">
                              <Label>Default Category</Label>
                              <Select
                                value={settings.youtube_category}
                                onValueChange={(v) => updateSettingLocal('youtube_category', v)}
                              >
                                <SelectTrigger>
                                  <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                  {YT_CATEGORIES.map((cat) => (
                                    <SelectItem key={cat.id} value={cat.id}>
                                      {cat.name}
                                    </SelectItem>
                                  ))}
                                </SelectContent>
                              </Select>
                            </div>
                          </>
                        )}
                      </div>

                      {/* Facebook Settings */}
                      <div className="space-y-4 pt-4 border-t">
                        <div className="flex items-center gap-2">
                          <Facebook className="h-5 w-5 text-blue-600" />
                          <h3 className="font-semibold">Facebook</h3>
                        </div>

                        <div className="flex items-center justify-between">
                          <Label>Enable Facebook Upload</Label>
                          <Switch
                            checked={settings.facebook_enabled}
                            onCheckedChange={(v) => updateSettingLocal('facebook_enabled', v)}
                          />
                        </div>

                        <div className="space-y-2">
                          <Label>Page Access Token</Label>
                          <Input
                            type="password"
                            value={settings.facebook_access_token}
                            onChange={(e) => updateSettingLocal('facebook_access_token', e.target.value)}
                            placeholder="Enter Facebook Page Access Token"
                          />
                        </div>

                        <div className="space-y-2">
                          <Label>Page ID</Label>
                          <Input
                            value={settings.facebook_page_id}
                            onChange={(e) => updateSettingLocal('facebook_page_id', e.target.value)}
                            placeholder="Enter Facebook Page ID"
                          />
                        </div>
                      </div>

                      {/* Upload Settings */}
                      <div className="space-y-4 pt-4 border-t">
                        <h3 className="font-semibold flex items-center gap-2">
                          <Clock className="h-4 w-4" />
                          Upload Settings
                        </h3>

                        <div className="space-y-2">
                          <Label>Delay Between Uploads (seconds)</Label>
                          <Input
                            type="number"
                            min={10}
                            max={3600}
                            value={settings.upload_delay_seconds || 60}
                            onChange={(e) => updateSettingLocal('upload_delay_seconds', parseInt(e.target.value) || 60)}
                          />
                          <p className="text-xs text-muted-foreground">
                            Wait time between uploading each video to avoid spam detection (min: 10s)
                          </p>
                        </div>

                        <div className="flex items-center justify-between">
                          <div>
                            <Label>Auto Upload on Complete</Label>
                            <p className="text-xs text-muted-foreground">
                              Automatically upload clips when processing completes
                            </p>
                          </div>
                          <Switch
                            checked={settings.auto_upload_on_complete}
                            onCheckedChange={(v) => updateSettingLocal('auto_upload_on_complete', v)}
                          />
                        </div>

                        <div className="space-y-2">
                          <Label>Title Template</Label>
                          <Input
                            value={settings.default_title_template}
                            onChange={(e) => updateSettingLocal('default_title_template', e.target.value)}
                            placeholder="{original_title} - Clip {clip_number}"
                          />
                        </div>

                        <div className="space-y-2">
                          <Label>Description Template</Label>
                          <Textarea
                            value={settings.default_description_template}
                            onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => updateSettingLocal('default_description_template', e.target.value)}
                            placeholder="Auto-generated clip..."
                            rows={3}
                          />
                        </div>
                      </div>
                    </div>
                  )}
                </ScrollArea>
              </TabsContent>

              {/* Folders Tab */}
              <TabsContent value="folders" className="flex-1 m-0">
                <ScrollArea className="h-[calc(100vh-250px)]">
                  {folders.length === 0 ? (
                    <div className="flex flex-col items-center justify-center py-16 text-muted-foreground">
                      <FolderOpen className="h-12 w-12 mb-4 opacity-50" />
                      <p>No output folders found</p>
                      <p className="text-xs mt-1">Process some videos first</p>
                    </div>
                  ) : (
                    <div className="p-4 space-y-3">
                      {folders.map((folder) => (
                        <div
                          key={folder.folder_path}
                          className={`border rounded-lg p-4 ${
                            selectedFolder === folder.folder_path ? 'ring-2 ring-primary' : ''
                          }`}
                        >
                          <div className="flex items-start justify-between">
                            <div className="flex-1 cursor-pointer" onClick={() => setSelectedFolder(
                              selectedFolder === folder.folder_path ? null : folder.folder_path
                            )}>
                              <h4 className="font-medium">{folder.folder_name}</h4>
                              <div className="flex items-center gap-2 mt-1 text-sm text-muted-foreground">
                                <Video className="h-4 w-4" />
                                <span>{folder.video_count} video(s)</span>
                                {folder.has_metadata && (
                                  <Badge variant="secondary" className="text-xs">Has Metadata</Badge>
                                )}
                              </div>
                            </div>

                            <Button
                              size="sm"
                              onClick={(e) => {
                                e.stopPropagation()
                                uploadFolder(folder.folder_path)
                              }}
                              disabled={uploadingFolder === folder.folder_path}
                            >
                              {uploadingFolder === folder.folder_path ? (
                                <Loader2 className="h-4 w-4 animate-spin" />
                              ) : (
                                <>
                                  <Play className="h-4 w-4 mr-1" />
                                  Upload All
                                </>
                              )}
                            </Button>
                          </div>

                          {/* Show videos with their metadata when folder is selected */}
                          {selectedFolder === folder.folder_path && (
                            <div className="mt-3 space-y-2 border-t pt-3">
                              {folder.videos.map((video, idx) => (
                                <div key={idx} className="p-2 bg-muted rounded text-sm">
                                  <div className="flex items-center gap-2">
                                    <Video className="h-3 w-3 text-primary" />
                                    <span className="font-medium truncate">{video.name}</span>
                                  </div>
                                  {video.metadata && video.metadata.title && (
                                    <div className="mt-1 pl-5 text-xs space-y-0.5">
                                      <p><strong>Title:</strong> {video.metadata.title}</p>
                                      {video.metadata.description && (
                                        <p className="truncate"><strong>Desc:</strong> {video.metadata.description}</p>
                                      )}
                                      {video.metadata.tags && video.metadata.tags.length > 0 && (
                                        <p><strong>Tags:</strong> {video.metadata.tags.length} tags</p>
                                      )}
                                    </div>
                                  )}
                                  {(!video.metadata || !video.metadata.title) && (
                                    <p className="mt-1 pl-5 text-xs text-muted-foreground italic">
                                      No metadata file (.txt) found
                                    </p>
                                  )}
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </ScrollArea>
              </TabsContent>

              {/* History Tab */}
              <TabsContent value="history" className="flex-1 m-0">
                <ScrollArea className="h-[calc(100vh-250px)]">
                  {history.length === 0 ? (
                    <div className="flex flex-col items-center justify-center py-16 text-muted-foreground">
                      <History className="h-12 w-12 mb-4 opacity-50" />
                      <p>No upload history</p>
                    </div>
                  ) : (
                    <div className="p-4 space-y-2">
                      {history.map((item) => (
                        <div
                          key={item.id}
                          className="flex items-center gap-3 p-3 border rounded-lg"
                        >
                          {item.platform === 'youtube' ? (
                            <Youtube className="h-4 w-4 text-red-500" />
                          ) : (
                            <Facebook className="h-4 w-4 text-blue-600" />
                          )}
                          <div className="flex-1 min-w-0">
                            <p className="font-medium text-sm truncate">{item.title}</p>
                            <p className="text-xs text-muted-foreground">
                              {new Date(item.created_at).toLocaleString()}
                            </p>
                            {item.error && (
                              <p className="text-xs text-red-500">{item.error}</p>
                            )}
                          </div>
                          <Badge
                            variant={item.status === 'completed' ? 'default' : item.status === 'failed' ? 'destructive' : 'secondary'}
                          >
                            {item.status}
                          </Badge>
                        </div>
                      ))}
                    </div>
                  )}
                </ScrollArea>
                {history.length > 0 && (
                  <div className="p-3 border-t">
                    <Button variant="outline" size="sm" onClick={clearHistory}>
                      Clear History
                    </Button>
                  </div>
                )}
              </TabsContent>
            </>
          )}
        </Tabs>
      </div>
    </div>
  )
}
