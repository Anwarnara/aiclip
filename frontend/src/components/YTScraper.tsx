/**
 * YouTube Scraper Panel
 * Search and download YouTube videos
 */

import { useState, useEffect } from 'react'
import {
  Youtube,
  Search,
  Download,
  Clock,
  Eye,
  User,
  Calendar,
  Filter,
  X,
  Loader2,
  CheckCircle2,
  AlertCircle,
  ThumbsUp,
  MessageCircle,
  Sparkles,
  Star,
  Globe,
  CheckCheck,
  EyeOff,
  ListPlus
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Checkbox } from '@/components/ui/checkbox'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'

interface Video {
  id: string
  title: string
  description?: string
  channel: string
  channel_id: string
  thumbnail: string
  duration: string
  duration_seconds: number
  views: string
  likes: string
  comments: string
  view_count: number
  like_count: number
  comment_count: number
  engagement_rate: number
  published: string
  url: string
}

interface Format {
  format_id: string
  resolution: string
  ext: string
  filesize: string
  has_audio: boolean
  height: number
}

interface Category {
  id: string
  name: string
}

interface Region {
  code: string
  name: string
}

interface Recommendation {
  index: number
  reason: string
  score: number
  clip_potential: string
}

interface YTScraperProps {
  isOpen: boolean
  onClose: () => void
  onVideoSelect?: (videoPath: string) => void
}

export function YTScraper({ isOpen, onClose, onVideoSelect }: YTScraperProps) {
  const [query, setQuery] = useState('')
  const [videos, setVideos] = useState<Video[]>([])
  const [isSearching, setIsSearching] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Filters
  const [durationFilter, setDurationFilter] = useState('any')
  const [orderBy, setOrderBy] = useState('relevance')
  const [channelFilter, setChannelFilter] = useState('')
  const [categoryFilter, setCategoryFilter] = useState('0')
  const [categories, setCategories] = useState<Category[]>([])
  const [regionFilter, setRegionFilter] = useState('ID')
  const [regions, setRegions] = useState<Region[]>([])
  const [hideProcessed, setHideProcessed] = useState(false)
  const [processedIds, setProcessedIds] = useState<string[]>([])

  // AI Recommendations
  const [recommendations, setRecommendations] = useState<Recommendation[]>([])
  const [aiSummary, setAiSummary] = useState('')
  const [isLoadingRecommendations, setIsLoadingRecommendations] = useState(false)

  // Download dialog
  const [selectedVideo, setSelectedVideo] = useState<Video | null>(null)
  const [formats, setFormats] = useState<Format[]>([])
  const [selectedFormat, setSelectedFormat] = useState('best')
  const [isLoadingFormats, setIsLoadingFormats] = useState(false)
  const [isDownloading, setIsDownloading] = useState(false)
  const [isAddingToQueue, setIsAddingToQueue] = useState(false)
  const [downloadStatus, setDownloadStatus] = useState<'idle' | 'success' | 'error' | 'queued'>('idle')

  // API status
  const [apiConfigured, setApiConfigured] = useState<boolean | null>(null)

  // Check API status and load categories on mount
  useEffect(() => {
    if (isOpen) {
      checkApiStatus()
      loadCategories()
      loadRegions()
      loadProcessedIds()
    }
  }, [isOpen])

  const checkApiStatus = async () => {
    try {
      const response = await fetch('/api/youtube/status')
      const data = await response.json()
      setApiConfigured(data.configured)
    } catch {
      setApiConfigured(false)
    }
  }

  const loadCategories = async () => {
    try {
      const response = await fetch('/api/youtube/categories')
      const data = await response.json()
      setCategories(data.categories || [])
    } catch {
      setCategories([{ id: '0', name: 'All' }])
    }
  }

  const loadRegions = async () => {
    try {
      const response = await fetch('/api/youtube/regions')
      const data = await response.json()
      setRegions(data.regions || [])
    } catch {
      setRegions([{ code: 'ID', name: 'Indonesia' }])
    }
  }

  const loadProcessedIds = async () => {
    try {
      const response = await fetch('/api/auto-process/processed/ids')
      const data = await response.json()
      setProcessedIds(data.ids || [])
    } catch {
      setProcessedIds([])
    }
  }

  const isVideoProcessed = (videoId: string): boolean => {
    return processedIds.includes(videoId)
  }

  const handleSearch = async () => {
    if (!query.trim()) return

    setIsSearching(true)
    setError(null)
    setVideos([])
    setRecommendations([])
    setAiSummary('')

    try {
      const response = await fetch('/api/youtube/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query.trim(),
          max_results: 20,
          duration_filter: durationFilter,
          order: orderBy,
          channel_name: channelFilter.trim() || null,
          category_id: categoryFilter !== '0' ? categoryFilter : null,
          region_code: regionFilter
        })
      })

      const data = await response.json()

      if (data.error) {
        setError(data.error)
      } else {
        const foundVideos = data.videos || []
        setVideos(foundVideos)

        // Get AI recommendations if we have videos
        if (foundVideos.length > 0) {
          getAiRecommendations(foundVideos)
        }
      }
    } catch (err) {
      setError('Failed to search. Check your connection.')
    } finally {
      setIsSearching(false)
    }
  }

  const getAiRecommendations = async (videoList: Video[]) => {
    setIsLoadingRecommendations(true)
    try {
      const response = await fetch('/api/youtube/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          videos: videoList,
          purpose: 'viral clips'
        })
      })
      const data = await response.json()
      setRecommendations(data.recommendations || [])
      setAiSummary(data.summary || '')
    } catch {
      // Silently fail - recommendations are optional
    } finally {
      setIsLoadingRecommendations(false)
    }
  }

  const getRecommendationForVideo = (index: number): Recommendation | undefined => {
    return recommendations.find(r => r.index === index + 1)
  }

  const handleVideoClick = async (video: Video) => {
    setSelectedVideo(video)
    setFormats([])
    setSelectedFormat('best')
    setIsLoadingFormats(true)
    setDownloadStatus('idle')

    try {
      const response = await fetch(`/api/youtube/formats?url=${encodeURIComponent(video.url)}`)
      const data = await response.json()
      setFormats(data.formats || [])
    } catch {
      setFormats([{ format_id: 'best', resolution: 'Best Quality', ext: 'mp4', filesize: 'Auto', has_audio: true, height: 9999 }])
    } finally {
      setIsLoadingFormats(false)
    }
  }

  const handleDownload = async () => {
    if (!selectedVideo) return

    setIsDownloading(true)
    setDownloadStatus('idle')

    try {
      const response = await fetch('/api/youtube/download', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          video_url: selectedVideo.url,
          format_id: selectedFormat
        })
      })

      const data = await response.json()

      if (response.ok && data.success) {
        setDownloadStatus('success')
        if (onVideoSelect && data.path) {
          // Wait a bit then close and use the video
          setTimeout(() => {
            onVideoSelect(data.path)
            setSelectedVideo(null)
            onClose()
          }, 1500)
        }
      } else {
        setDownloadStatus('error')
        setError(data.detail || 'Download failed')
      }
    } catch {
      setDownloadStatus('error')
      setError('Download failed')
    } finally {
      setIsDownloading(false)
    }
  }

  const handleAddToQueue = async () => {
    if (!selectedVideo) return

    setIsAddingToQueue(true)
    setDownloadStatus('idle')

    try {
      // Get resolution from selected format
      let resolution = '1080'
      if (selectedFormat !== 'best') {
        const fmt = formats.find(f => f.format_id === selectedFormat)
        if (fmt && fmt.height) {
          resolution = fmt.height.toString()
        }
      }

      const response = await fetch('/api/queue/add', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          video_id: selectedVideo.id,
          video_url: selectedVideo.url,
          title: selectedVideo.title,
          channel: selectedVideo.channel,
          thumbnail: selectedVideo.thumbnail,
          resolution: resolution
        })
      })

      if (response.ok) {
        setDownloadStatus('queued')
        setTimeout(() => {
          setSelectedVideo(null)
        }, 1500)
      } else {
        setDownloadStatus('error')
        setError('Failed to add to queue')
      }
    } catch {
      setDownloadStatus('error')
      setError('Failed to add to queue')
    } finally {
      setIsAddingToQueue(false)
    }
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 bg-background/80 backdrop-blur-sm">
      <div className="fixed inset-4 z-50 bg-card border rounded-lg shadow-lg flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b">
          <div className="flex items-center gap-2">
            <Youtube className="h-5 w-5 text-red-500" />
            <h2 className="text-lg font-semibold">YT Scraper</h2>
            {apiConfigured === false && (
              <Badge variant="destructive" className="text-xs">API Not Configured</Badge>
            )}
          </div>
          <Button variant="ghost" size="icon" onClick={onClose}>
            <X className="h-4 w-4" />
          </Button>
        </div>

        {/* Search Bar */}
        <div className="p-4 border-b space-y-3">
          <div className="flex gap-2">
            <Input
              placeholder="Search YouTube videos..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
              className="flex-1"
            />
            <Button onClick={handleSearch} disabled={isSearching || !query.trim()}>
              {isSearching ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Search className="h-4 w-4" />
              )}
              <span className="ml-2">Search</span>
            </Button>
          </div>

          {/* Filters */}
          <div className="flex gap-2 flex-wrap">
            <Select value={durationFilter} onValueChange={setDurationFilter}>
              <SelectTrigger className="w-[140px] h-8 text-xs">
                <Clock className="h-3 w-3 mr-1" />
                <SelectValue placeholder="Duration" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="any">Any Duration</SelectItem>
                <SelectItem value="short">&lt; 4 minutes</SelectItem>
                <SelectItem value="medium">4-20 minutes</SelectItem>
                <SelectItem value="long">&gt; 20 minutes</SelectItem>
              </SelectContent>
            </Select>

            <Select value={categoryFilter} onValueChange={setCategoryFilter}>
              <SelectTrigger className="w-[160px] h-8 text-xs">
                <Filter className="h-3 w-3 mr-1" />
                <SelectValue placeholder="Genre" />
              </SelectTrigger>
              <SelectContent>
                {categories.map((cat) => (
                  <SelectItem key={cat.id} value={cat.id}>
                    {cat.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            <Select value={orderBy} onValueChange={setOrderBy}>
              <SelectTrigger className="w-[130px] h-8 text-xs">
                <SelectValue placeholder="Sort by" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="relevance">Relevance</SelectItem>
                <SelectItem value="date">Upload Date</SelectItem>
                <SelectItem value="viewCount">View Count</SelectItem>
                <SelectItem value="rating">Rating</SelectItem>
              </SelectContent>
            </Select>

            <Select value={regionFilter} onValueChange={setRegionFilter}>
              <SelectTrigger className="w-[150px] h-8 text-xs">
                <Globe className="h-3 w-3 mr-1" />
                <SelectValue placeholder="Region" />
              </SelectTrigger>
              <SelectContent>
                {regions.map((region) => (
                  <SelectItem key={region.code} value={region.code}>
                    {region.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            <Input
              placeholder="Channel name (optional)"
              value={channelFilter}
              onChange={(e) => setChannelFilter(e.target.value)}
              className="w-[180px] h-8 text-xs"
            />

            {/* Hide Processed Toggle */}
            <div className="flex items-center gap-2 h-8 px-2 border rounded-md">
              <Checkbox
                id="hideProcessed"
                checked={hideProcessed}
                onCheckedChange={(checked) => setHideProcessed(checked === true)}
              />
              <label
                htmlFor="hideProcessed"
                className="text-xs cursor-pointer flex items-center gap-1"
              >
                <EyeOff className="h-3 w-3" />
                Hide Processed
              </label>
            </div>
          </div>
        </div>

        {/* Results */}
        <ScrollArea className="flex-1 p-4">
          {error && (
            <div className="text-center py-8 text-red-500">
              <AlertCircle className="h-8 w-8 mx-auto mb-2" />
              <p>{error}</p>
            </div>
          )}

          {!error && videos.length === 0 && !isSearching && (
            <div className="text-center py-16 text-muted-foreground">
              <Youtube className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <p>Search for YouTube videos to get started</p>
              <p className="text-xs mt-2">You can filter by duration, genre, or channel</p>
            </div>
          )}

          {isSearching && (
            <div className="text-center py-16">
              <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4" />
              <p className="text-muted-foreground">Searching...</p>
            </div>
          )}

          {/* AI Recommendations Banner */}
          {videos.length > 0 && (
            <div className="mb-4 p-3 bg-gradient-to-r from-purple-500/10 to-pink-500/10 rounded-lg border border-purple-500/20">
              <div className="flex items-center gap-2 mb-2">
                <Sparkles className="h-4 w-4 text-purple-500" />
                <span className="font-medium text-sm">AI Recommendations</span>
                {isLoadingRecommendations && (
                  <Loader2 className="h-3 w-3 animate-spin text-purple-500" />
                )}
              </div>
              {aiSummary && (
                <p className="text-xs text-muted-foreground mb-2">{aiSummary}</p>
              )}
              {recommendations.length > 0 && (
                <div className="flex flex-wrap gap-1">
                  {recommendations.map((rec) => (
                    <Badge
                      key={rec.index}
                      variant="secondary"
                      className="text-xs bg-purple-500/20 hover:bg-purple-500/30 cursor-pointer"
                      onClick={() => {
                        const video = videos[rec.index - 1]
                        if (video) handleVideoClick(video)
                      }}
                    >
                      <Star className="h-3 w-3 mr-1 text-yellow-500" />
                      #{rec.index} - Score: {rec.score}
                    </Badge>
                  ))}
                </div>
              )}
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {videos
              .filter(video => !hideProcessed || !isVideoProcessed(video.id))
              .map((video, index) => {
              const rec = getRecommendationForVideo(index)
              const processed = isVideoProcessed(video.id)
              return (
                <div
                  key={video.id}
                  className={`border rounded-lg overflow-hidden cursor-pointer transition-all ${
                    processed ? 'opacity-60 border-green-500/50' :
                    rec ? 'border-purple-500 ring-1 ring-purple-500/50' : 'hover:border-primary'
                  }`}
                  onClick={() => handleVideoClick(video)}
                >
                  <div className="relative">
                    <img
                      src={video.thumbnail}
                      alt={video.title}
                      className="w-full aspect-video object-cover"
                    />
                    <Badge className="absolute bottom-1 right-1 bg-black/80 text-white text-xs">
                      {video.duration}
                    </Badge>
                    {/* Processed Badge */}
                    {processed && (
                      <Badge className="absolute top-1 right-1 bg-green-500 text-white text-xs">
                        <CheckCheck className="h-3 w-3 mr-1" />
                        Processed
                      </Badge>
                    )}
                    {rec && (
                      <Badge className="absolute top-1 left-1 bg-purple-500 text-white text-xs">
                        <Star className="h-3 w-3 mr-1" />
                        AI Pick ({rec.score})
                      </Badge>
                    )}
                  </div>
                  <div className="p-3">
                    <h3 className="font-medium text-sm line-clamp-2 mb-2">{video.title}</h3>
                    <div className="flex items-center gap-2 text-xs text-muted-foreground">
                      <User className="h-3 w-3" />
                      <span className="truncate">{video.channel}</span>
                    </div>

                    {/* Stats Row */}
                    <div className="flex items-center gap-2 mt-2 text-xs text-muted-foreground flex-wrap">
                      <span className="flex items-center gap-1">
                        <Eye className="h-3 w-3" />
                        {video.views}
                      </span>
                      <span className="flex items-center gap-1">
                        <ThumbsUp className="h-3 w-3" />
                        {video.likes || '0'}
                      </span>
                      <span className="flex items-center gap-1">
                        <MessageCircle className="h-3 w-3" />
                        {video.comments || '0'}
                      </span>
                    </div>

                    {/* Engagement & Date */}
                    <div className="flex items-center justify-between mt-1 text-xs text-muted-foreground">
                      <span className="flex items-center gap-1">
                        <Calendar className="h-3 w-3" />
                        {video.published}
                      </span>
                      {video.engagement_rate > 0 && (
                        <span className={`${video.engagement_rate > 5 ? 'text-green-500' : ''}`}>
                          {video.engagement_rate.toFixed(1)}% eng
                        </span>
                      )}
                    </div>

                    {/* AI Reason */}
                    {rec && (
                      <div className="mt-2 p-2 bg-purple-500/10 rounded text-xs">
                        <span className="text-purple-400">{rec.reason}</span>
                      </div>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        </ScrollArea>

        {/* Download Dialog */}
        <Dialog open={!!selectedVideo} onOpenChange={() => setSelectedVideo(null)}>
          <DialogContent className="max-w-md">
            <DialogHeader>
              <DialogTitle className="flex items-center gap-2">
                <Download className="h-4 w-4" />
                Download Video
              </DialogTitle>
            </DialogHeader>

            {selectedVideo && (
              <div className="space-y-4">
                <div className="flex gap-3">
                  <img
                    src={selectedVideo.thumbnail}
                    alt={selectedVideo.title}
                    className="w-32 aspect-video object-cover rounded"
                  />
                  <div className="flex-1 min-w-0">
                    <h4 className="font-medium text-sm line-clamp-2">{selectedVideo.title}</h4>
                    <p className="text-xs text-muted-foreground mt-1">{selectedVideo.channel}</p>
                    <p className="text-xs text-muted-foreground">{selectedVideo.duration}</p>
                  </div>
                </div>

                {isLoadingFormats ? (
                  <div className="text-center py-4">
                    <Loader2 className="h-6 w-6 animate-spin mx-auto" />
                    <p className="text-sm text-muted-foreground mt-2">Loading formats...</p>
                  </div>
                ) : (
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Select Quality</label>
                    <Select value={selectedFormat} onValueChange={setSelectedFormat}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select resolution" />
                      </SelectTrigger>
                      <SelectContent>
                        {formats.map((fmt) => (
                          <SelectItem key={fmt.format_id} value={fmt.format_id}>
                            <span className="flex items-center gap-2">
                              {fmt.resolution}
                              <span className="text-muted-foreground text-xs">
                                ({fmt.ext} - {fmt.filesize})
                              </span>
                            </span>
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                )}

                {downloadStatus === 'success' && (
                  <div className="flex items-center gap-2 text-green-500 text-sm">
                    <CheckCircle2 className="h-4 w-4" />
                    Download complete! Loading video...
                  </div>
                )}

                {downloadStatus === 'queued' && (
                  <div className="flex items-center gap-2 text-blue-500 text-sm">
                    <CheckCircle2 className="h-4 w-4" />
                    Added to queue! Video will be downloaded and processed.
                  </div>
                )}

                {downloadStatus === 'error' && (
                  <div className="flex items-center gap-2 text-red-500 text-sm">
                    <AlertCircle className="h-4 w-4" />
                    {error || 'Download failed'}
                  </div>
                )}

                <div className="flex gap-2">
                  <Button
                    onClick={handleAddToQueue}
                    disabled={isDownloading || isLoadingFormats || isAddingToQueue}
                    variant="outline"
                    className="flex-1"
                  >
                    {isAddingToQueue ? (
                      <>
                        <Loader2 className="h-4 w-4 animate-spin mr-2" />
                        Adding...
                      </>
                    ) : (
                      <>
                        <ListPlus className="h-4 w-4 mr-2" />
                        Add to Queue
                      </>
                    )}
                  </Button>
                  <Button
                    onClick={handleDownload}
                    disabled={isDownloading || isLoadingFormats || isAddingToQueue}
                    className="flex-1"
                  >
                    {isDownloading ? (
                      <>
                        <Loader2 className="h-4 w-4 animate-spin mr-2" />
                        Downloading...
                      </>
                    ) : (
                      <>
                        <Download className="h-4 w-4 mr-2" />
                        Download Now
                      </>
                    )}
                  </Button>
                </div>
              </div>
            )}
          </DialogContent>
        </Dialog>
      </div>
    </div>
  )
}
