/**
 * Gallery Panel - File Manager
 * Full file manager functionality: browse, copy, move, rename, delete, create folder
 */

import { useState, useEffect, useCallback } from 'react'
import {
  FolderOpen,
  X,
  Loader2,
  Video,
  Music,
  Trash2,
  Play,
  ExternalLink,
  RefreshCw,
  Folder,
  FileText,
  Image,
  File,
  ChevronRight,
  Home,
  Copy,
  Scissors,
  ClipboardPaste,
  Edit3,
  FolderPlus,
  MoreVertical,
  Check,
  ArrowUp
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Input } from '@/components/ui/input'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from '@/components/ui/dialog'
import { cn } from '@/lib/utils'

interface FileItem {
  name: string
  path: string
  size: number
  size_formatted: string
  created: string
  modified: string
  type: string
  is_dir: boolean
  children_count: number
}

interface GalleryProps {
  isOpen: boolean
  onClose: () => void
  onSelectFile?: (path: string) => void
}

interface ClipboardData {
  path: string
  name: string
  operation: 'copy' | 'cut'
}

export function Gallery({ isOpen, onClose, onSelectFile }: GalleryProps) {
  const [currentPath, setCurrentPath] = useState<string | null>(null)
  const [parentPath, setParentPath] = useState<string | null>(null)
  const [folderName, setFolderName] = useState<string>('')
  const [items, setItems] = useState<FileItem[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [selectedItems, setSelectedItems] = useState<Set<string>>(new Set())
  const [clipboard, setClipboard] = useState<ClipboardData | null>(null)

  // Dialogs
  const [renameDialog, setRenameDialog] = useState<{ open: boolean; item: FileItem | null }>({ open: false, item: null })
  const [newFolderDialog, setNewFolderDialog] = useState(false)
  const [newName, setNewName] = useState('')
  const [textEditDialog, setTextEditDialog] = useState<{ open: boolean; path: string; content: string }>({ open: false, path: '', content: '' })

  const loadDirectory = useCallback(async (path: string | null = null) => {
    setIsLoading(true)
    try {
      const url = path ? `/api/gallery/browse?path=${encodeURIComponent(path)}&show_all=true` : '/api/gallery/browse'
      const res = await fetch(url)
      const data = await res.json()

      if (res.ok) {
        setCurrentPath(data.current_path)
        setParentPath(data.parent_path)
        setFolderName(data.folder_name || '')
        setItems(data.items || [])
        setSelectedItems(new Set())
      } else {
        console.error('Error:', data.detail)
      }
    } catch (err) {
      console.error('Error loading directory:', err)
    } finally {
      setIsLoading(false)
    }
  }, [])

  useEffect(() => {
    if (isOpen) {
      loadDirectory(null)
    }
  }, [isOpen, loadDirectory])

  const navigateTo = (path: string | null) => {
    loadDirectory(path)
  }

  const goUp = () => {
    if (parentPath) {
      navigateTo(parentPath)
    } else {
      navigateTo(null)
    }
  }

  const handleItemClick = (item: FileItem) => {
    if (item.is_dir) {
      navigateTo(item.path)
    }
  }

  const handleItemDoubleClick = (item: FileItem) => {
    if (!item.is_dir && onSelectFile) {
      onSelectFile(item.path)
      onClose()
    } else if (!item.is_dir && item.type === 'text') {
      openTextEditor(item.path)
    } else if (!item.is_dir) {
      openFile(item.path)
    }
  }

  const toggleSelect = (path: string, e: React.MouseEvent) => {
    e.stopPropagation()
    const newSelected = new Set(selectedItems)
    if (newSelected.has(path)) {
      newSelected.delete(path)
    } else {
      newSelected.add(path)
    }
    setSelectedItems(newSelected)
  }

  const deleteItem = async (path: string) => {
    if (!confirm('Are you sure you want to delete this item?')) return

    try {
      await fetch(`/api/gallery/file?path=${encodeURIComponent(path)}`, { method: 'DELETE' })
      loadDirectory(currentPath)
    } catch (err) {
      console.error('Error deleting:', err)
    }
  }

  const renameItem = async () => {
    if (!renameDialog.item || !newName.trim()) return

    try {
      const res = await fetch('/api/gallery/rename', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          old_path: renameDialog.item.path,
          new_name: newName.trim()
        })
      })

      if (res.ok) {
        loadDirectory(currentPath)
        setRenameDialog({ open: false, item: null })
        setNewName('')
      } else {
        const data = await res.json()
        alert(data.detail || 'Rename failed')
      }
    } catch (err) {
      console.error('Error renaming:', err)
    }
  }

  const createFolder = async () => {
    if (!newName.trim() || !currentPath) return

    try {
      const res = await fetch('/api/gallery/create-folder', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          parent_path: currentPath,
          folder_name: newName.trim()
        })
      })

      if (res.ok) {
        loadDirectory(currentPath)
        setNewFolderDialog(false)
        setNewName('')
      } else {
        const data = await res.json()
        alert(data.detail || 'Create folder failed')
      }
    } catch (err) {
      console.error('Error creating folder:', err)
    }
  }

  const copyItem = (path: string, name: string) => {
    setClipboard({ path, name, operation: 'copy' })
  }

  const cutItem = (path: string, name: string) => {
    setClipboard({ path, name, operation: 'cut' })
  }

  const pasteItem = async () => {
    if (!clipboard || !currentPath) return

    try {
      const endpoint = clipboard.operation === 'copy' ? '/api/gallery/copy' : '/api/gallery/move'
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          source_path: clipboard.path,
          dest_folder: currentPath
        })
      })

      if (res.ok) {
        loadDirectory(currentPath)
        if (clipboard.operation === 'cut') {
          setClipboard(null)
        }
      } else {
        const data = await res.json()
        alert(data.detail || 'Paste failed')
      }
    } catch (err) {
      console.error('Error pasting:', err)
    }
  }

  const openFolder = async (path: string) => {
    try {
      await fetch('/api/gallery/open-folder', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path })
      })
    } catch (err) {
      console.error('Error opening folder:', err)
    }
  }

  const openFile = async (path: string) => {
    try {
      await fetch('/api/gallery/open-file', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path })
      })
    } catch (err) {
      console.error('Error opening file:', err)
    }
  }

  const openTextEditor = async (path: string) => {
    try {
      const res = await fetch(`/api/gallery/file-content?path=${encodeURIComponent(path)}`)
      const data = await res.json()
      if (data.success) {
        setTextEditDialog({ open: true, path, content: data.content })
      }
    } catch (err) {
      console.error('Error reading file:', err)
    }
  }

  const saveTextFile = async () => {
    try {
      const res = await fetch(`/api/gallery/file-content?path=${encodeURIComponent(textEditDialog.path)}&content=${encodeURIComponent(textEditDialog.content)}`, {
        method: 'POST'
      })
      if (res.ok) {
        setTextEditDialog({ open: false, path: '', content: '' })
      }
    } catch (err) {
      console.error('Error saving file:', err)
    }
  }

  const getFileIcon = (item: FileItem) => {
    if (item.is_dir) return <Folder className="h-5 w-5 text-yellow-500" />
    switch (item.type) {
      case 'video': return <Video className="h-5 w-5 text-blue-500" />
      case 'audio': return <Music className="h-5 w-5 text-purple-500" />
      case 'image': return <Image className="h-5 w-5 text-green-500" />
      case 'text': return <FileText className="h-5 w-5 text-gray-500" />
      default: return <File className="h-5 w-5 text-gray-400" />
    }
  }

  const formatDate = (isoDate: string) => {
    const date = new Date(isoDate)
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 bg-background/80 backdrop-blur-sm">
      <div className="fixed inset-4 z-50 bg-card border rounded-lg shadow-lg flex flex-col max-w-4xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-3 border-b">
          <div className="flex items-center gap-2">
            <FolderOpen className="h-5 w-5 text-primary" />
            <h2 className="text-lg font-semibold">File Manager</h2>
          </div>
          <div className="flex items-center gap-1">
            <Button variant="ghost" size="icon" onClick={() => loadDirectory(currentPath)} title="Refresh">
              <RefreshCw className="h-4 w-4" />
            </Button>
            <Button variant="ghost" size="icon" onClick={onClose}>
              <X className="h-4 w-4" />
            </Button>
          </div>
        </div>

        {/* Toolbar */}
        <div className="flex items-center gap-2 p-2 border-b bg-muted/30">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => navigateTo(null)}
            title="Home"
          >
            <Home className="h-4 w-4" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={goUp}
            disabled={!currentPath}
            title="Go Up"
          >
            <ArrowUp className="h-4 w-4" />
          </Button>

          <div className="h-4 w-px bg-border mx-1" />

          {currentPath && (
            <>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => { setNewFolderDialog(true); setNewName('') }}
                title="New Folder"
              >
                <FolderPlus className="h-4 w-4 mr-1" />
                New Folder
              </Button>

              <Button
                variant="ghost"
                size="sm"
                onClick={pasteItem}
                disabled={!clipboard}
                title={clipboard ? `Paste "${clipboard.name}"` : 'Paste'}
              >
                <ClipboardPaste className="h-4 w-4 mr-1" />
                Paste
                {clipboard && (
                  <Badge variant="secondary" className="ml-1 text-xs">
                    {clipboard.operation === 'cut' ? 'Cut' : 'Copy'}
                  </Badge>
                )}
              </Button>

              <Button
                variant="ghost"
                size="sm"
                onClick={() => openFolder(currentPath)}
                title="Open in Explorer"
              >
                <ExternalLink className="h-4 w-4 mr-1" />
                Explorer
              </Button>
            </>
          )}
        </div>

        {/* Breadcrumb */}
        <div className="flex items-center gap-1 px-3 py-2 text-sm bg-muted/20 border-b overflow-x-auto">
          <Button
            variant="ghost"
            size="sm"
            className="h-6 px-2"
            onClick={() => navigateTo(null)}
          >
            <Home className="h-3 w-3" />
          </Button>
          {currentPath && (
            <>
              <ChevronRight className="h-3 w-3 text-muted-foreground shrink-0" />
              <span className="font-medium truncate">{folderName || currentPath}</span>
            </>
          )}
        </div>

        {/* File List */}
        <ScrollArea className="flex-1">
          {isLoading ? (
            <div className="flex items-center justify-center py-16">
              <Loader2 className="h-8 w-8 animate-spin" />
            </div>
          ) : items.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-16 text-muted-foreground">
              <Folder className="h-12 w-12 mb-4 opacity-50" />
              <p>Empty folder</p>
            </div>
          ) : (
            <div className="p-2">
              {items.map((item) => (
                <div
                  key={item.path}
                  className={cn(
                    "flex items-center gap-3 p-2 rounded-lg hover:bg-muted/50 cursor-pointer transition-colors group",
                    selectedItems.has(item.path) && "bg-primary/10 ring-1 ring-primary/30"
                  )}
                  onClick={() => handleItemClick(item)}
                  onDoubleClick={() => handleItemDoubleClick(item)}
                >
                  {/* Select checkbox */}
                  <div
                    className={cn(
                      "w-5 h-5 rounded border flex items-center justify-center cursor-pointer shrink-0",
                      selectedItems.has(item.path)
                        ? "bg-primary border-primary text-primary-foreground"
                        : "border-muted-foreground/30 opacity-0 group-hover:opacity-100"
                    )}
                    onClick={(e) => toggleSelect(item.path, e)}
                  >
                    {selectedItems.has(item.path) && <Check className="h-3 w-3" />}
                  </div>

                  {/* Icon */}
                  <div className="shrink-0">
                    {getFileIcon(item)}
                  </div>

                  {/* Info */}
                  <div className="flex-1 min-w-0">
                    <p className="font-medium text-sm truncate">{item.name}</p>
                    <div className="flex items-center gap-2 text-xs text-muted-foreground">
                      <span>{item.size_formatted}</span>
                      <span>•</span>
                      <span>{formatDate(item.modified)}</span>
                    </div>
                  </div>

                  {/* Actions */}
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild onClick={(e: React.MouseEvent) => e.stopPropagation()}>
                      <Button variant="ghost" size="icon" className="h-8 w-8 opacity-0 group-hover:opacity-100">
                        <MoreVertical className="h-4 w-4" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      {!item.is_dir && onSelectFile && (
                        <DropdownMenuItem onClick={() => { onSelectFile(item.path); onClose() }}>
                          <Play className="h-4 w-4 mr-2" />
                          Use this file
                        </DropdownMenuItem>
                      )}
                      {!item.is_dir && (
                        <DropdownMenuItem onClick={() => openFile(item.path)}>
                          <ExternalLink className="h-4 w-4 mr-2" />
                          Open
                        </DropdownMenuItem>
                      )}
                      {item.type === 'text' && (
                        <DropdownMenuItem onClick={() => openTextEditor(item.path)}>
                          <Edit3 className="h-4 w-4 mr-2" />
                          Edit
                        </DropdownMenuItem>
                      )}
                      <DropdownMenuSeparator />
                      <DropdownMenuItem onClick={() => copyItem(item.path, item.name)}>
                        <Copy className="h-4 w-4 mr-2" />
                        Copy
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={() => cutItem(item.path, item.name)}>
                        <Scissors className="h-4 w-4 mr-2" />
                        Cut
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={() => { setRenameDialog({ open: true, item }); setNewName(item.name) }}>
                        <Edit3 className="h-4 w-4 mr-2" />
                        Rename
                      </DropdownMenuItem>
                      <DropdownMenuSeparator />
                      <DropdownMenuItem
                        className="text-red-500 focus:text-red-500"
                        onClick={() => deleteItem(item.path)}
                      >
                        <Trash2 className="h-4 w-4 mr-2" />
                        Delete
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
              ))}
            </div>
          )}
        </ScrollArea>

        {/* Footer */}
        <div className="flex items-center justify-between p-2 border-t text-xs text-muted-foreground">
          <span>{items.length} items</span>
          {selectedItems.size > 0 && (
            <span>{selectedItems.size} selected</span>
          )}
        </div>

        {/* Rename Dialog */}
        <Dialog open={renameDialog.open} onOpenChange={(open) => setRenameDialog({ open, item: null })}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Rename</DialogTitle>
            </DialogHeader>
            <Input
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              placeholder="Enter new name"
              onKeyDown={(e) => e.key === 'Enter' && renameItem()}
            />
            <DialogFooter>
              <Button variant="outline" onClick={() => setRenameDialog({ open: false, item: null })}>
                Cancel
              </Button>
              <Button onClick={renameItem}>Rename</Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* New Folder Dialog */}
        <Dialog open={newFolderDialog} onOpenChange={setNewFolderDialog}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>New Folder</DialogTitle>
            </DialogHeader>
            <Input
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              placeholder="Folder name"
              onKeyDown={(e) => e.key === 'Enter' && createFolder()}
            />
            <DialogFooter>
              <Button variant="outline" onClick={() => setNewFolderDialog(false)}>
                Cancel
              </Button>
              <Button onClick={createFolder}>Create</Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* Text Edit Dialog */}
        <Dialog open={textEditDialog.open} onOpenChange={(open) => !open && setTextEditDialog({ open: false, path: '', content: '' })}>
          <DialogContent className="max-w-2xl max-h-[80vh]">
            <DialogHeader>
              <DialogTitle>Edit: {textEditDialog.path.split(/[/\\]/).pop()}</DialogTitle>
            </DialogHeader>
            <textarea
              className="w-full h-[400px] p-3 font-mono text-sm border rounded-md bg-muted resize-none"
              value={textEditDialog.content}
              onChange={(e) => setTextEditDialog({ ...textEditDialog, content: e.target.value })}
            />
            <DialogFooter>
              <Button variant="outline" onClick={() => setTextEditDialog({ open: false, path: '', content: '' })}>
                Cancel
              </Button>
              <Button onClick={saveTextFile}>Save</Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>
    </div>
  )
}
