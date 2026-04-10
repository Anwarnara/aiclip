"""
Auto Clip Maker - FastAPI Backend
Main entry point
"""

import os
import sys
import tracemalloc

# Enable tracemalloc for debugging memory issues
tracemalloc.start()

# Add backend directory to path for imports
BACKEND_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse

from backend.api.routes import video_router, clips_router, settings_router, youtube_router, gallery_router, upload_router
from backend.api.routes.auto_process import router as auto_process_router
from backend.api.routes.queue import router as queue_router
from backend.api.websocket import router as websocket_router
from backend.core.config import settings

# Create FastAPI app
app = FastAPI(
    title="Auto Clip Maker API",
    description="Backend API for Auto Clip Maker - AI-powered video clip extraction",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS + ["*"],  # Allow all in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(video_router)
app.include_router(clips_router)
app.include_router(settings_router)
app.include_router(youtube_router)
app.include_router(auto_process_router)
app.include_router(queue_router)
app.include_router(gallery_router)
app.include_router(upload_router)
app.include_router(websocket_router)


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "2.0.0"}


@app.get("/callback")
async def oauth_callback():
    """OAuth2 callback page - serves HTML that handles the authorization code"""
    callback_html = '''
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>YouTube Authorization</title>
  <style>
    body {
      font-family: system-ui, -apple-system, sans-serif;
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 100vh;
      margin: 0;
      background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
      color: #fff;
    }
    .container {
      text-align: center;
      padding: 2rem;
      background: rgba(255,255,255,0.1);
      border-radius: 12px;
      max-width: 400px;
    }
    h1 { margin-bottom: 1rem; }
    .spinner {
      width: 40px;
      height: 40px;
      border: 3px solid rgba(255,255,255,0.3);
      border-top-color: #fff;
      border-radius: 50%;
      animation: spin 1s linear infinite;
      margin: 1rem auto;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    .success { color: #4ade80; }
    .error { color: #f87171; }
    .message { margin-top: 1rem; font-size: 0.9rem; opacity: 0.8; }
  </style>
</head>
<body>
  <div class="container">
    <h1 id="title">Authorizing...</h1>
    <div class="spinner" id="spinner"></div>
    <p id="status">Processing YouTube authorization</p>
    <p class="message" id="message"></p>
  </div>

  <script>
    async function handleCallback() {
      const urlParams = new URLSearchParams(window.location.search);
      const code = urlParams.get('code');
      const error = urlParams.get('error');

      const titleEl = document.getElementById('title');
      const statusEl = document.getElementById('status');
      const messageEl = document.getElementById('message');
      const spinnerEl = document.getElementById('spinner');

      if (error) {
        titleEl.textContent = 'Authorization Failed';
        titleEl.className = 'error';
        statusEl.textContent = error;
        spinnerEl.style.display = 'none';
        messageEl.textContent = 'You can close this window.';
        return;
      }

      if (!code) {
        titleEl.textContent = 'No Code Received';
        titleEl.className = 'error';
        statusEl.textContent = 'Authorization code not found';
        spinnerEl.style.display = 'none';
        return;
      }

      try {
        statusEl.textContent = 'Exchanging code for tokens...';

        const response = await fetch('/api/upload/youtube/callback', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ code })
        });

        const data = await response.json();

        if (response.ok && data.success) {
          titleEl.textContent = 'Success!';
          titleEl.className = 'success';
          statusEl.textContent = 'Successfully connected to YouTube';
          spinnerEl.style.display = 'none';
          messageEl.textContent = 'This window will close automatically...';

          setTimeout(() => { window.close(); }, 2000);
        } else {
          throw new Error(data.detail || data.error || 'Unknown error');
        }
      } catch (err) {
        titleEl.textContent = 'Error';
        titleEl.className = 'error';
        statusEl.textContent = err.message;
        spinnerEl.style.display = 'none';
        messageEl.textContent = 'Please close this window and try again.';
      }
    }

    handleCallback();
  </script>
</body>
</html>
'''
    return HTMLResponse(content=callback_html)


# Serve frontend static files in production
FRONTEND_DIST = os.path.join(PROJECT_ROOT, "frontend", "dist")
if os.path.exists(FRONTEND_DIST):
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIST, "assets")), name="assets")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))

    @app.get("/{path:path}")
    async def serve_frontend_routes(path: str):
        # Serve index.html for all frontend routes
        file_path = os.path.join(FRONTEND_DIST, path)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
