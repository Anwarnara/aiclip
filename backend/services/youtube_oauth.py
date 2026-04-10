"""
YouTube OAuth2 Service
Handles authentication and video upload to YouTube
"""

import os
import json
import pickle
from datetime import datetime, timedelta
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

# Paths
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data"
)
CLIENT_ID_PATH = os.path.join(DATA_DIR, "client_id.txt")
CLIENT_SECRET_PATH = os.path.join(DATA_DIR, "client_secret.txt")
TOKEN_PATH = os.path.join(DATA_DIR, "youtube_token.pickle")
CREDENTIALS_PATH = os.path.join(DATA_DIR, "youtube_credentials.json")

# YouTube API scopes
SCOPES = [
    'https://www.googleapis.com/auth/youtube.upload',
    'https://www.googleapis.com/auth/youtube.readonly'
]

# Redirect URI for desktop app (localhost)
# This should match the redirect URI configured in Google Cloud Console
REDIRECT_URI = "http://localhost:8000/callback"


def load_client_credentials() -> tuple:
    """Load client ID and secret from files"""
    client_id = ""
    client_secret = ""

    try:
        if os.path.exists(CLIENT_ID_PATH):
            with open(CLIENT_ID_PATH, 'r') as f:
                client_id = f.read().strip()
    except Exception:
        pass

    try:
        if os.path.exists(CLIENT_SECRET_PATH):
            with open(CLIENT_SECRET_PATH, 'r') as f:
                client_secret = f.read().strip()
    except Exception:
        pass

    return client_id, client_secret


def create_client_config() -> dict:
    """Create OAuth2 client config from files"""
    client_id, client_secret = load_client_credentials()

    if not client_id or not client_secret:
        return None

    return {
        "installed": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [REDIRECT_URI]
        }
    }


def get_credentials() -> Credentials:
    """Get stored credentials or None if not authenticated"""
    creds = None

    # Try to load from pickle file
    if os.path.exists(TOKEN_PATH):
        try:
            with open(TOKEN_PATH, 'rb') as token:
                creds = pickle.load(token)
        except Exception:
            pass

    # Check if credentials are valid
    if creds and creds.valid:
        return creds

    # Try to refresh if expired
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            save_credentials(creds)
            return creds
        except Exception:
            # Refresh failed, need to re-authenticate
            return None

    return None


def save_credentials(creds: Credentials):
    """Save credentials to pickle file"""
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(TOKEN_PATH, 'wb') as token:
            pickle.dump(creds, token)
    except Exception as e:
        print(f"Error saving credentials: {e}")


def get_auth_url() -> tuple:
    """Generate OAuth2 authorization URL"""
    client_config = create_client_config()

    if not client_config:
        return None, "Client credentials not configured. Please add client_id.txt and client_secret.txt to /data folder."

    try:
        flow = Flow.from_client_config(
            client_config,
            scopes=SCOPES,
            redirect_uri=REDIRECT_URI
        )

        auth_url, state = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true',
            prompt='consent'
        )

        # Save flow state for callback
        flow_data = {
            "state": state,
            "client_config": client_config
        }
        with open(os.path.join(DATA_DIR, "oauth_flow.json"), 'w') as f:
            json.dump(flow_data, f)

        return auth_url, None
    except Exception as e:
        return None, str(e)


def handle_callback(code: str) -> tuple:
    """Handle OAuth2 callback and exchange code for tokens"""
    try:
        # Load flow data
        flow_path = os.path.join(DATA_DIR, "oauth_flow.json")
        if not os.path.exists(flow_path):
            return False, "OAuth flow not initialized. Please start authorization again."

        with open(flow_path, 'r') as f:
            flow_data = json.load(f)

        flow = Flow.from_client_config(
            flow_data["client_config"],
            scopes=SCOPES,
            redirect_uri=REDIRECT_URI
        )

        # Exchange code for tokens
        flow.fetch_token(code=code)
        creds = flow.credentials

        # Save credentials
        save_credentials(creds)

        # Clean up flow file
        os.remove(flow_path)

        return True, "Successfully authenticated with YouTube!"
    except Exception as e:
        return False, str(e)


def get_auth_status() -> dict:
    """Get current authentication status"""
    client_id, client_secret = load_client_credentials()

    status = {
        "client_configured": bool(client_id and client_secret),
        "client_id_exists": os.path.exists(CLIENT_ID_PATH),
        "client_secret_exists": os.path.exists(CLIENT_SECRET_PATH),
        "authenticated": False,
        "channel_name": None,
        "channel_id": None,
        "token_expires": None
    }

    creds = get_credentials()
    if creds:
        status["authenticated"] = True
        if creds.expiry:
            status["token_expires"] = creds.expiry.isoformat()

        # Get channel info
        try:
            youtube = build('youtube', 'v3', credentials=creds)
            request = youtube.channels().list(part='snippet', mine=True)
            response = request.execute()

            if response.get('items'):
                channel = response['items'][0]
                status["channel_name"] = channel['snippet']['title']
                status["channel_id"] = channel['id']
        except Exception:
            pass

    return status


def logout():
    """Clear stored credentials"""
    if os.path.exists(TOKEN_PATH):
        os.remove(TOKEN_PATH)
    return True


def get_youtube_service():
    """Get authenticated YouTube service"""
    creds = get_credentials()
    if not creds:
        return None

    return build('youtube', 'v3', credentials=creds)


def upload_video(
    file_path: str,
    title: str,
    description: str = "",
    tags: list = None,
    category_id: str = "22",
    privacy_status: str = "private",
    progress_callback=None
) -> dict:
    """
    Upload video to YouTube

    Returns:
        dict with 'success', 'video_id', 'video_url', 'error'
    """
    youtube = get_youtube_service()

    if not youtube:
        return {
            "success": False,
            "error": "Not authenticated. Please login first."
        }

    if not os.path.exists(file_path):
        return {
            "success": False,
            "error": f"File not found: {file_path}"
        }

    # Prepare video metadata
    body = {
        "snippet": {
            "title": title[:100],  # YouTube title limit
            "description": description[:5000],  # YouTube description limit
            "tags": tags[:500] if tags else [],  # YouTube tags limit
            "categoryId": category_id
        },
        "status": {
            "privacyStatus": privacy_status,
            "selfDeclaredMadeForKids": False
        }
    }

    try:
        # Create upload request
        media = MediaFileUpload(
            file_path,
            mimetype='video/*',
            resumable=True,
            chunksize=1024*1024  # 1MB chunks
        )

        request = youtube.videos().insert(
            part=",".join(body.keys()),
            body=body,
            media_body=media
        )

        # Execute upload with progress tracking
        response = None
        while response is None:
            status, response = request.next_chunk()
            if status and progress_callback:
                progress_callback(int(status.progress() * 100))

        video_id = response.get('id')
        video_url = f"https://www.youtube.com/watch?v={video_id}"

        return {
            "success": True,
            "video_id": video_id,
            "video_url": video_url,
            "title": title
        }

    except HttpError as e:
        error_msg = str(e)
        if 'quotaExceeded' in error_msg:
            error_msg = "YouTube API quota exceeded. Try again tomorrow."
        elif 'uploadLimitExceeded' in error_msg:
            error_msg = "Upload limit exceeded. Wait before uploading more videos."

        return {
            "success": False,
            "error": error_msg
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def upload_video_async(
    file_path: str,
    title: str,
    description: str = "",
    tags: list = None,
    category_id: str = "22",
    privacy_status: str = "private",
    upload_id: int = None,
    history_callback=None
):
    """Upload video in background and update history"""
    result = upload_video(
        file_path=file_path,
        title=title,
        description=description,
        tags=tags,
        category_id=category_id,
        privacy_status=privacy_status
    )

    if history_callback and upload_id:
        history_callback(upload_id, result)

    return result
