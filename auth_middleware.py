"""
Authentication Middleware for PGL Application

This module handles user authentication, session management, and access control
for the PGL FastAPI application.

"""

import os
import json
import secrets
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Callable
from fastapi import Request, Response, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import RedirectResponse

# Configure logging
logger = logging.getLogger(__name__)

# Simple in-memory session store
# In production, use a proper session store like Redis
SESSIONS = {}

# Admin credentials
# In production, use a proper database with hashed passwords
ADMIN_USERS = {
    "admin": "pgl_admin_password",  # Change this to a strong password!
}

# Staff users
STAFF_USERS = {
    "staff": "pgl_staff_password",  # Change this to a strong password!
}

# Session expiration time in minutes
SESSION_EXPIRY_MINUTES = 60

# Paths that don't require authentication
PUBLIC_PATHS = [
    "/login",
    "/static",
    "/favicon.ico",
    "/api-status",
    "/trigger_automation"  # Allow direct API access without auth
]

# Paths that only admins can access
ADMIN_ONLY_PATHS = [
    "/ai-usage",
    "/podcast-cost",
    "/podcast-cost-dashboard",
    "/storage-status",
]

# Additional paths that staff can access
STAFF_ACCESSIBLE_PATHS = [
    "/list_tasks",
    "/task_status",
    "/stop_task"
]

def generate_session_id() -> str:
    """Generate a secure random session ID"""
    return secrets.token_urlsafe(32)

def create_session(username: str, role: str) -> str:
    """
    Create a new session for a user
    
    Args:
        username: The username
        role: User role (admin or staff)
        
    Returns:
        Session ID
    """
    session_id = generate_session_id()
    expiry = datetime.now() + timedelta(minutes=SESSION_EXPIRY_MINUTES)
    
    SESSIONS[session_id] = {
        "username": username,
        "role": role,
        "expiry": expiry
    }
    
    logger.info(f"Created new session for {username} with role {role}")
    return session_id

def validate_session(session_id: str) -> Optional[Dict]:
    """
    Validate a session ID and return the session data if valid
    
    Args:
        session_id: The session ID to validate
        
    Returns:
        Session data dict or None if invalid
    """
    if not session_id or session_id not in SESSIONS:
        return None
    
    session = SESSIONS[session_id]
    
    # Check if session has expired
    if datetime.now() > session["expiry"]:
        # Clean up expired session
        del SESSIONS[session_id]
        return None
    
    # Extend session expiry
    session["expiry"] = datetime.now() + timedelta(minutes=SESSION_EXPIRY_MINUTES)
    return session

def authenticate_user(username: str, password: str) -> Optional[str]:
    """
    Authenticate a user and return their role if successful
    
    Args:
        username: The username
        password: The password
        
    Returns:
        User role (admin or staff) or None if authentication fails
    """
    if username in ADMIN_USERS and ADMIN_USERS[username] == password:
        return "admin"
    
    if username in STAFF_USERS and STAFF_USERS[username] == password:
        return "staff"
    
    return None

def is_path_public(path: str) -> bool:
    """
    Check if a path is public (doesn't require authentication)
    
    Args:
        path: The request path
        
    Returns:
        True if the path is public, False otherwise
    """
    for public_path in PUBLIC_PATHS:
        if path == public_path or path.startswith(public_path + "/"):
            return True
    return False

def requires_admin(path: str) -> bool:
    """
    Check if a path requires admin privileges
    
    Args:
        path: The request path
        
    Returns:
        True if the path requires admin privileges, False otherwise
    """
    # First check if the path is in the staff accessible paths
    for staff_path in STAFF_ACCESSIBLE_PATHS:
        if path == staff_path or path.startswith(staff_path + "/"):
            return False
    
    # Then check if it's in the admin only paths
    for admin_path in ADMIN_ONLY_PATHS:
        if path == admin_path or path.startswith(admin_path + "/"):
            return True
    
    # If not explicitly admin-only, allow access to staff
    return False

class AuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware that handles authentication and authorization
    """
    
    async def dispatch(self, request: Request, call_next):
        # Get the request path
        path = request.url.path
        
        # Check if the path requires authentication
        if is_path_public(path):
            # Allow access to public paths
            return await call_next(request)
        
        # Get session cookie
        session_id = request.cookies.get("session_id")
        session = validate_session(session_id) if session_id else None
        
        # Check authentication
        if not session:
            # Redirect to login page if not authenticated
            return RedirectResponse(url="/login", status_code=303)
        
        # Check authorization for admin-only paths
        if requires_admin(path) and session["role"] != "admin":
            # Return 403 Forbidden if not authorized
            return Response(
                content="Access denied: Admin privileges required",
                status_code=403
            )
        
        # Set session in request state for access in route handlers
        request.state.session = session
        request.state.username = session["username"]
        request.state.role = session["role"]
        
        # Continue with the request
        return await call_next(request)

# Function to get the current user from the session
def get_current_user(request: Request) -> Dict:
    """
    Dependency function to get the current user from the session
    
    Args:
        request: The FastAPI request object
        
    Returns:
        User data dict
        
    Raises:
        HTTPException: If user is not authenticated
    """
    if not hasattr(request.state, "session"):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    return {
        "username": request.state.username,
        "role": request.state.role
    }

# Function to get admin user
def get_admin_user(request: Request) -> Dict:
    """
    Dependency function to get an admin user
    
    Args:
        request: The FastAPI request object
        
    Returns:
        Admin user data dict
        
    Raises:
        HTTPException: If user is not an admin
    """
    user = get_current_user(request)
    
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin privileges required")
    
    return user 