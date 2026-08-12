#!/usr/bin/env python3
"""
FastMCP server for Basecamp integration.

This server implements the MCP (Model Context Protocol) using the official
Anthropic FastMCP framework, replacing the custom JSON-RPC implementation.
"""

import logging
import os
import sys
from typing import Any, Dict, List, Optional
import anyio
import httpx
from fastmcp import FastMCP


# Import existing business logic
from basecamp_client import BasecampClient
from search_utils import BasecampSearch
import token_storage
import auth_manager
from dotenv import load_dotenv

# Determine project root (directory containing this script)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DOTENV_PATH = os.path.join(PROJECT_ROOT, '.env')
load_dotenv(DOTENV_PATH)

# Set up logging to file AND stderr (following MCP best practices)
LOG_FILE_PATH = os.path.join(PROJECT_ROOT, 'basecamp_fastmcp.log')
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler(sys.stderr)  # Critical: log to stderr, not stdout
    ]
)
logger = logging.getLogger('basecamp_fastmcp')

# Initialize FastMCP server
mcp = FastMCP("basecamp")

# Auth helper functions (reused from original server)
async def _get_basecamp_client() -> Optional[BasecampClient]:
    """Get authenticated Basecamp client (sync version from original server)."""
    try:
        token_data = token_storage.get_token()
        logger.debug(f"Token data retrieved: {token_data}")

        if not token_data or not token_data.get('access_token'):
            logger.error("No OAuth token available")
            return None

        # Check and automatically refresh if token is expired
        if not auth_manager.ensure_authenticated():
            logger.error("OAuth token has expired and automatic refresh failed")
            return None

        # Get fresh token data after potential refresh
        token_data = token_storage.get_token()

        # Get account_id from token data first, then fall back to env var
        account_id = token_data.get('account_id') or os.getenv('BASECAMP_ACCOUNT_ID')
        user_agent = os.getenv('USER_AGENT') or "Basecamp MCP Server (cursor@example.com)"

        if not account_id:
            logger.error(f"Missing account_id. Token data: {token_data}, Env BASECAMP_ACCOUNT_ID: {os.getenv('BASECAMP_ACCOUNT_ID')}")
            return None

        logger.debug(f"Creating Basecamp client with account_id: {account_id}, user_agent: {user_agent}")

        return BasecampClient(
            access_token=token_data['access_token'],
            account_id=account_id,
            user_agent=user_agent,
            auth_mode='oauth'
        )
    except Exception as e:
        logger.error(f"Error creating Basecamp client: {e}")
        return None

async def _get_auth_error_response() -> Dict[str, Any]:
    """Return consistent auth error response."""
    if token_storage.is_token_expired():
        return {
            "error": "OAuth token expired",
            "message": "Your Basecamp OAuth token has expired. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
        }
    else:
        return {
            "error": "Authentication required", 
            "message": "Please authenticate with Basecamp first. Visit http://localhost:8000 to log in."
        }

async def _run_sync(func, *args, **kwargs):
    """Wrapper to run synchronous functions in thread pool."""
    return await anyio.to_thread.run_sync(func, *args, **kwargs)

# Core MCP Tools - Starting with essential ones from original server

@mcp.tool()
async def get_projects() -> Dict[str, Any]:
    """Get all Basecamp projects."""
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()
    
    try:
        projects = await _run_sync(client.get_projects)
        return {
            "status": "success",
            "projects": projects,
            "count": len(projects)
        }
    except Exception as e:
        logger.error(f"Error getting projects: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

@mcp.tool()
async def get_project(project_id: str) -> Dict[str, Any]:
    """Get details for a specific project.
    
    Args:
        project_id: The project ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()
    
    try:
        project = await _run_sync(client.get_project, project_id)
        return {
            "status": "success",
            "project": project
        }
    except Exception as e:
        logger.error(f"Error getting project {project_id}: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

@mcp.tool()
async def get_search_metadata() -> Dict[str, Any]:
    """Get valid filter values for search_basecamp's type_names and file_type parameters.

    Always call this before relying on a specific type_names or file_type value, since
    the available options can vary by account.
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        metadata = await _run_sync(client.get_search_metadata)
        return {"status": "success", "metadata": metadata}
    except Exception as e:
        logger.error(f"Error getting search metadata: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {"error": "OAuth token expired", "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."}
        return {"error": "Execution error", "message": str(e)}


@mcp.tool()
async def search_basecamp(
    query: str,
    type_names: str = "",
    bucket_ids: str = "",
    creator_ids: str = "",
    file_type: str = "",
    exclude_chat: bool = False,
    since: str = "",
    sort: str = "",
    page: int = 1,
    per_page: int = 50,
) -> Dict[str, Any]:
    """Search recordings across the account using Basecamp's native search.

    Call get_search_metadata first to discover valid type_names/file_type values.

    Args:
        query: The search query string
        type_names: Comma-separated recording types to include (e.g. "Todo,Message")
        bucket_ids: Comma-separated project IDs to filter by
        creator_ids: Comma-separated creator person IDs to filter by
        file_type: Attachment file type to filter by
        exclude_chat: Set True to exclude chat results
        since: One of 'last_7_days', 'last_30_days', 'last_90_days', 'last_12_months', 'forever'
        sort: 'best_match' (default) or 'recency'
        page: Page number, default 1
        per_page: Results per page, default 50
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        results = await _run_sync(
            lambda: client.search(
                query,
                type_names=[t.strip() for t in type_names.split(',') if t.strip()] if type_names else None,
                bucket_ids=[b.strip() for b in bucket_ids.split(',') if b.strip()] if bucket_ids else None,
                creator_ids=[c.strip() for c in creator_ids.split(',') if c.strip()] if creator_ids else None,
                file_type=file_type if file_type else None,
                exclude_chat=exclude_chat,
                since=since if since else None,
                sort=sort if sort else None,
                page=page,
                per_page=per_page,
            )
        )
        return {
            "status": "success",
            "query": query,
            "results": results,
            "count": len(results) if isinstance(results, list) else None,
        }
    except Exception as e:
        logger.error(f"Error searching Basecamp: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

@mcp.tool()
async def get_todolists(project_id, todoset_id=""):
    """Get todo lists for a project, optionally filtered by todoset.
    Args:
        project_id: The project ID
        todoset_id: Optional todoset ID to filter todolists. If empty, returns todolists from all todosets.
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()
    
    try:
        todolists = await _run_sync(client.get_todolists, project_id, todoset_id)
        return {
            "status": "success",
            "todolists": todolists,
            "count": len(todolists)
        }
    except Exception as e:
        logger.error(f"Error getting todolists: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

@mcp.tool()
async def get_todosets(project_id: str) -> Dict[str, Any]:
    """Get all todosets for a project.
    
    Args:
        project_id: The project ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()
    
    try:
        todosets = await _run_sync(client.get_todosets, project_id)
        return {
            "status": "success",
            "todosets": todosets,
            "count": len(todosets)
        }
    except Exception as e:
        logger.error(f"Error getting todosets: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

@mcp.tool()
async def get_todoset(project_id: str, todoset_id: str = "") -> Dict[str, Any]:
    """Get a specific todoset for a project.
    
    Args:
        project_id: The project ID
        todoset_id: Optional specific todoset ID. If not provided, returns the first todoset found.
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()
    
    try:
        todoset = await _run_sync(client.get_todoset, project_id, todoset_id if todoset_id else None)
        return {
            "status": "success",
            "todoset": todoset
        }
    except Exception as e:
        logger.error(f"Error getting todoset: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

@mcp.tool()
async def get_todos(project_id: str, todolist_id: str) -> Dict[str, Any]:
    """Get todos from a todo list.
    
    Args:
        project_id: Project ID
        todolist_id: The todo list ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()
    
    try:
        todos = await _run_sync(client.get_todos, project_id, todolist_id)
        return {
            "status": "success",
            "todos": todos,
            "count": len(todos)
        }
    except Exception as e:
        logger.error(f"Error getting todos: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

@mcp.tool()
async def get_todo(project_id: str, todo_id: str) -> Dict[str, Any]:
    """Get a single todo item by its ID.

    Args:
        project_id: Project ID. Kept for backward compatibility; the request is scoped by the record's own ID alone.
        todo_id: The todo ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        todo = await _run_sync(client.get_todo, project_id, todo_id)
        return {
            "status": "success",
            "todo": todo
        }
    except Exception as e:
        logger.error(f"Error getting todo {todo_id}: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

@mcp.tool()
async def create_todo(project_id, todolist_id, content, description="", notify=False):
    """Create a new todo item in a todo list.
    
    Args:
        project_id: Project ID
        todolist_id: The todo list ID
        content: The todo item's text (required)
        description: HTML description of the todo
        notify: Whether to notify assignees
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()
    
    try:
        # Convert empty strings to None
        desc = description if description else None
        
        # Use lambda to properly handle keyword arguments
        todo = await _run_sync(
            lambda: client.create_todo(
                project_id, todolist_id, content,
                description=desc,
                notify=notify
            )
        )
        return {
            "status": "success",
            "todo": todo,
            "message": f"Todo '{content}' created successfully"
        }
    except Exception as e:
        logger.error(f"Error creating todo: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

@mcp.tool()
async def update_todo(
    project_id: str,
    todo_id: str,
    content: str = "__NOT_SET__",
    description: str = "__NOT_SET__",
    assignee_ids: Optional[List[int]] = None,
    completion_subscriber_ids: Optional[List[int]] = None,
    notify: bool = False,
    due_on: str = "__NOT_SET__",
    starts_on: str = "__NOT_SET__",
) -> Dict[str, Any]:
    """Update an existing todo item.

    Args:
        project_id: Project ID. Kept for backward compatibility; the request is scoped by the record's own ID alone.
        todo_id: The todo ID
        content: The todo item's text
        description: HTML description of the todo
        assignee_ids: Optional list of person IDs to assign
        completion_subscriber_ids: Optional list of person IDs to notify on completion
        due_on: Due date in YYYY-MM-DD format
        starts_on: Start date in YYYY-MM-DD format
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()
    
    try:
        # Convert sentinel values to None
        content_val = None if content == "__NOT_SET__" else content
        desc_val = None if description == "__NOT_SET__" else description
        assignees_val = assignee_ids
        subscribers_val = completion_subscriber_ids
        due_val = None if due_on == "__NOT_SET__" else due_on
        starts_val = None if starts_on == "__NOT_SET__" else starts_on
        
        # Guard against no-op updates
        if all(v is None for v in [content_val, desc_val, assignees_val,
                                   subscribers_val, due_val, starts_val]) and notify == False:
            return {
                "error": "Invalid input",
                "message": "At least one field to update must be provided"
            }
        # Use lambda to properly handle keyword arguments
        todo = await _run_sync(
            lambda: client.update_todo(
                project_id, todo_id,
                content=content_val,
                description=desc_val,
                assignee_ids=assignees_val,
                completion_subscriber_ids=subscribers_val,
                notify=notify,
                due_on=due_val,
                starts_on=starts_val
            )
        )
        return {
            "status": "success",
            "todo": todo,
            "message": "Todo updated successfully"
        }
    except Exception as e:
        logger.error(f"Error updating todo: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

@mcp.tool()
async def delete_todo(project_id: str, todo_id: str) -> Dict[str, Any]:
    """Move a todo item to the trash.

    Trashed todos can be recovered from the Basecamp web UI within 30 days.

    Args:
        project_id: Project ID. Kept for backward compatibility; the request is scoped by the record's own ID alone.
        todo_id: The todo ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        await _run_sync(client.delete_todo, project_id, todo_id)
        return {
            "status": "success",
            "message": "Todo moved to trash"
        }
    except Exception as e:
        logger.error(f"Error trashing todo: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

@mcp.tool()
async def complete_todo(project_id: str, todo_id: str) -> Dict[str, Any]:
    """Mark a todo item as complete.

    Args:
        project_id: Project ID. Kept for backward compatibility; the request is scoped by the record's own ID alone.
        todo_id: The todo ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()
    
    try:
        completion = await _run_sync(client.complete_todo, project_id, todo_id)
        return {
            "status": "success",
            "completion": completion,
            "message": "Todo marked as complete"
        }
    except Exception as e:
        logger.error(f"Error completing todo: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

@mcp.tool()
async def uncomplete_todo(project_id: str, todo_id: str) -> Dict[str, Any]:
    """Mark a todo item as incomplete.

    Args:
        project_id: Project ID. Kept for backward compatibility; the request is scoped by the record's own ID alone.
        todo_id: The todo ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()
    
    try:
        await _run_sync(client.uncomplete_todo, project_id, todo_id)
        return {
            "status": "success",
            "message": "Todo marked as incomplete"
        }
    except Exception as e:
        logger.error(f"Error uncompleting todo: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

@mcp.tool()
async def archive_todo(project_id: str, todo_id: str) -> Dict[str, Any]:
    """Archive a todo item.

    Archived todos are hidden from the active list but remain accessible
    via the Basecamp web UI.

    Args:
        project_id: Project ID. Kept for backward compatibility; the request is scoped by the record's own ID alone.
        todo_id: The todo ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        await _run_sync(client.archive_todo, project_id, todo_id)
        return {"status": "success", "message": f"Todo {todo_id} archived"}
    except Exception as e:
        logger.error(f"Error archiving todo {todo_id}: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {"error": "OAuth token expired", "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."}
        return {"error": "Execution error", "message": str(e)}


@mcp.tool()
async def reposition_todo(
    project_id: str,
    todo_id: str,
    position: int,
    parent_id: str = "",
) -> Dict[str, Any]:
    """Reposition a todo within its list, or move it to another list or group.

    Args:
        project_id: The project ID. Kept for backward compatibility; the request is scoped by the record's own ID alone.
        todo_id: The todo ID
        position: New 1-based position within the target list
        parent_id: ID of the target todolist or group to move the todo into.
                   Omit to keep the todo in its current list and only change position.
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    if position < 1:
        return {"error": "Invalid input", "message": "position must be >= 1"}

    try:
        await _run_sync(
            lambda: client.reposition_todo(project_id, todo_id, position, parent_id if parent_id else None)
        )
        return {"status": "success", "message": f"Todo {todo_id} moved to position {position}"}
    except Exception as e:
        logger.error(f"Error repositioning todo {todo_id}: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {"error": "OAuth token expired", "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."}
        return {"error": "Execution error", "message": str(e)}


@mcp.tool()
async def global_search(query: str) -> Dict[str, Any]:
    """Search projects, todos and campfire messages across all projects.
    
    Args:
        query: Search query
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()
    
    try:
        search = BasecampSearch(client=client)
        results = await _run_sync(search.global_search, query)
        return {
            "status": "success",
            "query": query,
            "results": results
        }
    except Exception as e:
        logger.error(f"Error in global search: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

@mcp.tool()
async def get_comments(recording_id: str, project_id: str, page: int = 1) -> Dict[str, Any]:
    """Get comments for a Basecamp item.

    Args:
        recording_id: The item ID
        project_id: The project ID
        page: Page number for pagination (default: 1). Basecamp uses geared pagination:
              page 1 has 15 results, page 2 has 30, page 3 has 50, page 4+ has 100.
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        result = await _run_sync(client.get_comments, project_id, recording_id, page)
        return {
            "status": "success",
            "comments": result["comments"],
            "count": len(result["comments"]),
            "page": page,
            "total_count": result["total_count"],
            "next_page": result["next_page"]
        }
    except Exception as e:
        logger.error(f"Error getting comments: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

@mcp.tool()
async def create_comment(recording_id: str, project_id: str, content: str) -> Dict[str, Any]:
    """Create a comment on a Basecamp item.

    Args:
        recording_id: The item ID
        project_id: The project ID
        content: The comment content in HTML format
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        comment = await _run_sync(client.create_comment, recording_id, project_id, content)
        return {
            "status": "success",
            "comment": comment,
            "message": "Comment created successfully"
        }
    except Exception as e:
        logger.error(f"Error creating comment: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again.",
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }


@mcp.tool()
async def get_comment(project_id: str, comment_id: str) -> Dict[str, Any]:
    """Get a specific comment by ID.

    Args:
        project_id: The project ID
        comment_id: The comment ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        comment = await _run_sync(client.get_comment, project_id, comment_id)
        return {
            "status": "success",
            "comment": comment
        }
    except Exception as e:
        logger.error(f"Error getting comment: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }


@mcp.tool()
async def update_comment(project_id: str, comment_id: str, content: str) -> Dict[str, Any]:
    """Update the content of an existing comment.

    Args:
        project_id: The project ID
        comment_id: The comment ID
        content: New comment content in HTML format
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        comment = await _run_sync(client.update_comment, project_id, comment_id, content)
        return {
            "status": "success",
            "comment": comment,
            "message": "Comment updated successfully"
        }
    except Exception as e:
        logger.error(f"Error updating comment: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }


@mcp.tool()
async def trash_comment(project_id: str, comment_id: str) -> Dict[str, Any]:
    """Move a comment to the trash.

    Args:
        project_id: The project ID
        comment_id: The comment ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        await _run_sync(client.trash_comment, project_id, comment_id)
        return {"status": "success", "message": f"Comment {comment_id} moved to trash"}
    except Exception as e:
        logger.error(f"Error trashing comment {comment_id}: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {"error": "OAuth token expired", "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."}
        return {"error": "Execution error", "message": str(e)}

@mcp.tool()
async def get_campfire_lines(project_id: str, campfire_id: str) -> Dict[str, Any]:
    """Get recent messages from a Basecamp campfire (chat room).
    
    Args:
        project_id: The project ID
        campfire_id: The campfire/chat room ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()
    
    try:
        lines = await _run_sync(client.get_campfire_lines, project_id, campfire_id)
        return {
            "status": "success",
            "campfire_lines": lines,
            "count": len(lines)
        }
    except Exception as e:
        logger.error(f"Error getting campfire lines: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

@mcp.tool()
async def get_campfire(project_id: str) -> Dict[str, Any]:
    """Get the campfire (chat room) for a project.

    Args:
        project_id: The project ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        campfire = await _run_sync(client.get_campfire, project_id)
        return {
            "status": "success",
            "campfire": campfire
        }
    except Exception as e:
        logger.error(f"Error getting campfire: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

@mcp.tool()
async def get_message_board(project_id: str) -> Dict[str, Any]:
    """Get the message board for a project.

    Args:
        project_id: The project ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        message_board = await _run_sync(client.get_message_board, project_id)
        return {
            "status": "success",
            "message_board": message_board
        }
    except Exception as e:
        logger.error(f"Error getting message board: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

@mcp.tool()
async def get_messages(project_id: str, message_board_id: str = "") -> Dict[str, Any]:
    """Get all messages from a project's message board.

    Args:
        project_id: The project ID
        message_board_id: Optional message board ID. If not provided, will be auto-discovered from the project.
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        messages = await _run_sync(client.get_messages, project_id, message_board_id if message_board_id else None)
        return {
            "status": "success",
            "messages": messages,
            "count": len(messages)
        }
    except Exception as e:
        logger.error(f"Error getting messages: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

@mcp.tool()
async def get_message(project_id: str, message_id: str) -> Dict[str, Any]:
    """Get a specific message by ID.

    Args:
        project_id: The project ID. Kept for backward compatibility; the request is scoped by the record's own ID alone.
        message_id: The message ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        message = await _run_sync(client.get_message, project_id, message_id)
        return {
            "status": "success",
            "message": message
        }
    except Exception as e:
        logger.error(f"Error getting message: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }


@mcp.tool()
async def get_message_categories(project_id: str) -> Dict[str, Any]:
    """Get message categories (types) for a project.

    Args:
        project_id: The project ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        categories = await _run_sync(client.get_message_categories, project_id)
        return {
            "status": "success",
            "categories": categories,
            "count": len(categories)
        }
    except Exception as e:
        logger.error(f"Error getting message categories: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }


@mcp.tool()
async def create_message(project_id: str, subject: str, content: str,
                         message_board_id: str = "",
                         category_id: str = "",
                         status: str = "",
                         subscriptions=None,
                         visible_to_clients: bool = False) -> Dict[str, Any]:
    """Create a new message on a project's message board.

    Args:
        project_id: The project ID
        subject: Message title/subject
        content: Message body in HTML format
        message_board_id: Optional message board ID. If not provided, will be auto-discovered from the project.
        category_id: Optional message type/category ID
        status: Optional status - "active" to publish immediately (default) or "drafted" to save as a draft
        subscriptions: Optional list of person IDs to notify and subscribe. If omitted, everyone on the project is notified.
        visible_to_clients: Whether the message is visible to clients when the project has clients enabled
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        message = await _run_sync(
            lambda: client.create_message(
                project_id, subject, content,
                message_board_id=message_board_id if message_board_id else None,
                category_id=category_id if category_id else None,
                status=status if status else None,
                subscriptions=subscriptions,
                visible_to_clients=visible_to_clients
            )
        )
        return {
            "status": "success",
            "message": message,
            "result": f"Message '{subject}' created successfully"
        }
    except Exception as e:
        logger.error(f"Error creating message: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }


@mcp.tool()
async def update_message(message_id: str, subject: str = "", content: str = "",
                         category_id: str = "", subscriptions=None,
                         notify: bool = False) -> Dict[str, Any]:
    """Update an existing message's subject, content, or category.

    Args:
        message_id: The message ID
        subject: New title. Leave blank to keep the current subject.
        content: New HTML body. Leave blank to keep the current content.
        category_id: New message type/category ID
        subscriptions: Optional list of person IDs to recompute subscribers.
            Omit subscriptions to keep the message's current subscribers (notify is always sent).
        notify: Whether to notify newly added subscribers
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    if not subject and not content and not category_id and subscriptions is None:
        return {"error": "Invalid input", "message": "At least one of subject, content, category_id, or subscriptions must be provided"}

    try:
        message = await _run_sync(
            lambda: client.update_message(
                message_id,
                subject=subject if subject else None,
                content=content if content else None,
                category_id=category_id if category_id else None,
                subscriptions=subscriptions,
                notify=notify
            )
        )
        return {"status": "success", "message": message}
    except Exception as e:
        logger.error(f"Error updating message {message_id}: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {"error": "OAuth token expired", "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."}
        return {"error": "Execution error", "message": str(e)}


@mcp.tool()
async def pin_message(message_id: str) -> Dict[str, Any]:
    """Pin a message to the top of its message board.

    Args:
        message_id: The message ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        await _run_sync(client.pin_message, message_id)
        return {"status": "success", "message": f"Message {message_id} pinned"}
    except Exception as e:
        logger.error(f"Error pinning message {message_id}: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {"error": "OAuth token expired", "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."}
        return {"error": "Execution error", "message": str(e)}


@mcp.tool()
async def unpin_message(message_id: str) -> Dict[str, Any]:
    """Unpin a message.

    Args:
        message_id: The message ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        await _run_sync(client.unpin_message, message_id)
        return {"status": "success", "message": f"Message {message_id} unpinned"}
    except Exception as e:
        logger.error(f"Error unpinning message {message_id}: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {"error": "OAuth token expired", "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."}
        return {"error": "Execution error", "message": str(e)}


@mcp.tool()
async def trash_message(message_id: str) -> Dict[str, Any]:
    """Move a message to the trash.

    Args:
        message_id: The message ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        await _run_sync(client.trash_message, message_id)
        return {"status": "success", "message": f"Message {message_id} moved to trash"}
    except Exception as e:
        logger.error(f"Error trashing message {message_id}: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {"error": "OAuth token expired", "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."}
        return {"error": "Execution error", "message": str(e)}


# Inbox Tools (Email Forwards)
@mcp.tool()
async def get_inbox(project_id: str) -> Dict[str, Any]:
    """Get the inbox for a project (for email forwards).

    Args:
        project_id: The project ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        inbox = await _run_sync(client.get_inbox, project_id)
        return {
            "status": "success",
            "inbox": inbox
        }
    except Exception as e:
        logger.error(f"Error getting inbox: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }


@mcp.tool()
async def get_forwards(project_id: str, inbox_id: str = "") -> Dict[str, Any]:
    """Get all forwarded emails from a project's inbox.

    Args:
        project_id: The project ID
        inbox_id: Optional inbox ID. If not provided, will be auto-discovered from the project.
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        forwards = await _run_sync(client.get_forwards, project_id, inbox_id if inbox_id else None)
        return {
            "status": "success",
            "forwards": forwards,
            "count": len(forwards)
        }
    except Exception as e:
        logger.error(f"Error getting forwards: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }


@mcp.tool()
async def get_forward(project_id: str, forward_id: str) -> Dict[str, Any]:
    """Get a specific forwarded email by ID.

    Args:
        project_id: The project ID
        forward_id: The forward ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        forward = await _run_sync(client.get_forward, project_id, forward_id)
        return {
            "status": "success",
            "forward": forward
        }
    except Exception as e:
        logger.error(f"Error getting forward: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }


@mcp.tool()
async def get_inbox_replies(project_id: str, forward_id: str) -> Dict[str, Any]:
    """Get all replies to a forwarded email.

    Args:
        project_id: The project ID
        forward_id: The forward ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        replies = await _run_sync(client.get_inbox_replies, project_id, forward_id)
        return {
            "status": "success",
            "replies": replies,
            "count": len(replies)
        }
    except Exception as e:
        logger.error(f"Error getting inbox replies: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }


@mcp.tool()
async def get_inbox_reply(project_id: str, forward_id: str, reply_id: str) -> Dict[str, Any]:
    """Get a specific reply to a forwarded email.

    Args:
        project_id: The project ID
        forward_id: The forward ID
        reply_id: The reply ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        reply = await _run_sync(client.get_inbox_reply, project_id, forward_id, reply_id)
        return {
            "status": "success",
            "reply": reply
        }
    except Exception as e:
        logger.error(f"Error getting inbox reply: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }


@mcp.tool()
async def trash_forward(project_id: str, forward_id: str) -> Dict[str, Any]:
    """Move a forwarded email to trash.

    Args:
        project_id: The project ID
        forward_id: The forward ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        await _run_sync(client.trash_forward, project_id, forward_id)
        return {
            "status": "success",
            "message": "Forward trashed"
        }
    except Exception as e:
        logger.error(f"Error trashing forward: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }


@mcp.tool()
async def get_card_tables(project_id: str) -> Dict[str, Any]:
    """Get all card tables for a project.
    
    Args:
        project_id: The project ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()
    
    try:
        card_tables = await _run_sync(client.get_card_tables, project_id)
        return {
            "status": "success",
            "card_tables": card_tables,
            "count": len(card_tables)
        }
    except Exception as e:
        logger.error(f"Error getting card tables: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

@mcp.tool()
async def get_card_table(project_id: str) -> Dict[str, Any]:
    """Get the card table details for a project.
    
    Args:
        project_id: The project ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()
    
    try:
        card_table = await _run_sync(client.get_card_table, project_id)
        card_table_details = await _run_sync(client.get_card_table_details, project_id, card_table['id'])
        return {
            "status": "success",
            "card_table": card_table_details
        }
    except Exception as e:
        logger.error(f"Error getting card table: {e}")
        error_msg = str(e)
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "status": "error",
            "message": f"Error getting card table: {error_msg}",
            "debug": error_msg
        }

@mcp.tool()
async def get_columns(project_id: str, card_table_id: str) -> Dict[str, Any]:
    """Get all columns in a card table.
    
    Args:
        project_id: The project ID
        card_table_id: The card table ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()
    
    try:
        columns = await _run_sync(client.get_columns, project_id, card_table_id)
        return {
            "status": "success",
            "columns": columns,
            "count": len(columns)
        }
    except Exception as e:
        logger.error(f"Error getting columns: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

@mcp.tool()
async def get_cards(project_id: str, column_id: str) -> Dict[str, Any]:
    """Get all cards in a column.
    
    Args:
        project_id: The project ID
        column_id: The column ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()
    
    try:
        cards = await _run_sync(client.get_cards, project_id, column_id)
        return {
            "status": "success",
            "cards": cards,
            "count": len(cards)
        }
    except Exception as e:
        logger.error(f"Error getting cards: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

@mcp.tool()
async def create_card(project_id: str, column_id: str, title: str, content: str = "", due_on: str = "", notify: bool = False) -> Dict[str, Any]:
    """Create a new card in a column.
    
    Args:
        project_id: The project ID
        column_id: The column ID
        title: The card title
        content: Optional card content/description
        due_on: Optional due date (ISO 8601 format)
        notify: Whether to notify assignees (default: false)
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()
    
    try:
        card = await _run_sync(client.create_card, project_id, column_id, title, content if content else None, due_on if due_on else None, notify)
        return {
            "status": "success",
            "card": card,
            "message": f"Card '{title}' created successfully"
        }
    except Exception as e:
        logger.error(f"Error creating card: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

@mcp.tool()
async def get_column(project_id: str, column_id: str) -> Dict[str, Any]:
    """Get details for a specific column.
    
    Args:
        project_id: The project ID
        column_id: The column ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()
    
    try:
        column = await _run_sync(client.get_column, project_id, column_id)
        return {
            "status": "success",
            "column": column
        }
    except Exception as e:
        logger.error(f"Error getting column: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

@mcp.tool()
async def create_column(project_id: str, card_table_id: str, title: str) -> Dict[str, Any]:
    """Create a new column in a card table.
    
    Args:
        project_id: The project ID
        card_table_id: The card table ID
        title: The column title
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()
    
    try:
        column = await _run_sync(client.create_column, project_id, card_table_id, title)
        return {
            "status": "success",
            "column": column,
            "message": f"Column '{title}' created successfully"
        }
    except Exception as e:
        logger.error(f"Error creating column: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

@mcp.tool()
async def move_card(project_id: str, card_id: str, column_id: str) -> Dict[str, Any]:
    """Move a card to a new column.
    
    Args:
        project_id: The project ID
        card_id: The card ID
        column_id: The destination column ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()
    
    try:
        await _run_sync(client.move_card, project_id, card_id, column_id)
        return {
            "status": "success",
            "message": f"Card moved to column {column_id}"
        }
    except Exception as e:
        logger.error(f"Error moving card: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

@mcp.tool()
async def complete_card(project_id: str, card_id: str) -> Dict[str, Any]:
    """Mark a card as complete.
    
    Args:
        project_id: The project ID
        card_id: The card ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()
    
    try:
        await _run_sync(client.complete_card, project_id, card_id)
        return {
            "status": "success",
            "message": "Card marked as complete"
        }
    except Exception as e:
        logger.error(f"Error completing card: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

@mcp.tool()
async def get_card(project_id: str, card_id: str) -> Dict[str, Any]:
    """Get details for a specific card.
    
    Args:
        project_id: The project ID
        card_id: The card ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()
    
    try:
        card = await _run_sync(client.get_card, project_id, card_id)
        return {
            "status": "success",
            "card": card
        }
    except Exception as e:
        logger.error(f"Error getting card: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired", 
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

@mcp.tool()
async def update_card(project_id, card_id, title="__NOT_SET__", content="__NOT_SET__", due_on="__NOT_SET__", assignee_ids=None):
    """Update a card.
    
    Args:
        project_id: The project ID
        card_id: The card ID
        title: The new card title
        content: The new card content/description
        due_on: Due date (ISO 8601 format)
        assignee_ids: Optional array of person IDs to assign to the card
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()
    
    try:
        title_val = None if title == "__NOT_SET__" else title
        content_val = None if content == "__NOT_SET__" else content
        due_on_val = None if due_on == "__NOT_SET__" else due_on
        assignee_ids_val = assignee_ids
        
        card = await _run_sync(client.update_card, project_id, card_id, title_val, content_val, due_on_val, assignee_ids_val)
        return {
            "status": "success",
            "card": card,
            "message": "Card updated successfully"
        }
    except Exception as e:
        logger.error(f"Error updating card: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

@mcp.tool()
async def get_daily_check_ins(project_id: str, page: int = 0) -> Dict[str, Any]:
    """Get project's daily checking questionnaire.
    
    Args:
        project_id: The project ID
        page: Page number paginated response
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()
    
    try:
        page_val = page if page > 0 else None
        if page_val is not None and not isinstance(page_val, int):
            page_val = 1
        answers = await _run_sync(client.get_daily_check_ins, project_id, page=page_val or 1)
        return {
            "status": "success",
            "campfire_lines": answers,
            "count": len(answers)
        }
    except Exception as e:
        logger.error(f"Error getting daily check ins: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

@mcp.tool()
async def get_question_answers(project_id: str, question_id: str, page: int = 0) -> Dict[str, Any]:
    """Get answers on daily check-in question.
    
    Args:
        project_id: The project ID
        question_id: The question ID
        page: Page number paginated response
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()
    
    try:
        page_val = page if page > 0 else None
        if page_val is not None and not isinstance(page_val, int):
            page_val = 1
        answers = await _run_sync(client.get_question_answers, project_id, question_id, page=page_val or 1)
        return {
            "status": "success",
            "campfire_lines": answers,
            "count": len(answers)
        }
    except Exception as e:
        logger.error(f"Error getting question answers: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

# Column Management Tools
@mcp.tool()
async def update_column(project_id: str, column_id: str, title: str) -> Dict[str, Any]:
    """Update a column title.
    
    Args:
        project_id: The project ID
        column_id: The column ID
        title: The new column title
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()
    
    try:
        column = await _run_sync(client.update_column, project_id, column_id, title)
        return {
            "status": "success",
            "column": column,
            "message": "Column updated successfully"
        }
    except Exception as e:
        logger.error(f"Error updating column: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

@mcp.tool()
async def move_column(project_id: str, card_table_id: str, column_id: str, position: int) -> Dict[str, Any]:
    """Move a column to a new position.
    
    Args:
        project_id: The project ID
        card_table_id: The card table ID
        column_id: The column ID
        position: The new 1-based position
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()
    
    try:
        await _run_sync(client.move_column, project_id, column_id, position, card_table_id)
        return {
            "status": "success",
            "message": f"Column moved to position {position}"
        }
    except Exception as e:
        logger.error(f"Error moving column: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

@mcp.tool()
async def update_column_color(project_id: str, column_id: str, color: str) -> Dict[str, Any]:
    """Update a column color.
    
    Args:
        project_id: The project ID
        column_id: The column ID
        color: The hex color code (e.g., #FF0000)
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()
    
    try:
        column = await _run_sync(client.update_column_color, project_id, column_id, color)
        return {
            "status": "success",
            "column": column,
            "message": f"Column color updated to {color}"
        }
    except Exception as e:
        logger.error(f"Error updating column color: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

@mcp.tool()
async def put_column_on_hold(project_id: str, column_id: str) -> Dict[str, Any]:
    """Put a column on hold (freeze work).
    
    Args:
        project_id: The project ID
        column_id: The column ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()
    
    try:
        await _run_sync(client.put_column_on_hold, project_id, column_id)
        return {
            "status": "success",
            "message": "Column put on hold"
        }
    except Exception as e:
        logger.error(f"Error putting column on hold: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

@mcp.tool()
async def remove_column_hold(project_id: str, column_id: str) -> Dict[str, Any]:
    """Remove hold from a column (unfreeze work).
    
    Args:
        project_id: The project ID
        column_id: The column ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()
    
    try:
        await _run_sync(client.remove_column_hold, project_id, column_id)
        return {
            "status": "success",
            "message": "Column hold removed"
        }
    except Exception as e:
        logger.error(f"Error removing column hold: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

@mcp.tool()
async def watch_column(project_id: str, column_id: str) -> Dict[str, Any]:
    """Subscribe to notifications for changes in a column.
    
    Args:
        project_id: The project ID
        column_id: The column ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()
    
    try:
        await _run_sync(client.watch_column, project_id, column_id)
        return {
            "status": "success",
            "message": "Column notifications enabled"
        }
    except Exception as e:
        logger.error(f"Error watching column: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

@mcp.tool()
async def unwatch_column(project_id: str, column_id: str) -> Dict[str, Any]:
    """Unsubscribe from notifications for a column.
    
    Args:
        project_id: The project ID
        column_id: The column ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()
    
    try:
        await _run_sync(client.unwatch_column, project_id, column_id)
        return {
            "status": "success",
            "message": "Column notifications disabled"
        }
    except Exception as e:
        logger.error(f"Error unwatching column: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

# More Card Management Tools  
@mcp.tool()
async def uncomplete_card(project_id: str, card_id: str) -> Dict[str, Any]:
    """Mark a card as incomplete.
    
    Args:
        project_id: The project ID
        card_id: The card ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()
    
    try:
        await _run_sync(client.uncomplete_card, project_id, card_id)
        return {
            "status": "success",
            "message": "Card marked as incomplete"
        }
    except Exception as e:
        logger.error(f"Error uncompleting card: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

# Card Steps (Sub-tasks) Management
@mcp.tool()
async def get_card_steps(project_id: str, card_id: str) -> Dict[str, Any]:
    """Get all steps (sub-tasks) for a card.
    
    Args:
        project_id: The project ID
        card_id: The card ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()
    
    try:
        steps = await _run_sync(client.get_card_steps, project_id, card_id)
        return {
            "status": "success",
            "steps": steps,
            "count": len(steps)
        }
    except Exception as e:
        logger.error(f"Error getting card steps: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

@mcp.tool()
async def create_card_step(project_id, card_id, title, due_on="", assignee_ids=None):
    """Create a new step (sub-task) for a card.
    
    Args:
        project_id: The project ID
        card_id: The card ID
        title: The step title
        due_on: Optional due date (ISO 8601 format)
        assignee_ids: Optional array of person IDs to assign to the step
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()
    
    try:
        step = await _run_sync(client.create_card_step, project_id, card_id, title, due_on if due_on else None, assignee_ids if assignee_ids else None)
        return {
            "status": "success",
            "step": step,
            "message": f"Step '{title}' created successfully"
        }
    except Exception as e:
        logger.error(f"Error creating card step: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

@mcp.tool()
async def get_card_step(project_id: str, step_id: str) -> Dict[str, Any]:
    """Get details for a specific card step.
    
    Args:
        project_id: The project ID
        step_id: The step ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()
    
    try:
        step = await _run_sync(client.get_card_step, project_id, step_id)
        return {
            "status": "success",
            "step": step
        }
    except Exception as e:
        logger.error(f"Error getting card step: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

@mcp.tool()
async def update_card_step(project_id, step_id, title="__NOT_SET__", due_on="__NOT_SET__", assignee_ids=None):
    """Update a card step.
    
    Args:
        project_id: The project ID
        step_id: The step ID
        title: The step title
        due_on: Due date (ISO 8601 format)
        assignee_ids: Optional array of person IDs to assign to the step
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()
    
    try:
        title_val = None if title == "__NOT_SET__" else title
        due_on_val = None if due_on == "__NOT_SET__" else due_on
        assignee_ids_val = assignee_ids
        
        step = await _run_sync(client.update_card_step, project_id, step_id, title_val, due_on_val, assignee_ids_val)
        return {
            "status": "success",
            "step": step,
            "message": f"Step updated successfully"
        }
    except Exception as e:
        logger.error(f"Error updating card step: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

@mcp.tool()
async def delete_card_step(project_id: str, step_id: str) -> Dict[str, Any]:
    """Delete a card step.
    
    Args:
        project_id: The project ID
        step_id: The step ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()
    
    try:
        await _run_sync(client.delete_card_step, project_id, step_id)
        return {
            "status": "success",
            "message": "Step deleted successfully"
        }
    except Exception as e:
        logger.error(f"Error deleting card step: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

@mcp.tool()
async def complete_card_step(project_id: str, step_id: str) -> Dict[str, Any]:
    """Mark a card step as complete.
    
    Args:
        project_id: The project ID
        step_id: The step ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()
    
    try:
        await _run_sync(client.complete_card_step, project_id, step_id)
        return {
            "status": "success",
            "message": "Step marked as complete"
        }
    except Exception as e:
        logger.error(f"Error completing card step: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

@mcp.tool()
async def uncomplete_card_step(project_id: str, step_id: str) -> Dict[str, Any]:
    """Mark a card step as incomplete.
    
    Args:
        project_id: The project ID
        step_id: The step ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()
    
    try:
        await _run_sync(client.uncomplete_card_step, project_id, step_id)
        return {
            "status": "success",
            "message": "Step marked as incomplete"
        }
    except Exception as e:
        logger.error(f"Error uncompleting card step: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

# Attachments, Events, and Webhooks
@mcp.tool()
async def create_attachment(file_path: str, name: str, content_type: str = "") -> Dict[str, Any]:
    """Upload a file as an attachment.
    
    Args:
        file_path: Local path to file
        name: Filename for Basecamp
        content_type: MIME type
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()
    
    try:
        result = await _run_sync(client.create_attachment, file_path, name, content_type or "application/octet-stream")
        return {
            "status": "success",
            "attachment": result
        }
    except Exception as e:
        logger.error(f"Error creating attachment: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

@mcp.tool()
async def get_events(project_id: str, recording_id: str) -> Dict[str, Any]:
    """Get events for a recording.
    
    Args:
        project_id: Project ID
        recording_id: Recording ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()
    
    try:
        events = await _run_sync(client.get_events, project_id, recording_id)
        return {
            "status": "success",
            "events": events,
            "count": len(events)
        }
    except Exception as e:
        logger.error(f"Error getting events: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

@mcp.tool()
async def get_schedule(project_id: str) -> Dict[str, Any]:
    """Get the schedule for a project.

    Args:
        project_id: The project ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        schedule = await _run_sync(client.get_schedule, project_id)
        return {
            "status": "success",
            "schedule": schedule
        }
    except Exception as e:
        logger.error(f"Error getting schedule: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

@mcp.tool()
async def get_schedule_entries(project_id: str) -> Dict[str, Any]:
    """Get schedule entries (calendar events) for a project.

    Args:
        project_id: The project ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        entries = await _run_sync(client.get_schedule_entries, project_id)
        return {
            "status": "success",
            "schedule_entries": entries,
            "count": len(entries)
        }
    except Exception as e:
        logger.error(f"Error getting schedule entries: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

@mcp.tool()
async def get_webhooks(project_id: str) -> Dict[str, Any]:
    """List webhooks for a project.
    
    Args:
        project_id: Project ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()
    
    try:
        hooks = await _run_sync(client.get_webhooks, project_id)
        return {
            "status": "success",
            "webhooks": hooks,
            "count": len(hooks)
        }
    except Exception as e:
        logger.error(f"Error getting webhooks: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

@mcp.tool()
async def create_webhook(project_id, payload_url, types=None):
    """Create a webhook.
    
    Args:
        project_id: Project ID
        payload_url: Payload URL
        types: Optional event types
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()
    
    try:
        hook = await _run_sync(client.create_webhook, project_id, payload_url, types if types else None)
        return {
            "status": "success",
            "webhook": hook
        }
    except Exception as e:
        logger.error(f"Error creating webhook: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

@mcp.tool()
async def delete_webhook(project_id: str, webhook_id: str) -> Dict[str, Any]:
    """Delete a webhook.
    
    Args:
        project_id: Project ID
        webhook_id: Webhook ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()
    
    try:
        await _run_sync(client.delete_webhook, project_id, webhook_id)
        return {
            "status": "success",
            "message": "Webhook deleted"
        }
    except Exception as e:
        logger.error(f"Error deleting webhook: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }


# External Link Tools
@mcp.tool()
async def get_external_links(project_id: Optional[str] = None, status: Optional[str] = None) -> Dict[str, Any]:
    """List external links (dock tools pointing to outside services like Figma, GitHub, Dropbox).

    Args:
        project_id: Optional project ID to scope to. Omit to list across all visible projects.
        status: Optional filter -- 'active', 'archived', or 'trashed'
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        links = await _run_sync(client.get_external_links, project_id, status)
        return {
            "status": "success",
            "external_links": links,
            "count": len(links)
        }
    except Exception as e:
        logger.error(f"Error getting external links: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }


@mcp.tool()
async def get_external_link(link_id: str) -> Dict[str, Any]:
    """Get a single external link by ID. Note: omits service/description and its url
    is a Basecamp redirector, not the outside address -- use get_external_links for the full shape.

    Args:
        link_id: The external link ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        link = await _run_sync(client.get_external_link, link_id)
        return {
            "status": "success",
            "external_link": link
        }
    except Exception as e:
        logger.error(f"Error getting external link: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }


@mcp.tool()
async def create_external_link(project_id: str, service: str, url: str, title: Optional[str] = None, description: Optional[str] = None) -> Dict[str, Any]:
    """Create an external link (dock tool) pointing to an outside service.

    Args:
        project_id: The project ID
        service: Short service identifier -- e.g. figma, dropbox, google_drive, github, notion, trello, slack, zoom, or other
        url: HTTP/HTTPS address the link points to
        title: Optional display name (defaults to "Untitled")
        description: Optional rich-text (HTML) description
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        await _run_sync(client.create_external_link, project_id, service, url, title, description)
        return {
            "status": "success",
            "message": "External link created. Call get_external_links to find its ID -- Basecamp does not return it on creation."
        }
    except Exception as e:
        logger.error(f"Error creating external link: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }


@mcp.tool()
async def rename_external_link(link_id: str, title: str) -> Dict[str, Any]:
    """Rename an external link. Only the title can be changed this way --
    to change the URL, service, or description, trash this link and create a new one.

    Args:
        link_id: The external link ID
        title: New display name
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        link = await _run_sync(client.rename_external_link, link_id, title)
        return {
            "status": "success",
            "external_link": link,
            "message": "External link renamed successfully"
        }
    except Exception as e:
        logger.error(f"Error renaming external link: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }


@mcp.tool()
async def trash_external_link(link_id: str) -> Dict[str, Any]:
    """Move an external link to the trash.

    Args:
        link_id: The external link ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        await _run_sync(client.trash_external_link, link_id)
        return {"status": "success", "message": f"External link {link_id} moved to trash"}
    except Exception as e:
        logger.error(f"Error trashing external link {link_id}: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {"error": "OAuth token expired", "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."}
        return {"error": "Execution error", "message": str(e)}


# Vault Management
@mcp.tool()
async def get_vault(project_id: str, vault_id: str = "") -> Dict[str, Any]:
    """Get a vault. Returns the project's root vault if vault_id is omitted.

    Args:
        project_id: Project ID
        vault_id: Optional vault ID. If omitted, returns the project's root vault.
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        vault = await _run_sync(client.get_vault, project_id, vault_id if vault_id else None)
        return {
            "status": "success",
            "vault": vault
        }
    except Exception as e:
        logger.error(f"Error getting vault: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

@mcp.tool()
async def get_vault_children(project_id: str, vault_id: str = "") -> Dict[str, Any]:
    """List everything filed directly under a vault (sub-vaults, cloud files, docs, uploads). Uses the project's root vault if vault_id is omitted.

    Args:
        project_id: Project ID
        vault_id: Optional parent vault ID. If omitted, lists children of the project's root vault.
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        children = await _run_sync(client.get_vault_children, project_id, vault_id if vault_id else None)
        return {
            "status": "success",
            "children": children,
            "count": len(children)
        }
    except Exception as e:
        logger.error(f"Error getting vault children: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

# Document Management
@mcp.tool()
async def get_documents(project_id: str, vault_id: str) -> Dict[str, Any]:
    """List documents in a vault.
    
    Args:
        project_id: Project ID
        vault_id: Vault ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()
    
    try:
        docs = await _run_sync(client.get_documents, project_id, vault_id)
        return {
            "status": "success",
            "documents": docs,
            "count": len(docs)
        }
    except Exception as e:
        logger.error(f"Error getting documents: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

@mcp.tool()
async def get_document(project_id: str, document_id: str) -> Dict[str, Any]:
    """Get a single document.
    
    Args:
        project_id: Project ID
        document_id: Document ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()
    
    try:
        doc = await _run_sync(client.get_document, project_id, document_id)
        return {
            "status": "success",
            "document": doc
        }
    except Exception as e:
        logger.error(f"Error getting document: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

@mcp.tool()
async def create_document(project_id: str, vault_id: str, title: str, content: str) -> Dict[str, Any]:
    """Create a document in a vault.
    
    Args:
        project_id: Project ID
        vault_id: Vault ID
        title: Document title
        content: Document HTML content
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()
    
    try:
        doc = await _run_sync(client.create_document, project_id, vault_id, title, content)
        return {
            "status": "success",
            "document": doc
        }
    except Exception as e:
        logger.error(f"Error creating document: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

@mcp.tool()
async def update_document(project_id: str, document_id: str, title: str = "__NOT_SET__", content: str = "__NOT_SET__") -> Dict[str, Any]:
    """Update a document.
    
    Args:
        project_id: Project ID
        document_id: Document ID
        title: New title
        content: New HTML content
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()
    
    try:
        title_val = None if title == "__NOT_SET__" else title
        content_val = None if content == "__NOT_SET__" else content
        
        doc = await _run_sync(client.update_document, project_id, document_id, title_val, content_val)
        return {
            "status": "success",
            "document": doc
        }
    except Exception as e:
        logger.error(f"Error updating document: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

@mcp.tool()
async def trash_document(project_id: str, document_id: str) -> Dict[str, Any]:
    """Move a document to trash.
    
    Args:
        project_id: Project ID
        document_id: Document ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()
    
    try:
        await _run_sync(client.trash_document, project_id, document_id)
        return {
            "status": "success",
            "message": "Document trashed"
        }
    except Exception as e:
        logger.error(f"Error trashing document: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

# Upload Management
@mcp.tool()
async def get_uploads(project_id: str, vault_id: str = "") -> Dict[str, Any]:
    """List uploads in a project or vault.
    
    Args:
        project_id: Project ID
        vault_id: Optional vault ID to limit to specific vault
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()
    
    try:
        uploads = await _run_sync(client.get_uploads, project_id, vault_id if vault_id else None)
        return {
            "status": "success",
            "uploads": uploads,
            "count": len(uploads)
        }
    except Exception as e:
        logger.error(f"Error getting uploads: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

@mcp.tool()
async def get_upload(project_id: str, upload_id: str) -> Dict[str, Any]:
    """Get details for a specific upload.
    
    Args:
        project_id: Project ID
        upload_id: Upload ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()
    
    try:
        upload = await _run_sync(client.get_upload, project_id, upload_id)
        return {
            "status": "success",
            "upload": upload
        }
    except Exception as e:
        logger.error(f"Error getting upload: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }

@mcp.tool()
async def get_todolist(project_id: str, todolist_id: str) -> Dict[str, Any]:
    """Get a specific todo list by ID.

    Args:
        project_id: The project ID. Kept for backward compatibility; the request is scoped by the record's own ID alone.
        todolist_id: The todo list ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        todolist = await _run_sync(client.get_todolist, project_id, todolist_id)
        return {"status": "success", "todolist": todolist}
    except Exception as e:
        logger.error(f"Error getting todolist {todolist_id}: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {"error": "OAuth token expired", "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."}
        return {"error": "Execution error", "message": str(e)}


@mcp.tool()
async def create_todolist(
    project_id: str,
    name: str,
    description: str = "",
) -> Dict[str, Any]:
    """Create a new todo list in a project.

    Args:
        project_id: The project ID
        name: Todo list name
        description: Optional HTML description
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        todolist = await _run_sync(
            lambda: client.create_todolist(project_id, name, description if description else None)
        )
        return {"status": "success", "todolist": todolist}
    except Exception as e:
        logger.error(f"Error creating todolist: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {"error": "OAuth token expired", "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."}
        return {"error": "Execution error", "message": str(e)}


@mcp.tool()
async def update_todolist(
    project_id: str,
    todolist_id: str,
    name: str,
    description: str = "__NOT_SET__",
) -> Dict[str, Any]:
    """Update an existing todo list.

    The Basecamp API requires the name even when only updating the description.

    Args:
        project_id: The project ID
        todolist_id: The todo list ID
        name: Todo list name (required)
        description: Optional HTML description
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        desc_val = None if description == "__NOT_SET__" else description
        todolist = await _run_sync(
            lambda: client.update_todolist(project_id, todolist_id, name, desc_val)
        )
        return {"status": "success", "todolist": todolist}
    except Exception as e:
        logger.error(f"Error updating todolist {todolist_id}: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {"error": "OAuth token expired", "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."}
        return {"error": "Execution error", "message": str(e)}


@mcp.tool()
async def reposition_todolist(
    project_id: str, todolist_id: str, position: int
) -> Dict[str, Any]:
    """Reposition a to-do list within its to-do set.

    Args:
        project_id: The project ID. Kept for backward compatibility; the request is scoped by the record's own ID alone.
        todolist_id: The todo list ID
        position: New 1-based position among incomplete lists
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    if position < 1:
        return {"error": "Invalid input", "message": "position must be >= 1"}

    try:
        await _run_sync(
            lambda: client.reposition_todolist(project_id, todolist_id, position)
        )
        return {"status": "success", "message": f"Todolist {todolist_id} repositioned to position {position}"}
    except Exception as e:
        logger.error(f"Error repositioning todolist {todolist_id}: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {"error": "OAuth token expired", "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."}
        return {"error": "Execution error", "message": str(e)}


@mcp.tool()
async def trash_todolist(project_id: str, todolist_id: str) -> Dict[str, Any]:
    """Move a todo list to the trash.

    Trashed lists can be recovered from the Basecamp web UI within 30 days.

    Args:
        project_id: The project ID. Kept for backward compatibility; the request is scoped by the record's own ID alone.
        todolist_id: The todo list ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        await _run_sync(client.trash_todolist, project_id, todolist_id)
        return {"status": "success", "message": f"Todolist {todolist_id} moved to trash"}
    except Exception as e:
        logger.error(f"Error trashing todolist {todolist_id}: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {"error": "OAuth token expired", "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."}
        return {"error": "Execution error", "message": str(e)}


@mcp.tool()
async def get_todolist_groups(project_id: str, todolist_id: str) -> Dict[str, Any]:
    """Get all groups in a todo list.

    Groups are named sections within a todo list (e.g. "Phase 1", "Backlog").

    Args:
        project_id: The project ID
        todolist_id: The todo list ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        groups = await _run_sync(client.get_todolist_groups, project_id, todolist_id)
        return {"status": "success", "groups": groups, "count": len(groups)}
    except Exception as e:
        logger.error(f"Error getting todolist groups: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {"error": "OAuth token expired", "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."}
        return {"error": "Execution error", "message": str(e)}


@mcp.tool()
async def create_todolist_group(
    project_id: str,
    todolist_id: str,
    name: str,
    color: str = "",
) -> Dict[str, Any]:
    """Create a new group inside a todo list.

    Groups act as named sections to organise todos within a list.

    Args:
        project_id: The project ID
        todolist_id: The todo list ID
        name: Group name
        color: Optional color – one of: white, red, orange, yellow, green,
               blue, aqua, purple, gray, pink, brown
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        group = await _run_sync(
            lambda: client.create_todolist_group(project_id, todolist_id, name, color if color else None)
        )
        return {"status": "success", "group": group}
    except Exception as e:
        logger.error(f"Error creating todolist group: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {"error": "OAuth token expired", "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."}
        return {"error": "Execution error", "message": str(e)}


@mcp.tool()
async def reposition_todolist_group(
    project_id: str, group_id: str, position: int
) -> Dict[str, Any]:
    """Reposition a todo list group to a new location within its list.

    Args:
        project_id: The project ID
        group_id: The group ID
        position: New 1-based position
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    if position < 1:
        return {"error": "Invalid input", "message": "position must be >= 1"}

    try:
        await _run_sync(
            lambda: client.reposition_todolist_group(project_id, group_id, position)
        )
        return {"status": "success", "message": f"Group {group_id} repositioned to position {position}"}
    except Exception as e:
        logger.error(f"Error repositioning todolist group {group_id}: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {"error": "OAuth token expired", "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."}
        return {"error": "Execution error", "message": str(e)}


@mcp.tool()
async def search_projects(query: str) -> Dict[str, Any]:
    """Search projects by name or description.

    Args:
        query: Search query to match against project names and descriptions
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        search = BasecampSearch(client=client)
        projects = await _run_sync(search.search_projects, query)
        return {
            "status": "success",
            "query": query,
            "projects": projects,
            "count": len(projects)
        }
    except Exception as e:
        logger.error(f"Error searching projects: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {
                "error": "OAuth token expired",
                "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."
            }
        return {
            "error": "Execution error",
            "message": str(e)
        }


# My Assignments Tools
@mcp.tool()
async def get_my_assignments() -> Dict[str, Any]:
    """Get the current user's active assignments, grouped into priorities and non-priorities.

    Card table steps are normalized under their parent card as a "children" entry.
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        assignments = await _run_sync(client.get_my_assignments)
        return {"status": "success", "assignments": assignments}
    except Exception as e:
        logger.error(f"Error getting my assignments: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {"error": "OAuth token expired", "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."}
        return {"error": "Execution error", "message": str(e)}


@mcp.tool()
async def get_my_completed_assignments() -> Dict[str, Any]:
    """Get the current user's completed assignments (archived/trashed items excluded)."""
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        assignments = await _run_sync(client.get_my_completed_assignments)
        return {"status": "success", "assignments": assignments, "count": len(assignments)}
    except Exception as e:
        logger.error(f"Error getting completed assignments: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {"error": "OAuth token expired", "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."}
        return {"error": "Execution error", "message": str(e)}


@mcp.tool()
async def get_my_due_assignments(scope: str = "") -> Dict[str, Any]:
    """Get the current user's assignments filtered by due date scope.

    Args:
        scope: One of 'overdue', 'due_today', 'due_tomorrow', 'due_later_this_week',
            'due_next_week', 'due_later'. Defaults to 'overdue' when omitted.
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        assignments = await _run_sync(
            lambda: client.get_my_due_assignments(scope if scope else None)
        )
        return {"status": "success", "assignments": assignments, "count": len(assignments)}
    except Exception as e:
        logger.error(f"Error getting due assignments: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {"error": "OAuth token expired", "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."}
        return {"error": "Execution error", "message": str(e)}


@mcp.tool()
async def prioritize_assignment(recording_id: str) -> Dict[str, Any]:
    """Add a to-do, card, or card table step to "Up Next".

    Args:
        recording_id: The recording ID to prioritize. Use the item's top-level id
            for a to-do or card; for an unprioritized card table step use the
            step's own id from its parent card's "children".
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        await _run_sync(client.prioritize_assignment, recording_id)
        return {"status": "success", "message": f"Recording {recording_id} added to Up Next"}
    except Exception as e:
        logger.error(f"Error prioritizing assignment {recording_id}: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {"error": "OAuth token expired", "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."}
        return {"error": "Execution error", "message": str(e)}


@mcp.tool()
async def deprioritize_assignment(recording_id: str) -> Dict[str, Any]:
    """Remove a recording from "Up Next". Safe to call even if it isn't prioritized.

    Args:
        recording_id: The recording ID to deprioritize. For a prioritized card table
            step, use the step's own id (its parent card's priority_recording_id).
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        await _run_sync(client.deprioritize_assignment, recording_id)
        return {"status": "success", "message": f"Recording {recording_id} removed from Up Next"}
    except Exception as e:
        logger.error(f"Error deprioritizing assignment {recording_id}: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {"error": "OAuth token expired", "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."}
        return {"error": "Execution error", "message": str(e)}


@mcp.tool()
async def reorder_up_next(source_id: str, position: int) -> Dict[str, Any]:
    """Move an already-prioritized recording to a new position in "Up Next".

    Args:
        source_id: The recording ID to move
        position: New 1-based position within Up Next
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    if position < 1:
        return {"error": "Invalid input", "message": "position must be >= 1"}

    try:
        await _run_sync(
            lambda: client.reorder_up_next(source_id, position)
        )
        return {"status": "success", "message": f"Recording {source_id} moved to position {position} in Up Next"}
    except Exception as e:
        logger.error(f"Error reordering Up Next for {source_id}: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {"error": "OAuth token expired", "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."}
        return {"error": "Execution error", "message": str(e)}


# People Tools
@mcp.tool()
async def get_people() -> Dict[str, Any]:
    """Get all people visible to the current user across the account."""
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        people = await _run_sync(client.get_people)
        return {"status": "success", "people": people, "count": len(people)}
    except Exception as e:
        logger.error(f"Error getting people: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {"error": "OAuth token expired", "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."}
        return {"error": "Execution error", "message": str(e)}


@mcp.tool()
async def get_project_people(project_id: str) -> Dict[str, Any]:
    """Get all active people on a specific project.

    Args:
        project_id: The project ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        people = await _run_sync(client.get_project_people, project_id)
        return {"status": "success", "people": people, "count": len(people)}
    except Exception as e:
        logger.error(f"Error getting project people for {project_id}: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {"error": "OAuth token expired", "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."}
        return {"error": "Execution error", "message": str(e)}


@mcp.tool()
async def update_project_access(project_id: str, grant=None, revoke=None, create=None) -> Dict[str, Any]:
    """Grant, revoke, or invite people to a project.

    Args:
        project_id: The project ID
        grant: Optional list of person IDs to grant access
        revoke: Optional list of person IDs to revoke access
        create: Optional list of new people to invite, each a dict with 'name' and
            'email_address', and optional 'title'/'company_name'
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    if grant is None and revoke is None and create is None:
        return {"error": "Invalid input", "message": "At least one of grant, revoke, or create must be provided"}

    try:
        result = await _run_sync(
            lambda: client.update_project_access(project_id, grant=grant, revoke=revoke, create=create)
        )
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Error updating project access for {project_id}: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {"error": "OAuth token expired", "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."}
        return {"error": "Execution error", "message": str(e)}


@mcp.tool()
async def get_pingable_people() -> Dict[str, Any]:
    """Get all people on the account who can be pinged."""
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        people = await _run_sync(client.get_pingable_people)
        return {"status": "success", "people": people, "count": len(people)}
    except Exception as e:
        logger.error(f"Error getting pingable people: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {"error": "OAuth token expired", "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."}
        return {"error": "Execution error", "message": str(e)}


@mcp.tool()
async def get_person(person_id: str) -> Dict[str, Any]:
    """Get a single person's profile by ID.

    Args:
        person_id: The person ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        person = await _run_sync(client.get_person, person_id)
        return {"status": "success", "person": person}
    except Exception as e:
        logger.error(f"Error getting person {person_id}: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {"error": "OAuth token expired", "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."}
        return {"error": "Execution error", "message": str(e)}


@mcp.tool()
async def get_my_profile() -> Dict[str, Any]:
    """Get the current user's personal profile info."""
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        profile = await _run_sync(client.get_my_profile)
        return {"status": "success", "profile": profile}
    except Exception as e:
        logger.error(f"Error getting my profile: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {"error": "OAuth token expired", "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."}
        return {"error": "Execution error", "message": str(e)}


@mcp.tool()
async def update_my_profile(
    name: str = "",
    email_address: str = "",
    title: str = "",
    bio: str = "",
    location: str = "",
    time_zone_name: str = "",
    first_week_day: int = -1,
    time_format: str = "",
) -> Dict[str, Any]:
    """Update the current user's personal profile info. Leave any field blank to keep it unchanged.

    Args:
        name: Display name
        email_address: Email address
        title: Job title
        bio: Short bio
        location: Location
        time_zone_name: Time zone, e.g. 'America/Chicago'
        first_week_day: 0 for Sunday, 1 for Monday. Leave as -1 to keep unchanged.
        time_format: Time display format
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        await _run_sync(
            lambda: client.update_my_profile(
                name=name if name else None,
                email_address=email_address if email_address else None,
                title=title if title else None,
                bio=bio if bio else None,
                location=location if location else None,
                time_zone_name=time_zone_name if time_zone_name else None,
                first_week_day=first_week_day if first_week_day >= 0 else None,
                time_format=time_format if time_format else None,
            )
        )
        return {"status": "success", "message": "Profile updated successfully"}
    except Exception as e:
        logger.error(f"Error updating my profile: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {"error": "OAuth token expired", "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."}
        return {"error": "Execution error", "message": str(e)}


@mcp.tool()
async def get_my_preferences() -> Dict[str, Any]:
    """Get the current user's preferences."""
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        preferences = await _run_sync(client.get_my_preferences)
        return {"status": "success", "preferences": preferences}
    except Exception as e:
        logger.error(f"Error getting my preferences: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {"error": "OAuth token expired", "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."}
        return {"error": "Execution error", "message": str(e)}


@mcp.tool()
async def update_my_preferences(
    time_zone_name: str = "", first_week_day: str = "", time_format: str = ""
) -> Dict[str, Any]:
    """Update the current user's preferences. Leave any field blank to keep it unchanged.

    Args:
        time_zone_name: Time zone name, e.g. 'America/Chicago', 'London', 'UTC'
        first_week_day: 'Sunday' through 'Saturday'
        time_format: 'twelve_hour' or 'twenty_four_hour'
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        preferences = await _run_sync(
            lambda: client.update_my_preferences(
                time_zone_name=time_zone_name if time_zone_name else None,
                first_week_day=first_week_day if first_week_day else None,
                time_format=time_format if time_format else None,
            )
        )
        return {"status": "success", "preferences": preferences}
    except Exception as e:
        logger.error(f"Error updating my preferences: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {"error": "OAuth token expired", "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."}
        return {"error": "Execution error", "message": str(e)}


# Project Management Tools
@mcp.tool()
async def create_project(name: str, description: str = "") -> Dict[str, Any]:
    """Create a new project.

    Args:
        name: Project name
        description: Optional project description
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        project = await _run_sync(
            lambda: client.create_project(name, description if description else None)
        )
        return {"status": "success", "project": project}
    except Exception as e:
        logger.error(f"Error creating project: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {"error": "OAuth token expired", "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."}
        return {"error": "Execution error", "message": str(e)}


@mcp.tool()
async def update_project(
    project_id: str,
    name: str,
    description: str = "__NOT_SET__",
    start_date: str = "",
    end_date: str = "",
    admissions: str = "",
) -> Dict[str, Any]:
    """Update a project's name, description, schedule, or access policy.

    Args:
        project_id: The project ID
        name: Project name (required by the API even when unchanged)
        description: Leave as __NOT_SET__ to keep unchanged; pass an empty string to clear it
        start_date: Project start date (ISO 8601). Requires end_date to also be set.
        end_date: Project end date (ISO 8601). Requires start_date to also be set.
        admissions: Access policy - one of 'invite', 'employee', 'team'
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        desc_val = None if description == "__NOT_SET__" else description
        project = await _run_sync(
            lambda: client.update_project(
                project_id, name,
                description=desc_val,
                start_date=start_date if start_date else None,
                end_date=end_date if end_date else None,
                admissions=admissions if admissions else None,
            )
        )
        return {"status": "success", "project": project}
    except Exception as e:
        logger.error(f"Error updating project {project_id}: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {"error": "OAuth token expired", "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."}
        return {"error": "Execution error", "message": str(e)}


@mcp.tool()
async def archive_project(project_id: str) -> Dict[str, Any]:
    """Archive a project.

    Args:
        project_id: The project ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        await _run_sync(client.archive_project, project_id)
        return {"status": "success", "message": f"Project {project_id} archived"}
    except Exception as e:
        logger.error(f"Error archiving project {project_id}: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {"error": "OAuth token expired", "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."}
        return {"error": "Execution error", "message": str(e)}


@mcp.tool()
async def unarchive_project(project_id: str) -> Dict[str, Any]:
    """Restore a project from the archive or trash to active.

    Args:
        project_id: The project ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        await _run_sync(client.unarchive_project, project_id)
        return {"status": "success", "message": f"Project {project_id} restored to active"}
    except Exception as e:
        logger.error(f"Error unarchiving project {project_id}: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {"error": "OAuth token expired", "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."}
        return {"error": "Execution error", "message": str(e)}


@mcp.tool()
async def trash_project(project_id: str) -> Dict[str, Any]:
    """Move a project to the trash. Trashed projects are deleted after 30 days.

    Args:
        project_id: The project ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        await _run_sync(client.trash_project, project_id)
        return {"status": "success", "message": f"Project {project_id} moved to trash"}
    except Exception as e:
        logger.error(f"Error trashing project {project_id}: {e}")
        if "401" in str(e) and "expired" in str(e).lower():
            return {"error": "OAuth token expired", "message": "Your Basecamp OAuth token expired during the API call. Please re-authenticate by visiting http://localhost:8000 and completing the OAuth flow again."}
        return {"error": "Execution error", "message": str(e)}


# 🎉 COMPLETE FastMCP server with ALL tools migrated!

if __name__ == "__main__":
    logger.info("Starting Basecamp FastMCP server")
    mcp.run(transport='http', host="0.0.0.0", port=8000)
