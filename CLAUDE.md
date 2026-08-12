# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a **Basecamp 5 MCP (Model Context Protocol) Server** that allows AI assistants (Cursor, Claude Desktop) to interact with Basecamp directly. It uses OAuth 2.0 for authentication and provides 100+ tools for Basecamp operations. The actively deployed server is `basecamp_fastmcp_http.py` (HTTP transport); `basecamp_fastmcp.py` (stdio) and `mcp_server_cli.py` (legacy JSON-RPC) exist as alternate transports but are not actively maintained.

## Development Commands

```bash
# Setup (one-time) - requires Python 3.10+
# Option 1: Using uv (recommended - auto-downloads Python 3.12)
uv venv --python 3.12 venv && source venv/bin/activate && uv pip install -r requirements.txt && uv pip install mcp

# Option 2: Using pip (if Python 3.10+ already installed)
python setup.py                      # Creates venv, installs deps, tests server

# OAuth Authentication
python oauth_app.py                  # Start OAuth server at http://localhost:8000

# Run the MCP server (for testing)
./venv/bin/python basecamp_fastmcp_http.py    # FastMCP HTTP server (actively deployed)
./venv/bin/python basecamp_fastmcp.py         # FastMCP stdio server
./venv/bin/python mcp_server_cli.py           # Legacy CLI server

# Test the server manually
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}
{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' | python basecamp_fastmcp.py

# Run tests
python -m pytest tests/ -v           # All tests
python -m pytest tests/test_cli_server.py -v  # Specific test file

# Generate client configs
python generate_cursor_config.py           # For Cursor IDE
python generate_claude_desktop_config.py   # For Claude Desktop
```

## Architecture

### Core Files

| File                       | Purpose                                                                             |
| -------------------------- | ------------------------------------------------------------------------------------ |
| `basecamp_fastmcp_http.py` | **Actively deployed MCP server** — FastMCP over HTTP transport (100+ tools)          |
| `basecamp_fastmcp.py`      | FastMCP server over stdio transport (same tool set as of the last stdio sync)        |
| `mcp_server_cli.py`        | Legacy JSON-RPC server (same tools, custom implementation)                           |
| `basecamp_client.py`  | Basecamp 3 API client - all HTTP methods and endpoints                    |
| `basecamp_oauth.py`   | OAuth 2.0 client for 37signals Launchpad                                  |
| `auth_manager.py`     | Automatic token refresh before API calls                                  |
| `token_storage.py`    | Thread-safe OAuth token persistence (`oauth_tokens.json`)                 |
| `search_utils.py`     | Cross-project search functionality                                        |
| `oauth_app.py`        | Flask app for OAuth flow (browser-based login)                            |

### Data Flow

```
MCP Client (Cursor/Claude)
    ↓ JSON-RPC via stdio
basecamp_fastmcp.py (MCP Server)
    ↓ calls
auth_manager.ensure_authenticated() → token_storage → basecamp_oauth.refresh_token()
    ↓ if valid
basecamp_client.py (API calls)
    ↓ HTTP requests
Basecamp 3 API (https://3.basecampapi.com/{account_id})
```

### Authentication Flow

1. User runs `python oauth_app.py` and visits `http://localhost:8000`
2. Redirected to 37signals for authorization
3. Callback stores tokens in `oauth_tokens.json` (600 permissions)
4. MCP server uses `auth_manager.ensure_authenticated()` to auto-refresh expired tokens

### Tool Categories (100+ total, basecamp_fastmcp_http.py)

- **Projects**: `get_projects`, `get_project`, `create_project`, `update_project`, `archive_project`, `unarchive_project`, `trash_project`
- **Todos**: `get_todosets`, `get_todoset`, `get_todolists`, `get_todolist`, `create_todolist`, `update_todolist`, `reposition_todolist`, `trash_todolist`, `get_todos`, `get_todo`, `create_todo`, `update_todo`, `delete_todo`, `complete_todo`, `uncomplete_todo`, `reposition_todo`, `archive_todo`
- **Todo List Groups**: `get_todolist_groups`, `create_todolist_group`, `reposition_todolist_group`
- **Card Tables (Kanban)**: `get_card_table`, `get_columns`, `get_cards`, `create_card`, `move_card`, `complete_card`, etc.
- **Card Steps**: `get_card_steps`, `create_card_step`, `complete_card_step`, etc.
- **Comments**: `get_comments`, `create_comment`
- **Messages**: `get_message_board`, `get_messages`, `get_message`, `get_message_categories`, `create_message`, `update_message`, `pin_message`, `unpin_message`, `trash_message`
- **My Assignments**: `get_my_assignments`, `get_my_completed_assignments`, `get_my_due_assignments`, `prioritize_assignment`, `deprioritize_assignment`, `reorder_up_next`
- **People**: `get_people`, `get_project_people`, `update_project_access`, `get_pingable_people`, `get_person`, `get_my_profile`, `update_my_profile`, `get_my_preferences`, `update_my_preferences`
- **Campfire (Chat)**: `get_campfire_lines`
- **Documents**: `get_documents`, `create_document`, `update_document`, `trash_document`
- **Inbox (Email Forwards)**: `get_inbox`, `get_forwards`, `get_forward`, `get_inbox_replies`, `get_inbox_reply`, `trash_forward`
- **Search**: `search_basecamp`, `get_search_metadata`, `global_search`, `search_projects`
- **Webhooks**: `get_webhooks`, `create_webhook`, `delete_webhook`
- **Other**: `get_daily_check_ins`, `get_question_answers`, `get_events`, `create_attachment`, `get_uploads`

## API Coverage vs bc-api Reference

Maps every doc in [basecamp/bc-api/sections](https://github.com/basecamp/bc-api/tree/master/sections) (the Basecamp 4 API reference) against `basecamp_fastmcp_http.py`'s tool set. **Full** = every documented endpoint has a tool. **Partial** = some endpoints covered, gap noted. **None** = zero tool coverage. **N/A** = conceptual doc, not a CRUD resource. Last audited 2026-08-12 (64 sections).

**Summary: 19 Full · 9 Partial · 33 None · 3 N/A**

### Projects & account

| Section | Status | Notes |
| --- | --- | --- |
| `projects.md` | Full | `get_projects`, `get_project`, `create_project`, `update_project`, `archive_project`, `unarchive_project`, `trash_project` |
| `basecamps.md` | N/A | Redirect stub — "Basecamps" were renamed to "Projects" |
| `account.md` | None | No get/update account name, upload/delete account logo |
| `templates.md` | None | No list/get/create/update/trash templates or create-project-from-template |
| `tools.md` | None | No get/create/update/enable/reposition/disable/trash dock tools |
| `folders.md` | None | No `/stacks.json` folder CRUD (project folders/stacks) |
| `everything.md` | None | Cross-project flat aggregators (`/todos/open.json`, `/cards/overdue.json`, etc.) uncovered |
| `events.md` | Full | `get_events` |
| `recordings.md` | Partial | Per-type archive/trash tools exist; generic `GET /projects/recordings.json` cross-type listing missing |
| `boosts.md` | None | No get/create/delete emoji boosts on recordings |
| `subscriptions.md` | None | No get/subscribe/unsubscribe/update-subscribers |
| `client_visibility.md` | None | No tool to toggle a recording's client visibility |

### To-dos

| Section | Status | Notes |
| --- | --- | --- |
| `todosets.md` | Full | `get_todosets`, `get_todoset` (no create/update/delete in the API) |
| `todolists.md` | Full | `get_todolists`, `get_todolist`, `create_todolist`, `update_todolist`, `reposition_todolist`, `trash_todolist` |
| `todolist_groups.md` | Full | `get_todolist_groups`, `create_todolist_group`, `reposition_todolist_group` |
| `todos.md` | Full | `get_todos`, `get_todo`, `create_todo`, `update_todo`, `complete_todo`, `uncomplete_todo`, `reposition_todo`, `delete_todo` |

### Card Table (Kanban)

| Section | Status | Notes |
| --- | --- | --- |
| `card_tables.md` | Full | `get_card_tables`, `get_card_table` |
| `card_table_columns.md` | Full | `get_column`, `create_column`, `update_column`, `move_column`, `watch_column`, `unwatch_column`, `put_column_on_hold`, `remove_column_hold`, `update_column_color` |
| `card_table_cards.md` | Full | `get_cards`, `get_card`, `create_card`, `update_card`, `move_card`, `complete_card`, `uncomplete_card` |
| `card_table_steps.md` | Partial | Get/create/update/complete/uncomplete/delete covered; reposition-a-step missing |
| `card_table_wormholes.md` | None | No create/update/delete wormholes (cross-linking cards between tables) |

### Messages & chat

| Section | Status | Notes |
| --- | --- | --- |
| `message_boards.md` | Full | `get_message_board` |
| `messages.md` | Full | `get_messages`, `get_message`, `create_message`, `update_message`, `pin_message`, `unpin_message`, `trash_message` |
| `message_types.md` | Partial | List-only (`get_message_categories`); missing get-single, create, update, delete |
| `campfires.md` | Partial | Only line-reading covered (`get_campfire_lines`); missing create/update/delete line, get single campfire/line, campfire uploads. List-all-campfires exists unexposed in `basecamp_client.py` (`get_campfires`) |
| `comments.md` | Partial | `get_comments`/`create_comment` covered; get-single and update exist unexposed in `basecamp_client.py` (`get_comment`, `update_comment`) |

### Documents & files (Vault)

| Section | Status | Notes |
| --- | --- | --- |
| `documents.md` | Full | `get_documents`, `get_document`, `create_document`, `update_document`, `trash_document` |
| `uploads.md` | Partial | `get_uploads`, `get_upload` only; missing create, update, version list, replace-version, trash |
| `vaults.md` | None | No list/get/create/update vaults |
| `google_documents.md` | None | No Google Docs/Sheets/Slides vault resource support |
| `cloud_files.md` | None | No Google Drive/Dropbox/Box cloud file support |
| `attachments.md` | Full | `create_attachment` |

### People & personal

| Section | Status | Notes |
| --- | --- | --- |
| `people.md` | Full | `get_people`, `get_project_people`, `update_project_access`, `get_pingable_people`, `get_person`, `get_my_profile`, `update_my_profile`, `get_my_preferences`, `update_my_preferences` |
| `my_assignments.md` | Full | `get_my_assignments`, `get_my_completed_assignments`, `get_my_due_assignments`, `prioritize_assignment`, `deprioritize_assignment`, `reorder_up_next` |
| `my_bookmarks.md` | None | No list/check/create/delete personal bookmarks |
| `my_notes.md` | None | No get/update the user's singleton personal note |
| `my_notifications.md` | None | No notifications inbox, bubble-ups, or mark-as-read |
| `out_of_office.md` | None | No get/set/remove a person's out-of-office status |

### Scheduling & check-ins

| Section | Status | Notes |
| --- | --- | --- |
| `schedules.md` | None | `get_schedule` exists unexposed in `basecamp_client.py`; update not implemented at all |
| `schedule_entries.md` | None | `get_schedule_entries` exists unexposed in `basecamp_client.py`; create/update/occurrence lookups missing entirely |
| `calendars.md` | None | No get calendar or update calendar color |
| `questionnaires.md` | None | No tool fetches the questionnaire resource itself |
| `questions.md` | Partial | List-only, via `get_daily_check_ins`; missing get/create/update/pause/resume/notification-settings |
| `question_answers.md` | Partial | List-only (`get_question_answers`); missing get single, answers-by-person filter, create/update/trash |
| `question_reminders.md` | None | No `GET /my/question_reminders.json` |
| `timesheets.md` | None | No account/project/recording timesheets or entry CRUD |
| `timeline.md` | None | No `/reports/progress.json` or per-person progress |

### Client-side (HEY-style client-facing) features

| Section | Status | Notes |
| --- | --- | --- |
| `client_approvals.md` | None | No get client approvals (list or single) |
| `client_correspondences.md` | None | No get client correspondences (list or single) |
| `client_replies.md` | None | No get client replies (list or single) |

### Inbox

| Section | Status | Notes |
| --- | --- | --- |
| `inboxes.md` | Full | `get_inbox` |
| `forwards.md` | Full | `get_forwards`, `get_forward`, `trash_forward` |
| `inbox_replies.md` | Full | `get_inbox_replies`, `get_inbox_reply` |
| `drafts.md` | None | No `GET /my/drafts.json` (unpublished messages/documents/uploads) |

### Search & reports

| Section | Status | Notes |
| --- | --- | --- |
| `search.md` | Full | `search_basecamp`, `get_search_metadata`, `global_search`, `search_projects` |
| `reports.md` | None | No assignable-people, overdue-todos, or upcoming-schedule reports |

### Automation & integrations

| Section | Status | Notes |
| --- | --- | --- |
| `webhooks.md` | Partial | `get_webhooks`, `create_webhook`, `delete_webhook`; missing get-single (with deliveries) and update |
| `chatbots.md` | None | No list/get/create/update/delete chatbots or post-as-chatbot |
| `external_links.md` | None | No dock/external-link list, get, create, rename, or trash |
| `lineup_markers.md` | None | No Lineup marker list/create/update/delete |
| `gauges.md` | None | No gauge/gauge-needle read, create, update, delete, or toggle |
| `hill_charts.md` | None | No get hill chart data or update hill settings |

### Conceptual docs (not CRUD resources)

| Section | Status | Notes |
| --- | --- | --- |
| `authentication.md` | N/A | OAuth 2.0 flow — implemented outside MCP tools via `basecamp_oauth.py`/`oauth_app.py`/`auth_manager.py` |
| `rich_text.md` | N/A | Formatting/parsing guide, not an endpoint |

## Key Patterns

### Adding New MCP Tools (FastMCP)

```python
# In basecamp_fastmcp.py
@mcp.tool()
async def new_tool_name(project_id: str, other_param: Optional[str] = None) -> Dict[str, Any]:
    """Tool description shown to AI.

    Args:
        project_id: The project ID
        other_param: Optional description
    """
    client = _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        result = await _run_sync(client.some_method, project_id, other_param)
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Error: {e}")
        return {"error": "Execution error", "message": str(e)}
```

### Adding Basecamp API Methods

```python
# In basecamp_client.py
def new_api_method(self, project_id, resource_id):
    """Method description."""
    endpoint = f'buckets/{project_id}/resource/{resource_id}.json'
    response = self.get(endpoint)  # or .post(), .put(), .delete(), .patch()
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed: {response.status_code} - {response.text}")
```

### Pagination Handling

Basecamp paginates list endpoints (~15 items/page). See `get_todos()` in `basecamp_client.py` for the pattern using `Link` header.

## Environment Configuration

Required in `.env`:

```bash
BASECAMP_CLIENT_ID=your_client_id
BASECAMP_CLIENT_SECRET=your_client_secret
BASECAMP_ACCOUNT_ID=your_account_id
BASECAMP_REDIRECT_URI=http://localhost:8000/auth/callback
USER_AGENT="Your App Name (your@email.com)"
```

The account ID can be found in your Basecamp URL: `https://3.basecamp.com/{account_id}/projects`

## Troubleshooting

- **Token expired**: Visit `http://localhost:8000` to re-authenticate (auto-refresh usually handles this)
- **Missing tools in Cursor/Claude**: Restart the client completely after config changes
- **Logs**: Check `basecamp_fastmcp.log` or `mcp_cli_server.log` for errors
- **Test token validity**: `python auth_manager.py` to force refresh check

## Reference

- API docs in `reference/bc3-api/sections/` - useful when implementing new endpoints
- Local queries/scripts go in `local_queries/` (git-ignored)
