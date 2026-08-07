# Basecamp 5 MCP Update — Design

Date: 2026-08-07
Status: Approved for implementation planning

## Goal

Bring the MCP server's coverage of nine Basecamp components — Message Boards, Messages,
My Assignments, People, Projects, Todos, Todolists, Todosets, Search — in line with the
current Basecamp 5 API, as documented at
[github.com/basecamp/bc-api/tree/master/sections](https://github.com/basecamp/bc-api/tree/master/sections).

The docs describe two route styles for most resources: legacy project-scoped routes
(`/buckets/{project_id}/...`, still fully supported) and newer canonical flat routes
(`/todos/2.json`, no bucket prefix). Some resources — My Assignments, Search — exist
*only* as flat routes; they're new in Basecamp 5 and have no legacy form.

## Scope

**Files touched:** `basecamp_client.py` (shared API methods), `basecamp_fastmcp_http.py`
(the HTTP-transport MCP server — the only server this project actively deploys), and
`CLAUDE.md` (documentation).

**Files explicitly NOT touched:** `basecamp_fastmcp.py` (stdio server), `mcp_server_cli.py`
(legacy JSON-RPC server), `search_utils.py` (its brute-force helpers keep powering
`global_search` and `search_projects`, which are out of scope for this pass).

Since `basecamp_client.py` is shared by all three servers, route-only migrations there
will transparently change which URL the stdio/CLI servers hit too (same response shape,
so no functional change for them) — but no *new* tools are added to those two servers.

## Signature stability

Existing tool signatures keep their current parameters — including `project_id` on
by-id operations (get/update/complete/delete/etc.) — even where the new flat route no
longer needs `project_id` to build the URL. Dropping a required param would silently
break any existing caller (saved Cursor/Claude Desktop configs, prompts) for zero
functional gain. Only the internal URL construction changes.

## HTTP-variant convention

`basecamp_fastmcp_http.py` does not use `Optional[X] = None` — it consistently uses
string/bool sentinels instead (`due_on: str = ""`, `notify: bool = False`,
`title: str = "__NOT_SET__"` where empty string is itself a valid value to send). Every
new tool in this plan follows that existing convention rather than introducing
`Optional`.

## Error handling & response codes

Every new/migrated `basecamp_client.py` method follows the pattern already documented
in `CLAUDE.md`:

```python
def method_name(self, ...):
    endpoint = f'...'
    response = self.get(endpoint)  # or post/put/delete
    if response.status_code == 200:  # or 201, or 204
        return response.json()  # or True for 204s
    else:
        raise Exception(f"Failed to X: {response.status_code} - {response.text}")
```

MCP tool wrappers keep the existing try/except → OAuth-expiry-detection →
`{"error": ..., "message": ...}` shape, unchanged.

---

## 1. Message Boards — route migration only

| Method | Current route | New route |
|---|---|---|
| `get_message_board(project_id)` | `buckets/{p}/message_boards/{id}.json` | `message_boards/{id}.json` |

No new tools — the API exposes only this one endpoint for message boards.

## 2. Messages

| Method | Current route | New route |
|---|---|---|
| `get_messages(project_id, message_board_id)` | `buckets/{p}/message_boards/{b}/messages.json` | `message_boards/{b}/messages.json` |
| `get_message(project_id, message_id)` | `buckets/{p}/messages/{id}.json` | `messages/{id}.json` |
| `create_message(...)` | `buckets/{p}/message_boards/{b}/messages.json` | `message_boards/{b}/messages.json` |

`create_message` gains optional params: `status` (`active`/`drafted`, defaults to
`active` to preserve current always-publish behavior), `subscriptions` (list of person
IDs), `visible_to_clients` (bool).

**New methods/tools:**
- `update_message(message_id, subject=None, content=None, category_id=None, subscriptions=None, notify=None)` → `PUT /messages/{id}.json`, 200 + JSON.
- `pin_message(message_id)` → `POST /recordings/{id}/pin.json`, 204.
- `unpin_message(message_id)` → `DELETE /recordings/{id}/pin.json`, 204.
- `trash_message(message_id)` → `PUT /recordings/{id}/status/trashed.json`, 204 (generic recording-trash route, explicitly listed under the Messages endpoint list in the docs).

## 3. My Assignments — entirely new

No existing coverage. All flat routes, all new to Basecamp 5.

- `get_my_assignments()` → `GET /my/assignments.json` (unpaginated; returns `priorities`/`non_priorities`).
- `get_my_completed_assignments()` → `GET /my/assignments/completed.json` (unpaginated).
- `get_my_due_assignments(scope="")` → `GET /my/assignments/due.json?scope=...`; valid scopes: `overdue`, `due_today`, `due_tomorrow`, `due_later_this_week`, `due_next_week`, `due_later`. Defaults to `overdue` server-side when omitted.
- `prioritize_assignment(recording_id)` → `POST /my/priorities.json` with `{"id": recording_id}`, 204.
- `deprioritize_assignment(recording_id)` → `DELETE /my/priorities/{recording_id}.json`, 204 (no-op-safe per docs).
- `reorder_up_next(source_id, position)` → `POST /my/priority_moves.json` with `{"source_id": ..., "position": ...}`, 204. Surface the documented 400/422 error bodies (`Position is required`, etc.) through the normal exception path.

Note for the tool docstrings: assigning/completing/rescheduling is *not* done through
these endpoints — that already happens via the existing `update_todo`, `complete_todo`,
`update_card`, `update_card_step` tools. This section is read + Up-Next-ordering only,
matching the "Act on assignments" guidance in the docs.

## 4. People

`client.get_people()` already exists but has never been exposed as an MCP tool.

**New tools (all new client methods too, except `get_people`):**
- `get_people()` → `GET /people.json` (expose existing method).
- `get_project_people(project_id)` → `GET /projects/{id}/people.json`.
- `update_project_access(project_id, grant=None, revoke=None, create=None)` → `PUT /projects/{id}/people/users.json` with whichever of `grant`/`revoke`/`create` are provided; 200 + JSON.
- `get_pingable_people()` → `GET /circles/people.json` (unpaginated per docs).
- `get_person(person_id)` → `GET /people/{id}.json`.
- `get_my_profile()` → `GET /my/profile.json`.
- `update_my_profile(name=None, email_address=None, title=None, bio=None, location=None, time_zone_name=None, first_week_day=None, time_format=None)` → `PUT /my/profile.json`, 204.
- `get_my_preferences()` → `GET /my/preferences.json`.
- `update_my_preferences(time_zone_name=None, first_week_day=None, time_format=None)` → `PUT /my/preferences.json` (params nested under a top-level `person` object per docs), 200 + JSON.

## 5. Projects

`get_projects`/`get_project` already use canonical flat routes — untouched.

**New methods/tools:**
- `create_project(name, description=None)` → `POST /projects.json`, 201 + JSON. Surface the documented `507 Insufficient Storage` (free-plan project limit) distinctly rather than as a generic failure.
- `update_project(project_id, name, description=None, start_date=None, end_date=None, admissions=None)` → `PUT /projects/{id}.json`, 200 + JSON. `start_date`/`end_date` map to `schedule_attributes[start_date/end_date]` and must be provided together. `admissions` ∈ `invite`/`employee`/`team`.
- `archive_project(project_id)` → `PUT /projects/{id}/status/archived.json`, 204.
- `unarchive_project(project_id)` → `PUT /projects/{id}/status/active.json`, 204 (same 507 handling as create).
- `trash_project(project_id)` → `DELETE /projects/{id}.json`, 204.

## 6. Todos — route migration only

| Method | New route |
|---|---|
| `get_todos(project_id, todolist_id)` | `todolists/{id}/todos.json` |
| `get_todo(project_id, todo_id)` | `todos/{id}.json` |
| `create_todo(...)` | `todolists/{id}/todos.json` |
| `update_todo(...)` | `todos/{id}.json` |
| `complete_todo` / `uncomplete_todo` | `todos/{id}/completion.json` |
| `reposition_todo` | `todos/{id}/position.json` |
| `delete_todo` (trash) | `recordings/{id}/status/trashed.json` |
| `archive_todo` | `recordings/{id}/status/archived.json` |

No new tools — functional coverage already complete.

## 7. Todolists (incl. groups) — route migration only

| Method | New route |
|---|---|
| `get_todolist(project_id, todolist_id)` | `todolists/{id}.json` |
| `create_todolist(project_id, name, ...)` | `todosets/{id}/todolists.json` |
| `update_todolist(...)` | `todolists/{id}.json` |
| `reposition_todolist(...)` | `todosets/todolists/{id}/position.json` |
| `trash_todolist(...)` | `recordings/{id}/status/trashed.json` |
| `get_todolist_groups(project_id, todolist_id)` | `todolists/{id}/groups.json` |
| `create_todolist_group(...)` | `todolists/{id}/groups.json` |
| `reposition_todolist_group(...)` | `todolists/groups/{id}/position.json` |

`get_todolists`'s "all todosets on a project" aggregation mode keeps its
dock-discovery loop; each per-todoset fetch inside that loop moves to the flat
`todosets/{id}/todolists.json` route. No new tools.

Note: `message_types.md` (message categories, used by `create_message`) has **no**
flat-route form in the docs — `get_message_categories` stays on
`buckets/{p}/categories.json` unchanged.

## 8. Todosets — route migration only

| Method | New route |
|---|---|
| `get_todoset(project_id, todoset_id)` (by-id form) | `todosets/{id}.json` |

The no-id "first todoset in project dock" fallback mode is unaffected (it doesn't call
this endpoint directly). No new tools.

## 9. Search

**New methods/tools:**
- `get_search_metadata()` → `GET /searches/metadata.json`. New tool `get_search_metadata`; used to discover valid `type_names[]`/`file_type` values before filtering.
- `search(query, type_names=None, bucket_ids=None, creator_ids=None, file_type=None, exclude_chat=None, since=None, sort=None, page=1, per_page=50)` → `GET /search.json`. Single page per call (search results aren't auto-aggregated like other list endpoints — `page`/`per_page` are exposed directly to the caller).

**Behavior change (approved):** the `search_basecamp` MCP tool is rewritten to call
`client.search(...)` instead of fanning out through `BasecampSearch`'s brute-force
per-resource filtering. New signature:
`search_basecamp(query, type_names="", bucket_ids="", creator_ids="", file_type="", exclude_chat=False, since="", sort="", page=1, per_page=50)`
(comma-separated strings for the array params, split before sending, matching this
file's no-`Optional` convention). This is a breaking change to `search_basecamp`'s
params and return shape — approved, since the brute-force version was materially
worse (N+1 calls, no relevance ranking, no cross-project filtering).

`global_search` and `search_projects` (both still backed by `search_utils.py`) are
unchanged.

## 10. Documentation

Update `CLAUDE.md`:
- Add `basecamp_fastmcp_http.py` to the Core Files table, with a note that it's the
  actively deployed HTTP-transport server.
- Update the tool count and Tool Categories section to include My Assignments and the
  expanded People/Projects/Search/Messages tools.

---

## Out of scope (explicitly, per this design)

- `basecamp_fastmcp.py` (stdio) and `mcp_server_cli.py` — no new tools, no route
  changes beyond what they inherit for free via shared `basecamp_client.py` methods.
- `search_utils.py`'s brute-force helpers, `global_search`, `search_projects` — unchanged.
- Message archive/unarchive (not listed as a Messages-specific endpoint in the docs;
  only "Trash a message" is).
- Creating a to-do directly under a todoset (bypassing a todolist) — a documented flat
  route exists (`POST /buckets/{p}/todosets/{id}/todos.json`) but isn't part of current
  coverage and wasn't requested; adding it would be scope creep.
- Card Tables, Campfire, Documents, Inbox, Webhooks, Comments, and all other components
  not in the named list of nine.

## Testing plan

No live Basecamp account/credentials are available in this environment, so
verification is necessarily static + structural, not end-to-end:

1. `python -m pytest tests/ -v` — confirm no regression in existing coverage
   (`test_card_tables.py`, `test_cli_server.py`).
2. Manual JSON-RPC stdin smoke test (per `CLAUDE.md`) against
   `basecamp_fastmcp_http.py` — confirm the server initializes cleanly and
   `tools/list` enumerates every new/changed tool with a valid schema.
3. Line-by-line review of every new/migrated endpoint URL, HTTP method, required
   params, and success status code against the fetched `bc-api` docs (already done
   during design; re-verified during implementation).

Actual live-call correctness (auth flow, real payload shapes, actual Basecamp account
behavior) cannot be verified without OAuth credentials against a real account — this
is flagged explicitly rather than implied as tested.
