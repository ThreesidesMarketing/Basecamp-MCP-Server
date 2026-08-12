# Tier 0 Coverage Gaps + External Links Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the highest-priority gaps identified in CLAUDE.md's "API Coverage vs bc-api Reference" table: expose four already-implemented-but-unwired `basecamp_client.py` capabilities (comments, campfire, schedule, schedule entries) as MCP tools, fixing real bugs uncovered along the way, and add net-new support for External Links (`external_links.md`).

**Architecture:** Every feature follows the codebase's existing two-layer pattern: a `basecamp_client.py` method that builds the endpoint and parses the response, plus a thin `@mcp.tool()` wrapper in `basecamp_fastmcp_http.py` (the only actively maintained server — `basecamp_fastmcp.py` and `mcp_server_cli.py` are explicitly out of scope per the user's decision). Client methods are unit-tested by mocking `BasecampClient.get/post/put/delete`; tool wrappers stay thin and are not separately tested unless they contain their own branching logic (matching the existing convention in `tests/`).

**Tech Stack:** Python 3.10+, `requests`, FastMCP, `unittest` + `unittest.mock`.

## Global Constraints

- Target file for new/changed `@mcp.tool()` functions: `basecamp_fastmcp_http.py` only. Do not touch `basecamp_fastmcp.py` or `mcp_server_cli.py`.
- Match the Basecamp 5 **flat-route** convention this codebase has been migrating to (see recent commits "Migrate Todosets get-by-id to flat route", "Replace brute-force search_basecamp with native /search.json API"). Prefer flat routes (`/comments/2.json`) over legacy bucket-scoped routes (`/buckets/1/comments/2.json`) wherever the bc-api docs offer a flat form.
- Every new/changed client method must raise `Exception(f"Failed to <verb> <noun>: {response.status_code} - {response.text}")` on non-2xx, matching the existing style throughout `basecamp_client.py`.
- Every new tool wrapper must follow the exact existing pattern: call `_get_basecamp_client()`, return `_get_auth_error_response()` if absent, wrap the client call in `try/except`, and on exception return the standard OAuth-expired branch plus a generic `{"error": "Execution error", "message": str(e)}` fallback (copy verbatim from `get_message` at `basecamp_fastmcp_http.py:858-886`).
- After all 5 tasks are done, update the `### Tool Categories` list and the `## API Coverage vs bc-api Reference` table in `CLAUDE.md` to reflect the new tools (Task 6).

---

## Task 1: Comments — expose `get_comment`, `update_comment`, `trash_comment`

**Context:** `basecamp_client.py` already has `get_comment`/`update_comment`/`delete_comment` (lines 1512–1567), but they (a) are never called by any `@mcp.tool()`, (b) use the legacy bucket-scoped route `buckets/{bucket_id}/comments/{comment_id}.json` instead of the canonical flat route `/comments/{id}.json`, and (c) `delete_comment` calls `DELETE buckets/{bucket_id}/comments/{comment_id}.json`, which does not exist in the bc-api reference at all — comments are trashed through the generic recording-status endpoint (`comments.md`'s "Trash a comment" links to `recordings.md#trash-a-recording`), the same mechanism already used by `trash_message` (`recordings/{id}/status/trashed.json`, see `basecamp_client.py:1246-1258`).

**Files:**
- Modify: `basecamp_client.py:1512-1567` (replace `get_comment`, `update_comment`, `delete_comment`)
- Modify: `basecamp_fastmcp_http.py` (insert 3 new tools between `create_comment` and `get_campfire_lines`, i.e. after line 764)
- Test: `tests/test_basecamp_client_v5.py` (add `TestComments` class)

**Interfaces:**
- Produces: `client.get_comment(project_id, comment_id) -> dict`, `client.update_comment(project_id, comment_id, content) -> dict`, `client.trash_comment(project_id, comment_id) -> bool`. `project_id` is kept in every signature for interface consistency with the rest of the codebase (see `get_message(project_id, message_id)`) even though the flat route doesn't use it.

- [ ] **Step 1: Write the failing client tests**

Add to `tests/test_basecamp_client_v5.py`, right after the `TestMessageBoards` class (before `class TestMessages`):

```python
class TestComments(unittest.TestCase):
    def setUp(self):
        self.client = make_client()

    @patch.object(BasecampClient, 'get')
    def test_get_comment_uses_flat_route(self, mock_get):
        mock_get.return_value = make_response(200, {"id": 2, "content": "Hi"})

        result = self.client.get_comment('999', '2')

        self.assertEqual(result, {"id": 2, "content": "Hi"})
        mock_get.assert_called_once_with('comments/2.json')

    @patch.object(BasecampClient, 'put')
    def test_update_comment_uses_flat_route(self, mock_put):
        mock_put.return_value = make_response(200, {"id": 2, "content": "Updated"})

        result = self.client.update_comment('999', '2', 'Updated')

        self.assertEqual(result, {"id": 2, "content": "Updated"})
        mock_put.assert_called_once_with('comments/2.json', {'content': 'Updated'})

    @patch.object(BasecampClient, 'put')
    def test_trash_comment_uses_recording_status_route(self, mock_put):
        mock_put.return_value = make_response(204)

        result = self.client.trash_comment('999', '2')

        self.assertTrue(result)
        mock_put.assert_called_once_with('recordings/2/status/trashed.json')
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_basecamp_client_v5.py::TestComments -v`
Expected: FAIL — `AttributeError` or assertion mismatch, since `get_comment`/`update_comment` still hit the old bucket-scoped endpoint and `trash_comment` doesn't exist yet.

- [ ] **Step 3: Replace the three client methods**

In `basecamp_client.py`, replace lines 1512-1567 (`get_comment` through `delete_comment`) with:

```python
    def get_comment(self, project_id, comment_id):
        """
        Get a specific comment.

        Args:
            project_id: The project ID. Kept for interface consistency; the
                flat route scopes by the comment's own ID alone.
            comment_id (int): Comment ID

        Returns:
            dict: Comment details
        """
        endpoint = f"comments/{comment_id}.json"
        response = self.get(endpoint)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get comment: {response.status_code} - {response.text}")

    def update_comment(self, project_id, comment_id, content):
        """
        Update a comment.

        Args:
            project_id: The project ID. Kept for interface consistency; the
                flat route scopes by the comment's own ID alone.
            comment_id (int): Comment ID
            content (str): New content for the comment in HTML format

        Returns:
            dict: Updated comment
        """
        endpoint = f"comments/{comment_id}.json"
        data = {"content": content}
        response = self.put(endpoint, data)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to update comment: {response.status_code} - {response.text}")

    def trash_comment(self, project_id, comment_id):
        """
        Trash a comment via the generic recording-status endpoint.

        Args:
            project_id: The project ID. Kept for interface consistency; the
                flat route scopes by the comment's own ID alone.
            comment_id (int): Comment ID

        Returns:
            bool: True if successful
        """
        endpoint = f"recordings/{comment_id}/status/trashed.json"
        response = self.put(endpoint)
        if response.status_code == 204:
            return True
        else:
            raise Exception(f"Failed to trash comment: {response.status_code} - {response.text}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_basecamp_client_v5.py::TestComments -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Add the three MCP tool wrappers**

In `basecamp_fastmcp_http.py`, insert immediately after `create_comment`'s closing block (after line 764, before the `@mcp.tool()` on line 766 for `get_campfire_lines`):

```python

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
```

- [ ] **Step 6: Run the full test suite**

Run: `python -m pytest tests/ -v`
Expected: All PASS, no regressions.

- [ ] **Step 7: Manual smoke test against real Basecamp**

With a valid OAuth token (`python oauth_app.py` if needed), start the server and call `get_comment`/`update_comment`/`trash_comment` against a real comment ID (create one via the existing `create_comment` tool first, or a comment ID from `get_comments`). Confirm the flat route resolves correctly (this is the main real-API risk in this task — `comments.md` documents the flat route but it's worth confirming against your account before trusting it in production).

- [ ] **Step 8: Commit**

```bash
git add basecamp_client.py basecamp_fastmcp_http.py tests/test_basecamp_client_v5.py
git commit -m "Add get_comment/update_comment/trash_comment tools; migrate comment endpoints to flat routes"
```

---

## Task 2: Campfire — fix and expose `get_campfire`

**Context:** `basecamp_client.py`'s existing `get_campfires(project_id)` (lines 1019-1025) calls `buckets/{project_id}/chats.json`, which is **not a real endpoint** — per `campfires.md`, the only list endpoint is the flat, account-wide `GET /chats.json` (no bucket scoping), and the only bucket-scoped route is `GET /buckets/1/chats/2.json` for a **single** campfire. Since every project has exactly one Campfire (discoverable via the project's `dock` array, same pattern as `get_message_board`), rename/rewrite this as `get_campfire` (singular) using the dock-discovery pattern.

**Files:**
- Modify: `basecamp_client.py:1019-1025` (replace `get_campfires` with `get_campfire`)
- Modify: `basecamp_fastmcp_http.py` (insert 1 new tool after `get_campfire_lines`, i.e. after line 795)
- Test: `tests/test_basecamp_client_v5.py` (add `TestCampfire` class)

**Interfaces:**
- Produces: `client.get_campfire(project_id) -> dict`, discovered via `project["dock"]` entry with `name == "chat"` (same discovery mechanism as `get_message_board`'s `"message_board"` dock name).

- [ ] **Step 1: Write the failing client test**

Add to `tests/test_basecamp_client_v5.py`, in a new `TestCampfire` class placed after `TestMessageBoards`:

```python
class TestCampfire(unittest.TestCase):
    def setUp(self):
        self.client = make_client()

    @patch.object(BasecampClient, 'get')
    def test_get_campfire_discovers_via_dock(self, mock_get):
        project_response = make_response(200, {
            "dock": [{"name": "chat", "id": 777}]
        })
        campfire_response = make_response(200, {"id": 777, "title": "Chat"})
        mock_get.side_effect = [project_response, campfire_response]

        result = self.client.get_campfire('999')

        self.assertEqual(result, {"id": 777, "title": "Chat"})
        second_call_endpoint = mock_get.call_args_list[1][0][0]
        self.assertEqual(second_call_endpoint, 'buckets/999/chats/777.json')

    @patch.object(BasecampClient, 'get')
    def test_get_campfire_raises_when_no_chat_in_dock(self, mock_get):
        mock_get.return_value = make_response(200, {"dock": [{"name": "message_board", "id": 1}]})

        with self.assertRaises(Exception) as ctx:
            self.client.get_campfire('999')

        self.assertIn('No campfire found', str(ctx.exception))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_basecamp_client_v5.py::TestCampfire -v`
Expected: FAIL — `AttributeError: 'BasecampClient' object has no attribute 'get_campfire'`

- [ ] **Step 3: Replace `get_campfires` with `get_campfire`**

In `basecamp_client.py`, replace lines 1019-1025 with:

```python
    def get_campfire(self, project_id):
        """Get the campfire (chat) for a project.

        The campfire ID is discovered from the project's dock array,
        following the same pattern as get_message_board().

        Args:
            project_id: Project/bucket ID

        Returns:
            dict: Campfire details including id, title, lines_url, etc.
        """
        project = self.get_project(project_id)
        try:
            dock_item = next(_ for _ in project["dock"] if _["name"] == "chat")
            chat_id = dock_item['id']
            response = self.get(f'buckets/{project_id}/chats/{chat_id}.json')
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"Failed to get campfire: {response.status_code} - {response.text}")
        except (IndexError, TypeError, StopIteration):
            raise Exception(f"No campfire found for project: {project_id}")
```

- [ ] **Step 4: Search for any other callers of `get_campfires` and update them**

Run: `grep -rn "get_campfires" --include="*.py" .`
Expected: only the definition itself (now renamed) and no other callers, since it was never wired to a tool. If any other caller turns up, update it to `get_campfire`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_basecamp_client_v5.py::TestCampfire -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Add the MCP tool wrapper**

In `basecamp_fastmcp_http.py`, insert immediately after `get_campfire_lines`'s closing block (after line 795, before the `@mcp.tool()` on line 797 for `get_message_board`):

```python

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
```

- [ ] **Step 7: Run the full test suite**

Run: `python -m pytest tests/ -v`
Expected: All PASS.

- [ ] **Step 8: Manual smoke test against real Basecamp**

Call `get_campfire` for a real project ID and confirm it returns the project's chat room details (not a 404 — this validates the `"chat"` dock name assumption against a live account).

- [ ] **Step 9: Commit**

```bash
git add basecamp_client.py basecamp_fastmcp_http.py tests/test_basecamp_client_v5.py
git commit -m "Fix get_campfires (was calling a non-existent endpoint); rename to get_campfire and add tool"
```

---

## Task 3: Schedule — fix and expose `get_schedule`

**Context:** `basecamp_client.py`'s `get_schedule(project_id)` (lines 1415-1421) calls `projects/{project_id}/schedule.json`, which does not exist in the bc-api reference. The real endpoints are flat `GET /schedules/:id.json` or legacy `GET /buckets/:id/schedules/:id.json` — the schedule ID must be discovered from the project's `dock` array (`name == "schedule"`), same pattern as `get_message_board`/`get_campfire`.

**Files:**
- Modify: `basecamp_client.py:1415-1421` (replace `get_schedule`)
- Modify: `basecamp_fastmcp_http.py` (insert 1 new tool after `get_events`, i.e. after line 2218)
- Test: `tests/test_basecamp_client_v5.py` (add `TestSchedule` class)

**Interfaces:**
- Produces: `client.get_schedule(project_id) -> dict`, with `dict['id']` as the schedule ID other methods (Task 4) depend on.

- [ ] **Step 1: Write the failing client test**

Add to `tests/test_basecamp_client_v5.py`, in a new `TestSchedule` class after `TestCampfire`:

```python
class TestSchedule(unittest.TestCase):
    def setUp(self):
        self.client = make_client()

    @patch.object(BasecampClient, 'get')
    def test_get_schedule_discovers_via_dock(self, mock_get):
        project_response = make_response(200, {
            "dock": [{"name": "schedule", "id": 444}]
        })
        schedule_response = make_response(200, {"id": 444, "title": "Calendar", "entries_count": 3})
        mock_get.side_effect = [project_response, schedule_response]

        result = self.client.get_schedule('999')

        self.assertEqual(result, {"id": 444, "title": "Calendar", "entries_count": 3})
        second_call_endpoint = mock_get.call_args_list[1][0][0]
        self.assertEqual(second_call_endpoint, 'buckets/999/schedules/444.json')

    @patch.object(BasecampClient, 'get')
    def test_get_schedule_raises_when_no_schedule_in_dock(self, mock_get):
        mock_get.return_value = make_response(200, {"dock": [{"name": "chat", "id": 1}]})

        with self.assertRaises(Exception) as ctx:
            self.client.get_schedule('999')

        self.assertIn('No schedule found', str(ctx.exception))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_basecamp_client_v5.py::TestSchedule -v`
Expected: FAIL — the current implementation calls `projects/999/schedule.json`, not the dock-discovered endpoint, so the mocked `side_effect` sequence and assertion mismatch.

- [ ] **Step 3: Replace `get_schedule`**

In `basecamp_client.py`, replace lines 1415-1421 with:

```python
    def get_schedule(self, project_id):
        """Get the schedule for a project.

        The schedule ID is discovered from the project's dock array,
        following the same pattern as get_message_board().

        Args:
            project_id: Project/bucket ID

        Returns:
            dict: Schedule details including id, title, entries_count, etc.
        """
        project = self.get_project(project_id)
        try:
            dock_item = next(_ for _ in project["dock"] if _["name"] == "schedule")
            schedule_id = dock_item['id']
            response = self.get(f'buckets/{project_id}/schedules/{schedule_id}.json')
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"Failed to get schedule: {response.status_code} - {response.text}")
        except (IndexError, TypeError, StopIteration):
            raise Exception(f"No schedule found for project: {project_id}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_basecamp_client_v5.py::TestSchedule -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Add the MCP tool wrapper**

In `basecamp_fastmcp_http.py`, insert immediately after `get_events`'s closing block (after line 2218, before the `@mcp.tool()` on line 2220 for `get_webhooks`):

```python

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
```

- [ ] **Step 6: Run the full test suite**

Run: `python -m pytest tests/ -v`
Expected: All PASS.

- [ ] **Step 7: Commit**

```bash
git add basecamp_client.py basecamp_fastmcp_http.py tests/test_basecamp_client_v5.py
git commit -m "Fix get_schedule (was calling a non-existent endpoint) and add tool"
```

---

## Task 4: Schedule entries — fix and expose `get_schedule_entries`

**Context:** `basecamp_client.py`'s `get_schedule_entries(project_id)` (lines 1423-1444) is fully broken: `self.get(...)` returns a `requests.Response` object (see `self.get`'s definition at line 74), never a parsed list, so `isinstance(schedule, list)` is always `False` and the method silently returns `[]` on every call — no exception, no data, just an empty list. It also calls `buckets/{project_id}/schedules.json`, which isn't a real "list of schedules" endpoint (each project has exactly one schedule). Fix it to reuse `get_schedule` (Task 3) for ID discovery, then fetch entries from the correct endpoint and actually parse the JSON.

**Files:**
- Modify: `basecamp_client.py` (replace `get_schedule_entries`, originally at lines 1423-1444 — line numbers will have shifted after Task 3's edit, locate by function name)
- Modify: `basecamp_fastmcp_http.py` (insert 1 new tool immediately after the new `get_schedule` tool from Task 3)
- Test: `tests/test_basecamp_client_v5.py` (add to `TestSchedule` class)

**Interfaces:**
- Consumes: `client.get_schedule(project_id) -> dict` with `dict['id']` (Task 3).
- Produces: `client.get_schedule_entries(project_id) -> list`.

- [ ] **Step 1: Write the failing client test**

Add to the `TestSchedule` class (created in Task 3) in `tests/test_basecamp_client_v5.py`:

```python
    @patch.object(BasecampClient, 'get')
    def test_get_schedule_entries_uses_schedule_id(self, mock_get):
        project_response = make_response(200, {"dock": [{"name": "schedule", "id": 444}]})
        schedule_response = make_response(200, {"id": 444, "title": "Calendar"})
        entries_response = make_response(200, [{"id": 1, "summary": "Team Meeting"}])
        mock_get.side_effect = [project_response, schedule_response, entries_response]

        result = self.client.get_schedule_entries('999')

        self.assertEqual(result, [{"id": 1, "summary": "Team Meeting"}])
        third_call_endpoint = mock_get.call_args_list[2][0][0]
        self.assertEqual(third_call_endpoint, 'buckets/999/schedules/444/entries.json')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_basecamp_client_v5.py::TestSchedule::test_get_schedule_entries_uses_schedule_id -v`
Expected: FAIL — the current implementation returns `[]` unconditionally (the `isinstance(schedule, list)` bug) rather than raising or matching the mocked sequence.

- [ ] **Step 3: Replace `get_schedule_entries`**

In `basecamp_client.py`, find and replace the existing `get_schedule_entries` method (originally lines 1423-1444, now directly below the `get_schedule` method rewritten in Task 3) with:

```python
    def get_schedule_entries(self, project_id):
        """
        Get schedule entries for a project.

        Args:
            project_id (int): Project ID

        Returns:
            list: Schedule entries
        """
        schedule = self.get_schedule(project_id)
        schedule_id = schedule['id']
        endpoint = f"buckets/{project_id}/schedules/{schedule_id}/entries.json"
        response = self.get(endpoint)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get schedule entries: {response.status_code} - {response.text}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_basecamp_client_v5.py::TestSchedule -v`
Expected: PASS (3 tests total in the class)

- [ ] **Step 5: Add the MCP tool wrapper**

In `basecamp_fastmcp_http.py`, insert immediately after the `get_schedule` tool added in Task 3 (and before `get_webhooks`):

```python

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
```

- [ ] **Step 6: Run the full test suite**

Run: `python -m pytest tests/ -v`
Expected: All PASS.

- [ ] **Step 7: Manual smoke test against real Basecamp**

Call `get_schedule` then `get_schedule_entries` for a real project with at least one calendar event, and confirm entries come back populated (this is the regression check that actually proves the `isinstance` bug is gone — the old code would have silently returned `count: 0` even with real data).

- [ ] **Step 8: Commit**

```bash
git add basecamp_client.py basecamp_fastmcp_http.py tests/test_basecamp_client_v5.py
git commit -m "Fix get_schedule_entries (silently returned [] due to unparsed Response) and add tool"
```

---

## Task 5: External Links — new resource (`external_links.md`)

**Context:** External links (dock tools historically called "doors") have zero client or tool coverage today. Per `external_links.md`: listing goes through the generic flat recordings endpoint filtered by `type=Door`; get/rename/trash go through the *generic dock-tool* endpoint `/dock/tools/:id.json` (shared with `tools.md`, no project scoping needed since IDs are global); create is bucket-scoped (`POST /buckets/:id/dock/doors.json`) and returns a bodyless `302` with no ID — the caller must re-list to find the new link's ID. There is no reliable JSON endpoint to change `url`/`service`/`description` after creation (documented as a known API gap — don't build it).

**Files:**
- Modify: `basecamp_client.py` (add 5 new methods near the end of the file, or grouped with a new `# External link methods` comment block — follow the existing per-resource grouping convention seen at `# Message board methods` etc.)
- Modify: `basecamp_fastmcp_http.py` (insert 5 new tools after `delete_webhook`, i.e. after line 2309, before the `# Document Management` comment on line 2311)
- Test: `tests/test_basecamp_client_v5.py` (add `TestExternalLinks` class)

**Interfaces:**
- Produces: `client.get_external_links(project_id=None, status=None) -> list`, `client.get_external_link(link_id) -> dict`, `client.create_external_link(project_id, service, url, title=None, description=None) -> bool`, `client.rename_external_link(link_id, title) -> dict`, `client.trash_external_link(link_id) -> bool`.

- [ ] **Step 1: Write the failing client tests**

Add to `tests/test_basecamp_client_v5.py`, in a new `TestExternalLinks` class after `TestSchedule`:

```python
class TestExternalLinks(unittest.TestCase):
    def setUp(self):
        self.client = make_client()

    @patch.object(BasecampClient, 'get')
    def test_get_external_links_scoped_to_project(self, mock_get):
        mock_get.return_value = make_response(200, [{"id": 1, "title": "Design system", "url": "https://figma.com/x"}])

        result = self.client.get_external_links(project_id='999')

        self.assertEqual(result, [{"id": 1, "title": "Design system", "url": "https://figma.com/x"}])
        endpoint = mock_get.call_args[0][0]
        params = mock_get.call_args[1]['params']
        self.assertEqual(endpoint, 'projects/recordings.json')
        self.assertEqual(params, {'type': 'Door', 'bucket': '999'})

    @patch.object(BasecampClient, 'get')
    def test_get_external_links_unscoped(self, mock_get):
        mock_get.return_value = make_response(200, [])

        self.client.get_external_links()

        params = mock_get.call_args[1]['params']
        self.assertEqual(params, {'type': 'Door'})

    @patch.object(BasecampClient, 'get')
    def test_get_external_link(self, mock_get):
        mock_get.return_value = make_response(200, {"id": 1, "title": "Design system"})

        result = self.client.get_external_link('1')

        self.assertEqual(result, {"id": 1, "title": "Design system"})
        mock_get.assert_called_once_with('dock/tools/1.json')

    @patch.object(BasecampClient, 'post')
    def test_create_external_link_required_fields_only(self, mock_post):
        mock_post.return_value = make_response(302)

        result = self.client.create_external_link('999', 'figma', 'https://figma.com/file/abc')

        self.assertTrue(result)
        endpoint, data = mock_post.call_args[0]
        self.assertEqual(endpoint, 'buckets/999/dock/doors.json')
        self.assertEqual(data, {'door': {'service': 'figma', 'url': 'https://figma.com/file/abc'}})

    @patch.object(BasecampClient, 'post')
    def test_create_external_link_with_optional_fields(self, mock_post):
        mock_post.return_value = make_response(200)  # requests follows the 302 by default

        self.client.create_external_link(
            '999', 'github', 'https://github.com/org/repo',
            title='Repo', description='<div>Main repo</div>',
        )

        _, data = mock_post.call_args[0]
        self.assertEqual(data['door']['title'], 'Repo')
        self.assertEqual(data['door']['description'], '<div>Main repo</div>')

    @patch.object(BasecampClient, 'put')
    def test_rename_external_link(self, mock_put):
        mock_put.return_value = make_response(200, {"id": 1, "title": "New name"})

        result = self.client.rename_external_link('1', 'New name')

        self.assertEqual(result, {"id": 1, "title": "New name"})
        mock_put.assert_called_once_with('dock/tools/1.json', {'title': 'New name'})

    @patch.object(BasecampClient, 'delete')
    def test_trash_external_link(self, mock_delete):
        mock_delete.return_value = make_response(204)

        result = self.client.trash_external_link('1')

        self.assertTrue(result)
        mock_delete.assert_called_once_with('dock/tools/1.json')
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_basecamp_client_v5.py::TestExternalLinks -v`
Expected: FAIL — `AttributeError`, none of these methods exist yet.

- [ ] **Step 3: Add the 5 client methods**

In `basecamp_client.py`, add near the end of the class (after the last existing method, before the class's closing — check with `tail -30 basecamp_client.py` for the exact insertion point):

```python
    # External link methods (dock tools historically called "doors")
    def get_external_links(self, project_id=None, status=None):
        """
        List external links via the flat, generic recordings endpoint.

        This is the only endpoint that returns an external link's full
        shape (url, service, description) -- get_external_link() omits them.

        Args:
            project_id: Optional project/bucket ID to scope to. Omit to
                list external links across every active project visible
                to the current user.
            status (str): Optional filter -- 'active', 'archived', or
                'trashed'.

        Returns:
            list: External links
        """
        params = {'type': 'Door'}
        if project_id:
            params['bucket'] = project_id
        if status:
            params['status'] = status
        response = self.get('projects/recordings.json', params=params)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get external links: {response.status_code} - {response.text}")

    def get_external_link(self, link_id):
        """
        Get a single external link via the generic dock-tool endpoint.

        Note: this envelope's `url` is the Basecamp redirector, not the
        outside address, and it omits `service`/`description`. Use
        get_external_links() when you need the full shape.

        Args:
            link_id: External link ID

        Returns:
            dict: External link details
        """
        response = self.get(f'dock/tools/{link_id}.json')
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get external link: {response.status_code} - {response.text}")

    def create_external_link(self, project_id, service, url, title=None, description=None):
        """
        Create an external link (dock 'door' tool) in a project.

        Args:
            project_id: Project/bucket ID
            service (str): Short service identifier -- e.g. 'figma',
                'dropbox', 'google_drive', 'github', 'notion', 'trello',
                'slack', 'zoom', or 'other'.
            url (str): HTTP/HTTPS address the link points to.
            title (str): Optional display name. Defaults to "Untitled".
            description (str): Optional rich-text (HTML) description.

        Returns:
            bool: True if the link was created.

        Note: Basecamp returns a bodyless 302 on success with no ID --
        call get_external_links(project_id) afterward to find the new
        link.
        """
        door = {'service': service, 'url': url}
        if title is not None:
            door['title'] = title
        if description is not None:
            door['description'] = description
        endpoint = f'buckets/{project_id}/dock/doors.json'
        response = self.post(endpoint, {'door': door})
        if response.status_code in (200, 302):
            return True
        else:
            raise Exception(f"Failed to create external link: {response.status_code} - {response.text}")

    def rename_external_link(self, link_id, title):
        """
        Rename an external link via the generic dock-tool endpoint.

        Only the title can be changed this way -- there is no reliable
        JSON endpoint to change url/service/description after creation.

        Args:
            link_id: External link ID
            title (str): New display name

        Returns:
            dict: Updated external link
        """
        response = self.put(f'dock/tools/{link_id}.json', {'title': title})
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to rename external link: {response.status_code} - {response.text}")

    def trash_external_link(self, link_id):
        """
        Trash (soft-delete) an external link via the generic dock-tool
        endpoint.

        Args:
            link_id: External link ID

        Returns:
            bool: True if successful
        """
        response = self.delete(f'dock/tools/{link_id}.json')
        if response.status_code == 204:
            return True
        else:
            raise Exception(f"Failed to trash external link: {response.status_code} - {response.text}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_basecamp_client_v5.py::TestExternalLinks -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Add the 5 MCP tool wrappers**

In `basecamp_fastmcp_http.py`, insert after `delete_webhook`'s closing block (after line 2309), before the `# Document Management` comment on line 2311:

```python

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
```

Confirm `Optional` is already imported at the top of `basecamp_fastmcp_http.py` (it is used by other existing tools, e.g. `get_message_categories` or similar) before relying on it here.

- [ ] **Step 6: Run the full test suite**

Run: `python -m pytest tests/ -v`
Expected: All PASS.

- [ ] **Step 7: Manual smoke test against real Basecamp**

Create a real external link via `create_external_link` against a test project, then `get_external_links(project_id=...)` to find its ID, `get_external_link(link_id)` to fetch it, `rename_external_link` to rename it, and `trash_external_link` to clean up. This end-to-end pass is the main validation that the `302`-response handling and `dock/tools/:id.json` routing actually work — they're the least conventional parts of this codebase's API surface so far.

- [ ] **Step 8: Commit**

```bash
git add basecamp_client.py basecamp_fastmcp_http.py tests/test_basecamp_client_v5.py
git commit -m "Add External Links support (list/get/create/rename/trash)"
```

---

## Task 6: Update CLAUDE.md coverage table and tool list

**Files:**
- Modify: `CLAUDE.md` (`### Tool Categories` section and the `## API Coverage vs bc-api Reference` table)

- [ ] **Step 1: Update the Tool Categories list**

In `CLAUDE.md`, under `### Tool Categories`, add the 9 new tool names to their relevant bullets (or add new bullets):
- **Comments**: append `get_comment`, `update_comment`, `trash_comment` to the existing bullet
- **Campfire (Chat)**: append `get_campfire`
- New bullet: **Schedule**: `get_schedule`, `get_schedule_entries`
- New bullet: **External Links**: `get_external_links`, `get_external_link`, `create_external_link`, `rename_external_link`, `trash_external_link`

- [ ] **Step 2: Update the coverage table statuses**

In the `## API Coverage vs bc-api Reference` section:
- `comments.md`: change status to **Full**, update notes to reflect all 4 endpoints now covered
- `campfires.md`: keep **Partial** (still missing create/update/delete line, campfire uploads), update notes to remove the "list-all-campfires" gap since `get_campfire` now exists
- `schedules.md`: change status to **Partial** (get covered; update-a-schedule still missing)
- `schedule_entries.md`: change status to **Partial** (list covered; get-single/create/update/occurrence-lookup still missing)
- `external_links.md`: change status to **Partial** (list/get/create/rename/trash covered; "change URL/service/description" has no reliable API per bc-api docs, so it's permanently out of scope, not a gap)
- Update the summary line counts (`Full`/`Partial`/`None`/`N/A`) accordingly

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "Update API coverage table for Tier 0 + External Links tools"
```

---

## Self-Review Notes (from plan authoring)

- **Spec coverage:** All 5 features from the user-approved priority list (4x Tier 0 cheap wins + external_links) have a task. Task 6 keeps CLAUDE.md in sync, matching the "why" behind writing the coverage table in the first place.
- **Known real-API risks to watch during Step 7 manual smoke tests:** the `"chat"` and `"schedule"` dock item names (assumed from Basecamp's well-known dock vocabulary, not directly confirmed against this account); whether `requests`' default redirect-following turns the External Link create `302` into a `200` in practice (the `in (200, 302)` check handles both, but confirm once against a live call); and whether the flat `/comments/{id}.json` route is actually enabled on this account (some older documented flat routes lag behind an account's real rollout).
