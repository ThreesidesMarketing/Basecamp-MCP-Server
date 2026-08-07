# Basecamp 5 MCP Update Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring `basecamp_fastmcp_http.py` (the actively deployed HTTP-transport MCP server) up to full Basecamp 5 API coverage for nine components — Message Boards, Messages, My Assignments, People, Projects, Todos, Todolists, Todosets, Search — per the design in `docs/superpowers/specs/2026-08-07-basecamp5-mcp-update-design.md`.

**Architecture:** Add/migrate methods in the shared `basecamp_client.py`, wire new/changed MCP tools in `basecamp_fastmcp_http.py` only, and unit-test the client layer by mocking `BasecampClient`'s low-level `get`/`post`/`put`/`delete` wrappers (no live Basecamp account is available). `basecamp_fastmcp.py` and `mcp_server_cli.py` are untouched.

**Tech Stack:** Python 3.10+, FastMCP 3.1.0 (`fastmcp` package), `requests`, `unittest`/`pytest`.

## Global Constraints

- Only these files change: `basecamp_client.py`, `basecamp_fastmcp_http.py`, `tests/test_basecamp_client_v5.py` (new), `CLAUDE.md`. Do not touch `basecamp_fastmcp.py` or `mcp_server_cli.py`.
- Every existing tool signature in `basecamp_fastmcp_http.py` keeps its current parameters (including `project_id` on by-id operations) even where the new flat route no longer needs it for URL-building.
- New tools follow this file's existing no-`Optional` convention: string params use `""` as the "not provided" sentinel (or `"__NOT_SET__"` where empty string is itself a meaningful value to send, matching the nearest existing analogous tool); list params default to `None` with no type annotation, matching `assignee_ids`/`completion_subscriber_ids` elsewhere in the file.
- Every `basecamp_client.py` method follows the existing pattern exactly: build `endpoint`, call `self.get/post/put/delete`, check `response.status_code`, return `response.json()` (200/201) or `True` (204), else `raise Exception(f"Failed to X: {response.status_code} - {response.text}")`.
- Every MCP tool wrapper in `basecamp_fastmcp_http.py` follows the existing try/except shape: on exception, log it, check for `"401" in str(e) and "expired" in str(e).lower()` for the OAuth-expiry message, else return `{"error": "Execution error", "message": str(e)}`.
- No live Basecamp credentials are available in this environment. Verification is: unit tests against mocked HTTP wrappers, `pytest tests/ -v` for regressions, and a FastMCP tool-registration smoke check (Task 10). Live-call correctness against a real account is out of scope for verification here.

---

### Task 1: Message Boards + Messages

**Files:**
- Modify: `basecamp_client.py:636-763` (`get_message_board`, `get_messages`, `get_message`, `create_message`)
- Modify: `basecamp_client.py` (insert new methods after `create_message`, before the Inbox methods comment)
- Modify: `basecamp_fastmcp_http.py:873-913` (`create_message` tool)
- Modify: `basecamp_fastmcp_http.py` (insert new tools after `create_message`, before the Inbox Tools comment at line 915)
- Create: `tests/test_basecamp_client_v5.py`

**Interfaces:**
- Produces: `BasecampClient.update_message(message_id, subject=None, content=None, category_id=None, subscriptions=None, notify=None) -> dict`, `BasecampClient.pin_message(message_id) -> bool`, `BasecampClient.unpin_message(message_id) -> bool`, `BasecampClient.trash_message(message_id) -> bool`. `create_message` gains `status=None, subscriptions=None, visible_to_clients=None` kwargs.

- [ ] **Step 1: Create the test file with the Message Boards + Messages tests**

Create `tests/test_basecamp_client_v5.py`:

```python
#!/usr/bin/env python3
"""Tests for Basecamp 5 client method updates: Message Boards, Messages,
My Assignments, People, Projects, Todos, Todolists, Todosets, and Search.

These tests mock BasecampClient's low-level get/post/put/delete wrappers
directly, so they verify each method builds the correct endpoint and
payload and handles the response correctly -- without making real HTTP
calls or requiring Basecamp credentials.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from basecamp_client import BasecampClient


def make_client():
    return BasecampClient(
        access_token='test-token',
        account_id='12345',
        user_agent='Test Agent',
        auth_mode='oauth',
    )


def make_response(status_code, json_data=None, headers=None):
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = json_data
    response.headers = headers or {}
    response.text = str(json_data)
    return response


class TestMessageBoards(unittest.TestCase):
    def setUp(self):
        self.client = make_client()

    @patch.object(BasecampClient, 'get')
    def test_get_message_board_uses_flat_route(self, mock_get):
        project_response = make_response(200, {
            "dock": [{"name": "message_board", "id": 555}]
        })
        board_response = make_response(200, {"id": 555, "title": "Message Board"})
        mock_get.side_effect = [project_response, board_response]

        result = self.client.get_message_board('999')

        self.assertEqual(result, {"id": 555, "title": "Message Board"})
        second_call_endpoint = mock_get.call_args_list[1][0][0]
        self.assertEqual(second_call_endpoint, 'message_boards/555.json')


class TestMessages(unittest.TestCase):
    def setUp(self):
        self.client = make_client()

    @patch.object(BasecampClient, 'get')
    def test_get_messages_uses_flat_route(self, mock_get):
        mock_get.return_value = make_response(200, [{"id": 1, "subject": "Hi"}])

        result = self.client.get_messages('999', message_board_id='555')

        self.assertEqual(result, [{"id": 1, "subject": "Hi"}])
        endpoint = mock_get.call_args[0][0]
        self.assertEqual(endpoint, 'message_boards/555/messages.json')

    @patch.object(BasecampClient, 'get')
    def test_get_message_uses_flat_route(self, mock_get):
        mock_get.return_value = make_response(200, {"id": 2, "subject": "Hi"})

        result = self.client.get_message('999', '2')

        self.assertEqual(result, {"id": 2, "subject": "Hi"})
        mock_get.assert_called_once_with('messages/2.json')

    @patch.object(BasecampClient, 'post')
    def test_create_message_uses_flat_route_and_defaults_active(self, mock_post):
        mock_post.return_value = make_response(201, {"id": 3, "subject": "New"})

        result = self.client.create_message('999', 'New', '<p>Body</p>', message_board_id='555')

        self.assertEqual(result, {"id": 3, "subject": "New"})
        endpoint, data = mock_post.call_args[0]
        self.assertEqual(endpoint, 'message_boards/555/messages.json')
        self.assertEqual(data['status'], 'active')

    @patch.object(BasecampClient, 'post')
    def test_create_message_supports_draft_and_visibility(self, mock_post):
        mock_post.return_value = make_response(201, {"id": 4})

        self.client.create_message(
            '999', 'Draft', '<p>Body</p>', message_board_id='555',
            status='drafted', subscriptions=[1, 2], visible_to_clients=True,
        )

        _, data = mock_post.call_args[0]
        self.assertEqual(data['status'], 'drafted')
        self.assertEqual(data['subscriptions'], [1, 2])
        self.assertTrue(data['visible_to_clients'])

    @patch.object(BasecampClient, 'put')
    def test_update_message(self, mock_put):
        mock_put.return_value = make_response(200, {"id": 2, "subject": "Updated"})

        result = self.client.update_message('2', subject='Updated')

        self.assertEqual(result, {"id": 2, "subject": "Updated"})
        endpoint, data = mock_put.call_args[0]
        self.assertEqual(endpoint, 'messages/2.json')
        self.assertEqual(data, {'subject': 'Updated'})

    def test_update_message_requires_at_least_one_field(self):
        with self.assertRaises(ValueError):
            self.client.update_message('2')

    @patch.object(BasecampClient, 'post')
    def test_pin_message(self, mock_post):
        mock_post.return_value = make_response(204)

        result = self.client.pin_message('2')

        self.assertTrue(result)
        mock_post.assert_called_once_with('recordings/2/pin.json')

    @patch.object(BasecampClient, 'delete')
    def test_unpin_message(self, mock_delete):
        mock_delete.return_value = make_response(204)

        result = self.client.unpin_message('2')

        self.assertTrue(result)
        mock_delete.assert_called_once_with('recordings/2/pin.json')

    @patch.object(BasecampClient, 'put')
    def test_trash_message(self, mock_put):
        mock_put.return_value = make_response(204)

        result = self.client.trash_message('2')

        self.assertTrue(result)
        mock_put.assert_called_once_with('recordings/2/status/trashed.json')


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `python -m pytest tests/test_basecamp_client_v5.py -v`
Expected: FAIL — `AttributeError: 'BasecampClient' object has no attribute 'update_message'` (and similar for `pin_message`/`unpin_message`/`trash_message`); the route-assertion tests fail because the current endpoints still include `buckets/{project_id}/`.

- [ ] **Step 3: Migrate `get_message_board`, `get_messages`, `get_message` to flat routes**

In `basecamp_client.py`, replace:

```python
            response = self.get(f'buckets/{project_id}/message_boards/{board_id}.json')
```

with:

```python
            response = self.get(f'message_boards/{board_id}.json')
```

Replace:

```python
        endpoint = f'buckets/{project_id}/message_boards/{message_board_id}/messages.json'

        all_messages = []
```

with:

```python
        endpoint = f'message_boards/{message_board_id}/messages.json'

        all_messages = []
```

Replace:

```python
        endpoint = f'buckets/{project_id}/messages/{message_id}.json'
        response = self.get(endpoint)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get message: {response.status_code} - {response.text}")
```

with:

```python
        endpoint = f'messages/{message_id}.json'
        response = self.get(endpoint)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get message: {response.status_code} - {response.text}")
```

- [ ] **Step 4: Migrate `create_message` to a flat route and add `status`/`subscriptions`/`visible_to_clients`**

Replace the entire `create_message` method body:

```python
    def create_message(self, project_id, subject, content, message_board_id=None, category_id=None):
        """Create a new message on a project's message board.

        Args:
            project_id: Project/bucket ID
            subject: Message title/subject
            content: Message body in HTML format
            message_board_id: Optional message board ID (auto-discovered if not provided)
            category_id: Optional message type/category ID

        Returns:
            dict: Created message details
        """
        if not message_board_id:
            message_board = self.get_message_board(project_id)
            message_board_id = message_board['id']

        endpoint = f'buckets/{project_id}/message_boards/{message_board_id}/messages.json'
        data = {'subject': subject, 'content': content, 'status': 'active'}
        if category_id is not None:
            data['category_id'] = category_id

        response = self.post(endpoint, data)
        if response.status_code == 201:
            return response.json()
        else:
            raise Exception(f"Failed to create message: {response.status_code} - {response.text}")
```

with:

```python
    def create_message(self, project_id, subject, content, message_board_id=None, category_id=None,
                        status=None, subscriptions=None, visible_to_clients=None):
        """Create a new message on a project's message board.

        Args:
            project_id: Project/bucket ID
            subject: Message title/subject
            content: Message body in HTML format
            message_board_id: Optional message board ID (auto-discovered if not provided)
            category_id: Optional message type/category ID
            status: Optional status - 'active' to publish immediately (default) or
                'drafted' to save as a draft that notifies no one until published
            subscriptions: Optional list of person IDs to notify and subscribe.
                If not provided, all people on the project will be notified.
            visible_to_clients: Optional bool - whether the message is visible to
                clients when the project has clients enabled

        Returns:
            dict: Created message details
        """
        if not message_board_id:
            message_board = self.get_message_board(project_id)
            message_board_id = message_board['id']

        endpoint = f'message_boards/{message_board_id}/messages.json'
        data = {'subject': subject, 'content': content, 'status': status or 'active'}
        if category_id is not None:
            data['category_id'] = category_id
        if subscriptions is not None:
            data['subscriptions'] = subscriptions
        if visible_to_clients is not None:
            data['visible_to_clients'] = visible_to_clients

        response = self.post(endpoint, data)
        if response.status_code == 201:
            return response.json()
        else:
            raise Exception(f"Failed to create message: {response.status_code} - {response.text}")
```

- [ ] **Step 5: Add `update_message`, `pin_message`, `unpin_message`, `trash_message` to `basecamp_client.py`**

Insert immediately after the `create_message` method (before the `# Inbox methods (Email Forwards)` comment):

```python

    def update_message(self, message_id, subject=None, content=None, category_id=None,
                        subscriptions=None, notify=None):
        """Update an existing message.

        Args:
            message_id: Message ID
            subject: Optional new title
            content: Optional new HTML body
            category_id: Optional new message type/category ID
            subscriptions: Optional list of person IDs to recompute subscribers.
                Omit both subscriptions and notify to keep current subscribers.
            notify: Optional bool - whether to notify newly added subscribers

        Returns:
            dict: The updated message
        """
        endpoint = f'messages/{message_id}.json'
        data = {}
        if subject is not None:
            data['subject'] = subject
        if content is not None:
            data['content'] = content
        if category_id is not None:
            data['category_id'] = category_id
        if subscriptions is not None:
            data['subscriptions'] = subscriptions
        if notify is not None:
            data['notify'] = notify

        if not data:
            raise ValueError("No fields provided to update")

        response = self.put(endpoint, data)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to update message: {response.status_code} - {response.text}")

    def pin_message(self, message_id):
        """Pin a message to the top of its message board.

        Args:
            message_id: Message ID

        Returns:
            bool: True if successful
        """
        endpoint = f'recordings/{message_id}/pin.json'
        response = self.post(endpoint)
        if response.status_code == 204:
            return True
        else:
            raise Exception(f"Failed to pin message: {response.status_code} - {response.text}")

    def unpin_message(self, message_id):
        """Unpin a message.

        Args:
            message_id: Message ID

        Returns:
            bool: True if successful
        """
        endpoint = f'recordings/{message_id}/pin.json'
        response = self.delete(endpoint)
        if response.status_code == 204:
            return True
        else:
            raise Exception(f"Failed to unpin message: {response.status_code} - {response.text}")

    def trash_message(self, message_id):
        """Move a message to the trash.

        Args:
            message_id: Message ID

        Returns:
            bool: True if successful
        """
        endpoint = f'recordings/{message_id}/status/trashed.json'
        response = self.put(endpoint)
        if response.status_code == 204:
            return True
        else:
            raise Exception(f"Failed to trash message: {response.status_code} - {response.text}")
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python -m pytest tests/test_basecamp_client_v5.py -v`
Expected: PASS — all `TestMessageBoards` and `TestMessages` tests green.

- [ ] **Step 7: Update the `create_message` tool and add the four new Messages tools in `basecamp_fastmcp_http.py`**

Replace the existing `create_message` tool:

```python
@mcp.tool()
async def create_message(project_id: str, subject: str, content: str,
                         message_board_id: str = "",
                         category_id: str = "") -> Dict[str, Any]:
    """Create a new message on a project's message board.

    Args:
        project_id: The project ID
        subject: Message title/subject
        content: Message body in HTML format
        message_board_id: Optional message board ID. If not provided, will be auto-discovered from the project.
        category_id: Optional message type/category ID
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        message = await _run_sync(
            lambda: client.create_message(
                project_id, subject, content,
                message_board_id=message_board_id if message_board_id else None,
                category_id=category_id if category_id else None
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
```

with:

```python
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
                visible_to_clients=visible_to_clients if visible_to_clients else None
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
            Omit both subscriptions and notify to keep the message's current subscribers.
        notify: Whether to notify newly added subscribers
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        message = await _run_sync(
            lambda: client.update_message(
                message_id,
                subject=subject if subject else None,
                content=content if content else None,
                category_id=category_id if category_id else None,
                subscriptions=subscriptions,
                notify=notify if notify else None
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
```

- [ ] **Step 8: Commit**

```bash
git add basecamp_client.py basecamp_fastmcp_http.py tests/test_basecamp_client_v5.py
git commit -m "$(cat <<'EOF'
Migrate Message Boards/Messages to flat routes, add update/pin/trash tools

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: My Assignments

**Files:**
- Modify: `basecamp_client.py` (insert new section before the `# People methods` comment at line 609)
- Modify: `basecamp_fastmcp_http.py` (append new tools before the `# 🎉 COMPLETE FastMCP server...` comment at the end of the file)
- Modify: `tests/test_basecamp_client_v5.py` (append `TestMyAssignments`)

**Interfaces:**
- Produces: `BasecampClient.get_my_assignments() -> dict`, `.get_my_completed_assignments() -> list`, `.get_my_due_assignments(scope=None) -> list`, `.prioritize_assignment(recording_id) -> bool`, `.deprioritize_assignment(recording_id) -> bool`, `.reorder_up_next(source_id, position) -> bool`.

- [ ] **Step 1: Append the failing tests**

Append to `tests/test_basecamp_client_v5.py` (before the `if __name__ == '__main__':` line):

```python
class TestMyAssignments(unittest.TestCase):
    def setUp(self):
        self.client = make_client()

    @patch.object(BasecampClient, 'get')
    def test_get_my_assignments(self, mock_get):
        mock_get.return_value = make_response(200, {"priorities": [], "non_priorities": []})

        result = self.client.get_my_assignments()

        self.assertEqual(result, {"priorities": [], "non_priorities": []})
        mock_get.assert_called_once_with('my/assignments.json')

    @patch.object(BasecampClient, 'get')
    def test_get_my_completed_assignments(self, mock_get):
        mock_get.return_value = make_response(200, [{"id": 1}])

        result = self.client.get_my_completed_assignments()

        self.assertEqual(result, [{"id": 1}])
        mock_get.assert_called_once_with('my/assignments/completed.json')

    @patch.object(BasecampClient, 'get')
    def test_get_my_due_assignments_with_scope(self, mock_get):
        mock_get.return_value = make_response(200, [{"id": 1}])

        result = self.client.get_my_due_assignments(scope='due_today')

        self.assertEqual(result, [{"id": 1}])
        mock_get.assert_called_once_with('my/assignments/due.json', params={'scope': 'due_today'})

    @patch.object(BasecampClient, 'get')
    def test_get_my_due_assignments_no_scope(self, mock_get):
        mock_get.return_value = make_response(200, [])

        self.client.get_my_due_assignments()

        mock_get.assert_called_once_with('my/assignments/due.json', params=None)

    @patch.object(BasecampClient, 'post')
    def test_prioritize_assignment(self, mock_post):
        mock_post.return_value = make_response(204)

        result = self.client.prioritize_assignment('123')

        self.assertTrue(result)
        mock_post.assert_called_once_with('my/priorities.json', {'id': '123'})

    @patch.object(BasecampClient, 'delete')
    def test_deprioritize_assignment(self, mock_delete):
        mock_delete.return_value = make_response(204)

        result = self.client.deprioritize_assignment('123')

        self.assertTrue(result)
        mock_delete.assert_called_once_with('my/priorities/123.json')

    @patch.object(BasecampClient, 'post')
    def test_reorder_up_next(self, mock_post):
        mock_post.return_value = make_response(204)

        result = self.client.reorder_up_next('123', 2)

        self.assertTrue(result)
        mock_post.assert_called_once_with('my/priority_moves.json', {'source_id': '123', 'position': 2})
```

- [ ] **Step 2: Run to verify the new tests fail**

Run: `python -m pytest tests/test_basecamp_client_v5.py::TestMyAssignments -v`
Expected: FAIL with `AttributeError: 'BasecampClient' object has no attribute 'get_my_assignments'` (and similarly for the other five methods).

- [ ] **Step 3: Add the My Assignments methods to `basecamp_client.py`**

Insert immediately before the `# People methods` comment (currently at line 609):

```python
    # My Assignments methods
    def get_my_assignments(self):
        """Get the current user's active assignments, grouped by priority.

        Returns:
            dict: Object with 'priorities' and 'non_priorities' assignment lists
        """
        response = self.get('my/assignments.json')
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get assignments: {response.status_code} - {response.text}")

    def get_my_completed_assignments(self):
        """Get the current user's completed assignments.

        Archived and trashed recordings are excluded.

        Returns:
            list: Completed assignments
        """
        response = self.get('my/assignments/completed.json')
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get completed assignments: {response.status_code} - {response.text}")

    def get_my_due_assignments(self, scope=None):
        """Get the current user's assignments filtered by due date scope.

        Args:
            scope (str, optional): One of 'overdue', 'due_today', 'due_tomorrow',
                'due_later_this_week', 'due_next_week', 'due_later'. Defaults to
                'overdue' server-side when omitted.

        Returns:
            list: Assignments due within the given scope
        """
        params = {'scope': scope} if scope else None
        response = self.get('my/assignments/due.json', params=params)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get due assignments: {response.status_code} - {response.text}")

    def prioritize_assignment(self, recording_id):
        """Add a recording to "Up Next", the current user's prioritized assignment list.

        Args:
            recording_id: The recording ID to prioritize (a to-do's or card's top-level
                id, or a not-yet-prioritized card table step's own id)

        Returns:
            bool: True if successful
        """
        response = self.post('my/priorities.json', {'id': recording_id})
        if response.status_code == 204:
            return True
        else:
            raise Exception(f"Failed to prioritize assignment: {response.status_code} - {response.text}")

    def deprioritize_assignment(self, recording_id):
        """Remove a recording from "Up Next".

        This is a no-op (still returns True) if the recording carries no priority,
        so a repeated call is always safe.

        Args:
            recording_id: The recording ID to deprioritize

        Returns:
            bool: True if successful
        """
        response = self.delete(f'my/priorities/{recording_id}.json')
        if response.status_code == 204:
            return True
        else:
            raise Exception(f"Failed to deprioritize assignment: {response.status_code} - {response.text}")

    def reorder_up_next(self, source_id, position):
        """Move an already-prioritized recording to a new position in "Up Next".

        Args:
            source_id: The recording ID to move
            position: The 1-based position to move it to

        Returns:
            bool: True if successful
        """
        response = self.post('my/priority_moves.json', {'source_id': source_id, 'position': position})
        if response.status_code == 204:
            return True
        else:
            raise Exception(f"Failed to reorder Up Next: {response.status_code} - {response.text}")

```

- [ ] **Step 4: Run to verify the tests pass**

Run: `python -m pytest tests/test_basecamp_client_v5.py::TestMyAssignments -v`
Expected: PASS — all 7 tests green.

- [ ] **Step 5: Add the six My Assignments tools to `basecamp_fastmcp_http.py`**

Replace:

```python
# 🎉 COMPLETE FastMCP server with ALL tools migrated!

if __name__ == "__main__":
```

with:

```python
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


# 🎉 COMPLETE FastMCP server with ALL tools migrated!

if __name__ == "__main__":
```

- [ ] **Step 6: Commit**

```bash
git add basecamp_client.py basecamp_fastmcp_http.py tests/test_basecamp_client_v5.py
git commit -m "$(cat <<'EOF'
Add My Assignments component (get/completed/due, Up Next prioritization)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: People

**Files:**
- Modify: `basecamp_client.py` (insert new section after `get_people` at line 616, before `# Campfire (chat) methods`)
- Modify: `basecamp_fastmcp_http.py` (append new tools at the end-of-file anchor)
- Modify: `tests/test_basecamp_client_v5.py` (append `TestPeople`)

**Interfaces:**
- Produces: `BasecampClient.get_project_people(project_id) -> list`, `.update_project_access(project_id, grant=None, revoke=None, create=None) -> dict`, `.get_pingable_people() -> list`, `.get_person(person_id) -> dict`, `.get_my_profile() -> dict`, `.update_my_profile(**fields) -> bool`, `.get_my_preferences() -> dict`, `.update_my_preferences(time_zone_name=None, first_week_day=None, time_format=None) -> dict`.

- [ ] **Step 1: Append the failing tests**

Append to `tests/test_basecamp_client_v5.py` (before `if __name__ == '__main__':`):

```python
class TestPeople(unittest.TestCase):
    def setUp(self):
        self.client = make_client()

    @patch.object(BasecampClient, 'get')
    def test_get_project_people(self, mock_get):
        mock_get.return_value = make_response(200, [{"id": 1}])

        result = self.client.get_project_people('999')

        self.assertEqual(result, [{"id": 1}])
        mock_get.assert_called_once_with('projects/999/people.json')

    @patch.object(BasecampClient, 'put')
    def test_update_project_access(self, mock_put):
        mock_put.return_value = make_response(200, {"granted": [1], "revoked": []})

        result = self.client.update_project_access('999', grant=[1])

        self.assertEqual(result, {"granted": [1], "revoked": []})
        mock_put.assert_called_once_with('projects/999/people/users.json', {'grant': [1]})

    def test_update_project_access_requires_a_field(self):
        with self.assertRaises(ValueError):
            self.client.update_project_access('999')

    @patch.object(BasecampClient, 'get')
    def test_get_pingable_people(self, mock_get):
        mock_get.return_value = make_response(200, [{"id": 1}])

        result = self.client.get_pingable_people()

        self.assertEqual(result, [{"id": 1}])
        mock_get.assert_called_once_with('circles/people.json')

    @patch.object(BasecampClient, 'get')
    def test_get_person(self, mock_get):
        mock_get.return_value = make_response(200, {"id": 2})

        result = self.client.get_person('2')

        self.assertEqual(result, {"id": 2})
        mock_get.assert_called_once_with('people/2.json')

    @patch.object(BasecampClient, 'get')
    def test_get_my_profile(self, mock_get):
        mock_get.return_value = make_response(200, {"id": 1})

        result = self.client.get_my_profile()

        self.assertEqual(result, {"id": 1})
        mock_get.assert_called_once_with('my/profile.json')

    @patch.object(BasecampClient, 'put')
    def test_update_my_profile(self, mock_put):
        mock_put.return_value = make_response(204)

        result = self.client.update_my_profile(name='New Name')

        self.assertTrue(result)
        mock_put.assert_called_once_with('my/profile.json', {'name': 'New Name'})

    def test_update_my_profile_requires_a_field(self):
        with self.assertRaises(ValueError):
            self.client.update_my_profile()

    @patch.object(BasecampClient, 'get')
    def test_get_my_preferences(self, mock_get):
        mock_get.return_value = make_response(200, {"time_zone_name": "UTC"})

        result = self.client.get_my_preferences()

        self.assertEqual(result, {"time_zone_name": "UTC"})
        mock_get.assert_called_once_with('my/preferences.json')

    @patch.object(BasecampClient, 'put')
    def test_update_my_preferences_nests_under_person(self, mock_put):
        mock_put.return_value = make_response(200, {"time_zone_name": "America/Chicago"})

        result = self.client.update_my_preferences(time_zone_name='America/Chicago')

        self.assertEqual(result, {"time_zone_name": "America/Chicago"})
        mock_put.assert_called_once_with('my/preferences.json', {'person': {'time_zone_name': 'America/Chicago'}})
```

- [ ] **Step 2: Run to verify the new tests fail**

Run: `python -m pytest tests/test_basecamp_client_v5.py::TestPeople -v`
Expected: FAIL — `AttributeError: 'BasecampClient' object has no attribute 'get_project_people'` (and similarly down the list).

- [ ] **Step 3: Add the People methods to `basecamp_client.py`**

Replace:

```python
    # People methods
    def get_people(self):
        """Get all people in the account."""
        response = self.get('people.json')
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get people: {response.status_code} - {response.text}")

    # Campfire (chat) methods
```

with:

```python
    # People methods
    def get_people(self):
        """Get all people in the account."""
        response = self.get('people.json')
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get people: {response.status_code} - {response.text}")

    def get_project_people(self, project_id):
        """Get all active people on a project.

        Args:
            project_id: Project ID

        Returns:
            list: People with access to the project
        """
        response = self.get(f'projects/{project_id}/people.json')
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get project people: {response.status_code} - {response.text}")

    def update_project_access(self, project_id, grant=None, revoke=None, create=None):
        """Grant, revoke, or create access to a project.

        Args:
            project_id: Project ID
            grant (list, optional): Person IDs to grant access
            revoke (list, optional): Person IDs to revoke access
            create (list, optional): New people to invite, each a dict with 'name'
                and 'email_address', and optional 'title'/'company_name'

        Returns:
            dict: The access-change result
        """
        data = {}
        if grant is not None:
            data['grant'] = grant
        if revoke is not None:
            data['revoke'] = revoke
        if create is not None:
            data['create'] = create

        if not data:
            raise ValueError("At least one of grant, revoke, or create must be provided")

        endpoint = f'projects/{project_id}/people/users.json'
        response = self.put(endpoint, data)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to update project access: {response.status_code} - {response.text}")

    def get_pingable_people(self):
        """Get all people on the account who can be pinged.

        Returns:
            list: Pingable people
        """
        response = self.get('circles/people.json')
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get pingable people: {response.status_code} - {response.text}")

    def get_person(self, person_id):
        """Get a single person's profile.

        Args:
            person_id: Person ID

        Returns:
            dict: The person's profile
        """
        response = self.get(f'people/{person_id}.json')
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get person: {response.status_code} - {response.text}")

    def get_my_profile(self):
        """Get the current user's personal info.

        Returns:
            dict: The current user's profile
        """
        response = self.get('my/profile.json')
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get my profile: {response.status_code} - {response.text}")

    def update_my_profile(self, name=None, email_address=None, title=None, bio=None,
                           location=None, time_zone_name=None, first_week_day=None,
                           time_format=None):
        """Update the current user's personal info.

        Args:
            name (str, optional): Display name
            email_address (str, optional): Email address
            title (str, optional): Job title
            bio (str, optional): Short bio
            location (str, optional): Location
            time_zone_name (str, optional): Time zone, e.g. 'America/Chicago'
            first_week_day (int, optional): 0 for Sunday, 1 for Monday
            time_format (str, optional): Time display format

        Returns:
            bool: True if successful
        """
        data = {}
        if name is not None:
            data['name'] = name
        if email_address is not None:
            data['email_address'] = email_address
        if title is not None:
            data['title'] = title
        if bio is not None:
            data['bio'] = bio
        if location is not None:
            data['location'] = location
        if time_zone_name is not None:
            data['time_zone_name'] = time_zone_name
        if first_week_day is not None:
            data['first_week_day'] = first_week_day
        if time_format is not None:
            data['time_format'] = time_format

        if not data:
            raise ValueError("No fields provided to update")

        response = self.put('my/profile.json', data)
        if response.status_code == 204:
            return True
        else:
            raise Exception(f"Failed to update my profile: {response.status_code} - {response.text}")

    def get_my_preferences(self):
        """Get the current user's preferences.

        Returns:
            dict: The current user's preferences
        """
        response = self.get('my/preferences.json')
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get my preferences: {response.status_code} - {response.text}")

    def update_my_preferences(self, time_zone_name=None, first_week_day=None, time_format=None):
        """Update the current user's preferences.

        Args:
            time_zone_name (str, optional): Time zone name, e.g. 'America/Chicago'
            first_week_day (str, optional): 'Sunday' through 'Saturday'
            time_format (str, optional): 'twelve_hour' or 'twenty_four_hour'

        Returns:
            dict: The updated preferences
        """
        person = {}
        if time_zone_name is not None:
            person['time_zone_name'] = time_zone_name
        if first_week_day is not None:
            person['first_week_day'] = first_week_day
        if time_format is not None:
            person['time_format'] = time_format

        if not person:
            raise ValueError("No fields provided to update")

        response = self.put('my/preferences.json', {'person': person})
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to update my preferences: {response.status_code} - {response.text}")

    # Campfire (chat) methods
```

- [ ] **Step 4: Run to verify the tests pass**

Run: `python -m pytest tests/test_basecamp_client_v5.py::TestPeople -v`
Expected: PASS — all 10 tests green.

- [ ] **Step 5: Add the nine People tools to `basecamp_fastmcp_http.py`**

Replace:

```python
# 🎉 COMPLETE FastMCP server with ALL tools migrated!

if __name__ == "__main__":
```

with:

```python
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


# 🎉 COMPLETE FastMCP server with ALL tools migrated!

if __name__ == "__main__":
```

- [ ] **Step 6: Commit**

```bash
git add basecamp_client.py basecamp_fastmcp_http.py tests/test_basecamp_client_v5.py
git commit -m "$(cat <<'EOF'
Add People component: expose get_people, add profile/preferences/access tools

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Projects

**Files:**
- Modify: `basecamp_client.py` (insert new section after `get_project` at line 148, before `# To-do list methods`)
- Modify: `basecamp_fastmcp_http.py` (append new tools at the end-of-file anchor)
- Modify: `tests/test_basecamp_client_v5.py` (append `TestProjects`)

**Interfaces:**
- Produces: `BasecampClient.create_project(name, description=None) -> dict`, `.update_project(project_id, name, description=None, start_date=None, end_date=None, admissions=None) -> dict`, `.archive_project(project_id) -> bool`, `.unarchive_project(project_id) -> bool`, `.trash_project(project_id) -> bool`.

- [ ] **Step 1: Append the failing tests**

Append to `tests/test_basecamp_client_v5.py` (before `if __name__ == '__main__':`):

```python
class TestProjects(unittest.TestCase):
    def setUp(self):
        self.client = make_client()

    @patch.object(BasecampClient, 'post')
    def test_create_project(self, mock_post):
        mock_post.return_value = make_response(201, {"id": 1, "name": "New"})

        result = self.client.create_project('New')

        self.assertEqual(result, {"id": 1, "name": "New"})
        mock_post.assert_called_once_with('projects.json', {'name': 'New'})

    @patch.object(BasecampClient, 'put')
    def test_update_project(self, mock_put):
        mock_put.return_value = make_response(200, {"id": 1, "name": "Renamed"})

        result = self.client.update_project('1', 'Renamed', admissions='team')

        self.assertEqual(result, {"id": 1, "name": "Renamed"})
        mock_put.assert_called_once_with('projects/1.json', {'name': 'Renamed', 'admissions': 'team'})

    @patch.object(BasecampClient, 'put')
    def test_update_project_schedule_dates_grouped_together(self, mock_put):
        mock_put.return_value = make_response(200, {"id": 1})

        self.client.update_project('1', 'Name', start_date='2026-01-01', end_date='2026-02-01')

        _, data = mock_put.call_args[0]
        self.assertEqual(data['schedule_attributes'], {'start_date': '2026-01-01', 'end_date': '2026-02-01'})

    @patch.object(BasecampClient, 'put')
    def test_archive_project(self, mock_put):
        mock_put.return_value = make_response(204)

        result = self.client.archive_project('1')

        self.assertTrue(result)
        mock_put.assert_called_once_with('projects/1/status/archived.json')

    @patch.object(BasecampClient, 'put')
    def test_unarchive_project(self, mock_put):
        mock_put.return_value = make_response(204)

        result = self.client.unarchive_project('1')

        self.assertTrue(result)
        mock_put.assert_called_once_with('projects/1/status/active.json')

    @patch.object(BasecampClient, 'delete')
    def test_trash_project(self, mock_delete):
        mock_delete.return_value = make_response(204)

        result = self.client.trash_project('1')

        self.assertTrue(result)
        mock_delete.assert_called_once_with('projects/1.json')
```

- [ ] **Step 2: Run to verify the new tests fail**

Run: `python -m pytest tests/test_basecamp_client_v5.py::TestProjects -v`
Expected: FAIL — `AttributeError: 'BasecampClient' object has no attribute 'create_project'` (and similarly down the list).

- [ ] **Step 3: Add the Projects methods to `basecamp_client.py`**

Replace:

```python
    def get_project(self, project_id):
        """Get a specific project by ID."""
        response = self.get(f'projects/{project_id}.json')
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get project: {response.status_code} - {response.text}")

    # To-do list methods
```

with:

```python
    def get_project(self, project_id):
        """Get a specific project by ID."""
        response = self.get(f'projects/{project_id}.json')
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get project: {response.status_code} - {response.text}")

    def create_project(self, name, description=None):
        """Create a new project.

        Args:
            name: Project name (required)
            description (str, optional): Project description

        Returns:
            dict: The created project
        """
        data = {'name': name}
        if description is not None:
            data['description'] = description
        response = self.post('projects.json', data)
        if response.status_code == 201:
            return response.json()
        else:
            raise Exception(f"Failed to create project: {response.status_code} - {response.text}")

    def update_project(self, project_id, name, description=None, start_date=None,
                        end_date=None, admissions=None):
        """Update a project's name, description, schedule, or access policy.

        Args:
            project_id: Project ID
            name: Project name (required by the API even when unchanged)
            description (str, optional): Project description
            start_date (str, optional): Project start date (ISO 8601). Requires end_date.
            end_date (str, optional): Project end date (ISO 8601). Requires start_date.
            admissions (str, optional): One of 'invite', 'employee', 'team'

        Returns:
            dict: The updated project
        """
        data = {'name': name}
        if description is not None:
            data['description'] = description
        if start_date is not None or end_date is not None:
            data['schedule_attributes'] = {'start_date': start_date, 'end_date': end_date}
        if admissions is not None:
            data['admissions'] = admissions

        response = self.put(f'projects/{project_id}.json', data)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to update project: {response.status_code} - {response.text}")

    def archive_project(self, project_id):
        """Archive a project.

        Args:
            project_id: Project ID

        Returns:
            bool: True if successful
        """
        response = self.put(f'projects/{project_id}/status/archived.json')
        if response.status_code == 204:
            return True
        else:
            raise Exception(f"Failed to archive project: {response.status_code} - {response.text}")

    def unarchive_project(self, project_id):
        """Restore a project from the archive or trash to active.

        Args:
            project_id: Project ID

        Returns:
            bool: True if successful
        """
        response = self.put(f'projects/{project_id}/status/active.json')
        if response.status_code == 204:
            return True
        else:
            raise Exception(f"Failed to unarchive project: {response.status_code} - {response.text}")

    def trash_project(self, project_id):
        """Move a project to the trash. Trashed projects are deleted after 30 days.

        Args:
            project_id: Project ID

        Returns:
            bool: True if successful
        """
        response = self.delete(f'projects/{project_id}.json')
        if response.status_code == 204:
            return True
        else:
            raise Exception(f"Failed to trash project: {response.status_code} - {response.text}")

    # To-do list methods
```

- [ ] **Step 4: Run to verify the tests pass**

Run: `python -m pytest tests/test_basecamp_client_v5.py::TestProjects -v`
Expected: PASS — all 6 tests green.

- [ ] **Step 5: Add the five Projects tools to `basecamp_fastmcp_http.py`**

Replace:

```python
# 🎉 COMPLETE FastMCP server with ALL tools migrated!

if __name__ == "__main__":
```

with:

```python
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
    description: str = "",
    start_date: str = "",
    end_date: str = "",
    admissions: str = "",
) -> Dict[str, Any]:
    """Update a project's name, description, schedule, or access policy.

    Args:
        project_id: The project ID
        name: Project name (required by the API even when unchanged)
        description: Optional project description
        start_date: Project start date (ISO 8601). Requires end_date to also be set.
        end_date: Project end date (ISO 8601). Requires start_date to also be set.
        admissions: Access policy - one of 'invite', 'employee', 'team'
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()

    try:
        project = await _run_sync(
            lambda: client.update_project(
                project_id, name,
                description=description if description else None,
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
```

- [ ] **Step 6: Commit**

```bash
git add basecamp_client.py basecamp_fastmcp_http.py tests/test_basecamp_client_v5.py
git commit -m "$(cat <<'EOF'
Add Projects component: create/update/archive/unarchive/trash tools

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Todos (route migration only)

**Files:**
- Modify: `basecamp_client.py:196-540` (`get_todos`, `get_todo`, `create_todo`, `update_todo`, `delete_todo`, `archive_todo`, `reposition_todo`, `complete_todo`, `uncomplete_todo`)
- Modify: `tests/test_basecamp_client_v5.py` (append `TestTodos`)

**Interfaces:**
- Consumes: nothing new. No `basecamp_fastmcp_http.py` changes — every Todos tool already calls these same method names with the same signatures; only the internal endpoint URLs change.

- [ ] **Step 1: Append the failing tests**

Append to `tests/test_basecamp_client_v5.py` (before `if __name__ == '__main__':`):

```python
class TestTodos(unittest.TestCase):
    def setUp(self):
        self.client = make_client()

    @patch.object(BasecampClient, 'get')
    def test_get_todos_uses_flat_route(self, mock_get):
        mock_get.return_value = make_response(200, [{"id": 1}])

        result = self.client.get_todos('999', '3')

        self.assertEqual(result, [{"id": 1}])
        endpoint = mock_get.call_args[0][0]
        self.assertEqual(endpoint, 'todolists/3/todos.json')

    @patch.object(BasecampClient, 'get')
    def test_get_todo_uses_flat_route(self, mock_get):
        mock_get.return_value = make_response(200, {"id": 2})

        result = self.client.get_todo('999', '2')

        self.assertEqual(result, {"id": 2})
        mock_get.assert_called_once_with('todos/2.json')

    @patch.object(BasecampClient, 'post')
    def test_create_todo_uses_flat_route(self, mock_post):
        mock_post.return_value = make_response(201, {"id": 3})

        result = self.client.create_todo('999', '3', 'New todo')

        self.assertEqual(result, {"id": 3})
        endpoint = mock_post.call_args[0][0]
        self.assertEqual(endpoint, 'todolists/3/todos.json')

    @patch.object(BasecampClient, 'put')
    def test_update_todo_uses_flat_route(self, mock_put):
        mock_put.return_value = make_response(200, {"id": 2})

        result = self.client.update_todo('999', '2', content='Updated')

        self.assertEqual(result, {"id": 2})
        endpoint = mock_put.call_args[0][0]
        self.assertEqual(endpoint, 'todos/2.json')

    @patch.object(BasecampClient, 'put')
    def test_delete_todo_uses_flat_recordings_route(self, mock_put):
        mock_put.return_value = make_response(204)

        result = self.client.delete_todo('999', '2')

        self.assertTrue(result)
        mock_put.assert_called_once_with('recordings/2/status/trashed.json')

    @patch.object(BasecampClient, 'put')
    def test_archive_todo_uses_flat_recordings_route(self, mock_put):
        mock_put.return_value = make_response(204)

        result = self.client.archive_todo('999', '2')

        self.assertTrue(result)
        mock_put.assert_called_once_with('recordings/2/status/archived.json')

    @patch.object(BasecampClient, 'put')
    def test_reposition_todo_uses_flat_route(self, mock_put):
        mock_put.return_value = make_response(204)

        result = self.client.reposition_todo('999', '2', 3)

        self.assertTrue(result)
        endpoint = mock_put.call_args[0][0]
        self.assertEqual(endpoint, 'todos/2/position.json')

    @patch.object(BasecampClient, 'post')
    def test_complete_todo_uses_flat_route(self, mock_post):
        mock_post.return_value = make_response(201, {"id": 2, "status": "completed"})

        result = self.client.complete_todo('999', '2')

        self.assertEqual(result, {"id": 2, "status": "completed"})
        mock_post.assert_called_once_with('todos/2/completion.json')

    @patch.object(BasecampClient, 'delete')
    def test_uncomplete_todo_uses_flat_route(self, mock_delete):
        mock_delete.return_value = make_response(204)

        result = self.client.uncomplete_todo('999', '2')

        self.assertTrue(result)
        mock_delete.assert_called_once_with('todos/2/completion.json')
```

- [ ] **Step 2: Run to verify the new tests fail**

Run: `python -m pytest tests/test_basecamp_client_v5.py::TestTodos -v`
Expected: FAIL — each assertion fails because the actual endpoint still contains `buckets/999/...`.

- [ ] **Step 3: Migrate the six distinct endpoint strings to their flat form**

In `basecamp_client.py`, replace (this exact string appears twice — in `get_todos` and `create_todo` — use a find-and-replace-all edit so both are updated identically):

```
buckets/{project_id}/todolists/{todolist_id}/todos.json
```

with:

```
todolists/{todolist_id}/todos.json
```

Replace (this exact string appears twice — in `get_todo` and `update_todo` — replace both):

```
buckets/{project_id}/todos/{todo_id}.json
```

with:

```
todos/{todo_id}.json
```

Replace (this exact string appears twice — in `complete_todo` and `uncomplete_todo` — replace both):

```
buckets/{project_id}/todos/{todo_id}/completion.json
```

with:

```
todos/{todo_id}/completion.json
```

Replace, in `delete_todo` only:

```
buckets/{project_id}/recordings/{todo_id}/status/trashed.json
```

with:

```
recordings/{todo_id}/status/trashed.json
```

Replace, in `archive_todo` only:

```
buckets/{project_id}/recordings/{todo_id}/status/archived.json
```

with:

```
recordings/{todo_id}/status/archived.json
```

Replace, in `reposition_todo` only:

```
buckets/{project_id}/todos/{todo_id}/position.json
```

with:

```
todos/{todo_id}/position.json
```

- [ ] **Step 4: Run to verify the tests pass**

Run: `python -m pytest tests/test_basecamp_client_v5.py::TestTodos -v`
Expected: PASS — all 9 tests green.

- [ ] **Step 5: Run the full existing suite to confirm no regressions**

Run: `python -m pytest tests/ -v`
Expected: PASS — `test_card_tables.py` and `test_cli_server.py` are unaffected since `mcp_server_cli.py`'s tools call the same `basecamp_client.py` methods and only assert on mocked return values, not URLs.

- [ ] **Step 6: Commit**

```bash
git add basecamp_client.py tests/test_basecamp_client_v5.py
git commit -m "$(cat <<'EOF'
Migrate Todos to canonical Basecamp 5 flat routes

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Todolists + Todolist Groups

**Files:**
- Modify: `basecamp_client.py:196-608` (`get_todolists`, `get_todolist`, `create_todolist`, `update_todolist`, `trash_todolist`, `get_todolist_groups`, `create_todolist_group`, `reposition_todolist_group`)
- Modify: `basecamp_client.py` (add new `reposition_todolist` method — see note below)
- Modify: `basecamp_fastmcp_http.py` (add new `reposition_todolist` tool after `update_todolist`, before `trash_todolist`)
- Modify: `tests/test_basecamp_client_v5.py` (append `TestTodolists`)

**Note:** The Basecamp 5 docs list "Reposition a to-do list" (`PUT /todosets/todolists/{id}/position.json`) as one of the five canonical Todolists endpoints, but the current codebase never implemented it — only `reposition_todolist_group` (a different, group-level operation) exists. This is a genuine coverage gap within the already-approved Todolists scope, not a new decision — it's added here alongside the route migration.

**Interfaces:**
- Produces: `BasecampClient.reposition_todolist(project_id, todolist_id, position) -> bool` (new). All other Todolists/groups methods keep their existing signatures.

- [ ] **Step 1: Append the failing tests**

Append to `tests/test_basecamp_client_v5.py` (before `if __name__ == '__main__':`):

```python
class TestTodolists(unittest.TestCase):
    def setUp(self):
        self.client = make_client()

    @patch.object(BasecampClient, 'get')
    def test_get_todolist_uses_flat_route(self, mock_get):
        mock_get.return_value = make_response(200, {"id": 2})

        result = self.client.get_todolist('999', '2')

        self.assertEqual(result, {"id": 2})
        mock_get.assert_called_once_with('todolists/2.json')

    @patch.object(BasecampClient, 'post')
    @patch.object(BasecampClient, 'get')
    def test_create_todolist_uses_flat_route(self, mock_get, mock_post):
        mock_get.return_value = make_response(200, {"dock": [{"name": "todoset", "id": 3}]})
        mock_post.return_value = make_response(201, {"id": 2})

        result = self.client.create_todolist('999', 'New list')

        self.assertEqual(result, {"id": 2})
        endpoint = mock_post.call_args[0][0]
        self.assertEqual(endpoint, 'todosets/3/todolists.json')

    @patch.object(BasecampClient, 'put')
    def test_update_todolist_uses_flat_route(self, mock_put):
        mock_put.return_value = make_response(200, {"id": 2})

        result = self.client.update_todolist('999', '2', 'Renamed')

        self.assertEqual(result, {"id": 2})
        mock_put.assert_called_once_with('todolists/2.json', {'name': 'Renamed'})

    @patch.object(BasecampClient, 'put')
    def test_reposition_todolist_uses_flat_route(self, mock_put):
        mock_put.return_value = make_response(204)

        result = self.client.reposition_todolist('999', '2', 1)

        self.assertTrue(result)
        mock_put.assert_called_once_with('todosets/todolists/2/position.json', {'position': 1})

    @patch.object(BasecampClient, 'put')
    def test_trash_todolist_uses_flat_route(self, mock_put):
        mock_put.return_value = make_response(204)

        result = self.client.trash_todolist('999', '2')

        self.assertTrue(result)
        mock_put.assert_called_once_with('recordings/2/status/trashed.json')

    @patch.object(BasecampClient, 'get')
    def test_get_todolist_groups_uses_flat_route(self, mock_get):
        mock_get.return_value = make_response(200, [{"id": 5}])

        result = self.client.get_todolist_groups('999', '2')

        self.assertEqual(result, [{"id": 5}])
        endpoint = mock_get.call_args[0][0]
        self.assertEqual(endpoint, 'todolists/2/groups.json')

    @patch.object(BasecampClient, 'post')
    def test_create_todolist_group_uses_flat_route(self, mock_post):
        mock_post.return_value = make_response(201, {"id": 5})

        result = self.client.create_todolist_group('999', '2', 'Phase 1')

        self.assertEqual(result, {"id": 5})
        endpoint = mock_post.call_args[0][0]
        self.assertEqual(endpoint, 'todolists/2/groups.json')

    @patch.object(BasecampClient, 'put')
    def test_reposition_todolist_group_uses_flat_route(self, mock_put):
        mock_put.return_value = make_response(204)

        result = self.client.reposition_todolist_group('999', '5', 1)

        self.assertTrue(result)
        mock_put.assert_called_once_with('todolists/groups/5/position.json', {'position': 1})
```

- [ ] **Step 2: Run to verify the new tests fail**

Run: `python -m pytest tests/test_basecamp_client_v5.py::TestTodolists -v`
Expected: FAIL — route-assertion failures for the migrated methods, and `AttributeError` for `reposition_todolist` (doesn't exist yet).

- [ ] **Step 3: Migrate `get_todolist`/`update_todolist` to a flat route**

Replace (this exact string appears twice — in `get_todolist` and `update_todolist` — replace both):

```
buckets/{project_id}/todolists/{todolist_id}.json
```

with:

```
todolists/{todolist_id}.json
```

- [ ] **Step 4: Migrate the todolist-collection route (`get_todolists` and `create_todolist`)**

This exact string appears three times in `basecamp_client.py` — twice inside `get_todolists` (the `todoset_id`-provided branch and the "all todosets" loop) and once inside `create_todolist`. Replace all three occurrences:

```
buckets/{project_id}/todosets/{todoset_id}/todolists.json
```

with:

```
todosets/{todoset_id}/todolists.json
```

- [ ] **Step 5: Migrate `trash_todolist` and add `reposition_todolist`**

Replace:

```python
    def trash_todolist(self, project_id, todolist_id):
        """Move a todolist to the trash.

        Args:
            project_id (str): Project ID
            todolist_id (str): Todolist ID

        Returns:
            bool: True if successful
        """
        endpoint = f'buckets/{project_id}/recordings/{todolist_id}/status/trashed.json'
        response = self.put(endpoint)
        if response.status_code == 204:
            return True
        else:
            raise Exception(f"Failed to trash todolist: {response.status_code} - {response.text}")
```

with:

```python
    def reposition_todolist(self, project_id, todolist_id, position):
        """Reposition a to-do list within its to-do set.

        Args:
            project_id (str): Project ID
            todolist_id (str): Todolist ID
            position (int): New 1-based position among incomplete lists

        Returns:
            bool: True if successful
        """
        endpoint = f'todosets/todolists/{todolist_id}/position.json'
        response = self.put(endpoint, {'position': position})
        if response.status_code == 204:
            return True
        else:
            raise Exception(f"Failed to reposition todolist: {response.status_code} - {response.text}")

    def trash_todolist(self, project_id, todolist_id):
        """Move a todolist to the trash.

        Args:
            project_id (str): Project ID
            todolist_id (str): Todolist ID

        Returns:
            bool: True if successful
        """
        endpoint = f'recordings/{todolist_id}/status/trashed.json'
        response = self.put(endpoint)
        if response.status_code == 204:
            return True
        else:
            raise Exception(f"Failed to trash todolist: {response.status_code} - {response.text}")
```

- [ ] **Step 6: Migrate `get_todolist_groups`/`create_todolist_group` and `reposition_todolist_group`**

Replace (this exact string appears twice — in `get_todolist_groups` and `create_todolist_group` — replace both):

```
buckets/{project_id}/todolists/{todolist_id}/groups.json
```

with:

```
todolists/{todolist_id}/groups.json
```

Replace, in `reposition_todolist_group`:

```
buckets/{project_id}/todolists/groups/{group_id}/position.json
```

with:

```
todolists/groups/{group_id}/position.json
```

- [ ] **Step 7: Run to verify the tests pass**

Run: `python -m pytest tests/test_basecamp_client_v5.py::TestTodolists -v`
Expected: PASS — all 8 tests green.

- [ ] **Step 8: Add the `reposition_todolist` tool to `basecamp_fastmcp_http.py`**

Replace:

```python
@mcp.tool()
async def trash_todolist(project_id: str, todolist_id: str) -> Dict[str, Any]:
```

with:

```python
@mcp.tool()
async def reposition_todolist(
    project_id: str, todolist_id: str, position: int
) -> Dict[str, Any]:
    """Reposition a to-do list within its to-do set.

    Args:
        project_id: The project ID
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
```

- [ ] **Step 9: Run the full existing suite to confirm no regressions**

Run: `python -m pytest tests/ -v`
Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add basecamp_client.py basecamp_fastmcp_http.py tests/test_basecamp_client_v5.py
git commit -m "$(cat <<'EOF'
Migrate Todolists/groups to flat routes, add missing reposition_todolist

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Todosets (route migration only)

**Files:**
- Modify: `basecamp_client.py:167-195` (`get_todoset`)
- Modify: `tests/test_basecamp_client_v5.py` (append `TestTodosets`)

**Interfaces:**
- No new methods, no `basecamp_fastmcp_http.py` changes.

- [ ] **Step 1: Append the failing test**

Append to `tests/test_basecamp_client_v5.py` (before `if __name__ == '__main__':`):

```python
class TestTodosets(unittest.TestCase):
    def setUp(self):
        self.client = make_client()

    @patch.object(BasecampClient, 'get')
    def test_get_todoset_by_id_uses_flat_route(self, mock_get):
        mock_get.return_value = make_response(200, {"id": 3})

        result = self.client.get_todoset('999', todoset_id='3')

        self.assertEqual(result, {"id": 3})
        mock_get.assert_called_once_with('todosets/3.json')
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_basecamp_client_v5.py::TestTodosets -v`
Expected: FAIL — the actual call is `buckets/999/todosets/3.json`.

- [ ] **Step 3: Migrate `get_todoset`'s by-id branch**

Replace:

```python
        if todoset_id:
            # Get specific todoset by ID
            endpoint = f'buckets/{project_id}/todosets/{todoset_id}.json'
```

with:

```python
        if todoset_id:
            # Get specific todoset by ID
            endpoint = f'todosets/{todoset_id}.json'
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_basecamp_client_v5.py::TestTodosets -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add basecamp_client.py tests/test_basecamp_client_v5.py
git commit -m "$(cat <<'EOF'
Migrate Todosets get-by-id to flat route

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Search

**Files:**
- Modify: `basecamp_client.py` (append `get_search_metadata` and `search` at the end of the file)
- Modify: `basecamp_fastmcp_http.py:157-198` (replace the existing `search_basecamp` tool; add `get_search_metadata` tool)
- Modify: `tests/test_basecamp_client_v5.py` (append `TestSearch`)

**Interfaces:**
- Produces: `BasecampClient.get_search_metadata() -> dict`, `.search(query, type_names=None, bucket_ids=None, creator_ids=None, file_type=None, exclude_chat=None, since=None, sort=None, page=1, per_page=50) -> list`.
- Breaking change (approved in the design spec): `search_basecamp`'s tool signature and return shape change from the old brute-force `{"results": {"todolists": ..., "todos": ..., ...}}` shape to `{"results": [...], "count": N}` backed by the real `/search.json` endpoint. `global_search` and `search_projects` are untouched.

- [ ] **Step 1: Append the failing tests**

Append to `tests/test_basecamp_client_v5.py` (before `if __name__ == '__main__':`):

```python
class TestSearch(unittest.TestCase):
    def setUp(self):
        self.client = make_client()

    @patch.object(BasecampClient, 'get')
    def test_get_search_metadata(self, mock_get):
        mock_get.return_value = make_response(200, {"recording_search_types": [], "file_search_types": []})

        result = self.client.get_search_metadata()

        self.assertEqual(result, {"recording_search_types": [], "file_search_types": []})
        mock_get.assert_called_once_with('searches/metadata.json')

    @patch.object(BasecampClient, 'get')
    def test_search_basic_query(self, mock_get):
        mock_get.return_value = make_response(200, [{"id": 1}])

        result = self.client.search('deploy')

        self.assertEqual(result, [{"id": 1}])
        endpoint = mock_get.call_args[0][0]
        params = mock_get.call_args[1]['params']
        self.assertEqual(endpoint, 'search.json')
        self.assertEqual(params, {'q': 'deploy', 'page': 1, 'per_page': 50})

    @patch.object(BasecampClient, 'get')
    def test_search_with_filters(self, mock_get):
        mock_get.return_value = make_response(200, [])

        self.client.search(
            'deploy', type_names=['Todo', 'Message'], bucket_ids=[1, 2],
            creator_ids=[9], file_type='pdf', exclude_chat=True,
            since='last_30_days', sort='recency', page=2, per_page=25,
        )

        params = mock_get.call_args[1]['params']
        self.assertEqual(params['type_names[]'], ['Todo', 'Message'])
        self.assertEqual(params['bucket_ids[]'], [1, 2])
        self.assertEqual(params['creator_ids[]'], [9])
        self.assertEqual(params['file_type'], 'pdf')
        self.assertEqual(params['exclude_chat'], 1)
        self.assertEqual(params['since'], 'last_30_days')
        self.assertEqual(params['sort'], 'recency')
        self.assertEqual(params['page'], 2)
        self.assertEqual(params['per_page'], 25)
```

- [ ] **Step 2: Run to verify the new tests fail**

Run: `python -m pytest tests/test_basecamp_client_v5.py::TestSearch -v`
Expected: FAIL — `AttributeError: 'BasecampClient' object has no attribute 'get_search_metadata'`.

- [ ] **Step 3: Add `get_search_metadata` and `search` to the end of `basecamp_client.py`**

Append after the final `get_upload` method (end of file):

```python

    # Search methods
    def get_search_metadata(self):
        """Get valid filter options for search (type_names[] and file_type values).

        Returns:
            dict: Available recording_search_types and file_search_types
        """
        response = self.get('searches/metadata.json')
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get search metadata: {response.status_code} - {response.text}")

    def search(self, query, type_names=None, bucket_ids=None, creator_ids=None,
               file_type=None, exclude_chat=None, since=None, sort=None,
               page=1, per_page=50):
        """Search recordings across the account using Basecamp's native search.

        Args:
            query: The search query string (required)
            type_names (list, optional): Recording types to include -- key values
                from get_search_metadata()'s recording_search_types
            bucket_ids (list, optional): Project IDs to filter by
            creator_ids (list, optional): Creator person IDs to filter by
            file_type (str, optional): Attachment file type -- a key value from
                get_search_metadata()'s file_search_types
            exclude_chat (bool, optional): Set True to exclude chat results
            since (str, optional): One of 'last_7_days', 'last_30_days',
                'last_90_days', 'last_12_months', 'forever' (default)
            sort (str, optional): 'best_match' (default) or 'recency'
            page (int, optional): Page number, default 1
            per_page (int, optional): Results per page, default 50

        Returns:
            list: Matching recordings for the requested page
        """
        params = {'q': query, 'page': page, 'per_page': per_page}
        if type_names is not None:
            params['type_names[]'] = type_names
        if bucket_ids is not None:
            params['bucket_ids[]'] = bucket_ids
        if creator_ids is not None:
            params['creator_ids[]'] = creator_ids
        if file_type is not None:
            params['file_type'] = file_type
        if exclude_chat is not None:
            params['exclude_chat'] = 1 if exclude_chat else 0
        if since is not None:
            params['since'] = since
        if sort is not None:
            params['sort'] = sort

        response = self.get('search.json', params=params)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to search: {response.status_code} - {response.text}")
```

- [ ] **Step 4: Run to verify the tests pass**

Run: `python -m pytest tests/test_basecamp_client_v5.py::TestSearch -v`
Expected: PASS — all 3 tests green.

- [ ] **Step 5: Replace the `search_basecamp` tool and add `get_search_metadata` in `basecamp_fastmcp_http.py`**

Replace:

```python
@mcp.tool()
async def search_basecamp(query: str, project_id: str = "") -> Dict[str, Any]:
    """Search across Basecamp projects, todos, and messages.
    
    Args:
        query: Search query
        project_id: Optional project ID to limit search scope
    """
    client = await _get_basecamp_client()
    if not client:
        return _get_auth_error_response()
    
    try:
        search = BasecampSearch(client=client)
        results = {}

        if project_id:
            # Search within specific project
            results["todolists"] = await _run_sync(search.search_todolists, query, project_id)
            results["todos"] = await _run_sync(search.search_todos, query, project_id)
        else:
            # Search across all projects
            results["projects"] = await _run_sync(search.search_projects, query)
            results["todos"] = await _run_sync(search.search_todos, query)
            results["messages"] = await _run_sync(search.search_messages, query)

        return {
            "status": "success",
            "query": query,
            "results": results
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
```

with:

```python
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
                exclude_chat=exclude_chat if exclude_chat else None,
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
            "count": len(results),
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
```

- [ ] **Step 6: Run the full existing suite to confirm no regressions**

Run: `python -m pytest tests/ -v`
Expected: PASS. `global_search` and `search_projects` tools are untouched and still import/use `BasecampSearch` from `search_utils.py` exactly as before.

- [ ] **Step 7: Commit**

```bash
git add basecamp_client.py basecamp_fastmcp_http.py tests/test_basecamp_client_v5.py
git commit -m "$(cat <<'EOF'
Replace brute-force search_basecamp with native /search.json API

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Documentation

**Files:**
- Modify: `CLAUDE.md`

**Interfaces:** None — documentation only.

- [ ] **Step 1: Update the Overview paragraph**

Replace:

```markdown
This is a **Basecamp 3 MCP (Model Context Protocol) Server** that allows AI assistants (Cursor, Claude Desktop) to interact with Basecamp directly. It uses OAuth 2.0 for authentication and provides 79 tools for Basecamp operations.
```

with:

```markdown
This is a **Basecamp 5 MCP (Model Context Protocol) Server** that allows AI assistants (Cursor, Claude Desktop) to interact with Basecamp directly. It uses OAuth 2.0 for authentication and provides 100+ tools for Basecamp operations. The actively deployed server is `basecamp_fastmcp_http.py` (HTTP transport); `basecamp_fastmcp.py` (stdio) and `mcp_server_cli.py` (legacy JSON-RPC) exist as alternate transports but are not actively maintained.
```

- [ ] **Step 2: Add `basecamp_fastmcp_http.py` to the Core Files table**

Replace:

```markdown
| File                  | Purpose                                                                   |
| --------------------- | ------------------------------------------------------------------------- |
| `basecamp_fastmcp.py` | **Main MCP server** using official Anthropic FastMCP framework (79 tools) |
| `mcp_server_cli.py`   | Legacy JSON-RPC server (same tools, custom implementation)                |
```

with:

```markdown
| File                       | Purpose                                                                             |
| -------------------------- | ------------------------------------------------------------------------------------ |
| `basecamp_fastmcp_http.py` | **Actively deployed MCP server** — FastMCP over HTTP transport (100+ tools)          |
| `basecamp_fastmcp.py`      | FastMCP server over stdio transport (same tool set as of the last stdio sync)        |
| `mcp_server_cli.py`        | Legacy JSON-RPC server (same tools, custom implementation)                           |
```

- [ ] **Step 3: Add a run command for the HTTP server**

Replace:

```markdown
# Run the MCP server (for testing)
./venv/bin/python basecamp_fastmcp.py    # FastMCP server (recommended)
./venv/bin/python mcp_server_cli.py      # Legacy CLI server
```

with:

```markdown
# Run the MCP server (for testing)
./venv/bin/python basecamp_fastmcp_http.py    # FastMCP HTTP server (actively deployed)
./venv/bin/python basecamp_fastmcp.py         # FastMCP stdio server
./venv/bin/python mcp_server_cli.py           # Legacy CLI server
```

- [ ] **Step 4: Update the Tool Categories section**

Replace:

```markdown
### Tool Categories (75 total)

- **Projects**: `get_projects`, `get_project`
- **Todos**: `get_todosets`, `get_todoset`, `get_todolists`, `get_todolist`, `create_todolist`, `update_todolist`, `trash_todolist`, `get_todos`, `get_todo`, `create_todo`, `update_todo`, `delete_todo`, `complete_todo`, `uncomplete_todo`, `reposition_todo`, `archive_todo`
- **Todo List Groups**: `get_todolist_groups`, `create_todolist_group`, `reposition_todolist_group`
- **Card Tables (Kanban)**: `get_card_table`, `get_columns`, `get_cards`, `create_card`, `move_card`, `complete_card`, etc.
- **Card Steps**: `get_card_steps`, `create_card_step`, `complete_card_step`, etc.
- **Comments**: `get_comments`, `create_comment`
- **Messages**: `get_message_board`, `get_messages`, `get_message`, `get_message_categories`, `create_message`
- **Campfire (Chat)**: `get_campfire_lines`
- **Documents**: `get_documents`, `create_document`, `update_document`, `trash_document`
- **Inbox (Email Forwards)**: `get_inbox`, `get_forwards`, `get_forward`, `get_inbox_replies`, `get_inbox_reply`, `trash_forward`
- **Search**: `search_basecamp`, `global_search`
- **Webhooks**: `get_webhooks`, `create_webhook`, `delete_webhook`
- **Other**: `get_daily_check_ins`, `get_question_answers`, `get_events`, `create_attachment`, `get_uploads`
```

with:

```markdown
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
```

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "$(cat <<'EOF'
Document Basecamp 5 MCP update: HTTP server, new components, tool count

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: Final verification

**Files:** None modified — this task only runs checks.

**Interfaces:** None.

- [ ] **Step 1: Run the full test suite**

Run: `python -m pytest tests/ -v`
Expected: PASS — `test_card_tables.py`, `test_cli_server.py`, and every class in `test_basecamp_client_v5.py` (`TestMessageBoards`, `TestMessages`, `TestMyAssignments`, `TestPeople`, `TestProjects`, `TestTodos`, `TestTodolists`, `TestTodosets`, `TestSearch`) green.

- [ ] **Step 2: Verify every new/changed tool is registered with FastMCP**

Run this from the project root (uses `venv`'s Python so `fastmcp`/`dotenv`/etc. resolve):

```bash
./venv/bin/python -c "
import asyncio
from basecamp_fastmcp_http import mcp

tools = asyncio.run(mcp.list_tools())
names = sorted(t.name for t in tools)
print(len(names), 'tools registered')

expected_new = [
    'update_message', 'pin_message', 'unpin_message', 'trash_message',
    'get_my_assignments', 'get_my_completed_assignments', 'get_my_due_assignments',
    'prioritize_assignment', 'deprioritize_assignment', 'reorder_up_next',
    'get_people', 'get_project_people', 'update_project_access', 'get_pingable_people',
    'get_person', 'get_my_profile', 'update_my_profile', 'get_my_preferences',
    'update_my_preferences',
    'create_project', 'update_project', 'archive_project', 'unarchive_project', 'trash_project',
    'reposition_todolist',
    'get_search_metadata',
]
missing = [n for n in expected_new if n not in names]
assert not missing, f'Missing tools: {missing}'
print('All new tools registered successfully')
"
```

Expected: prints a tool count (104 if no other tools were added/removed elsewhere in this pass) followed by `All new tools registered successfully`, with no `AssertionError` or import traceback. An import traceback here means a syntax or reference error was introduced in one of the earlier tasks — fix it before proceeding.

- [ ] **Step 3: Confirm the server process starts cleanly**

```bash
timeout 5 ./venv/bin/python basecamp_fastmcp_http.py 2>&1 | head -20 || true
```

Expected: log lines including `Starting Basecamp FastMCP server` and no traceback, before the 5-second timeout kills it (the HTTP server runs forever otherwise, so a clean startup with no immediate crash is the pass condition — full HTTP request/response verification requires a live Basecamp OAuth session and account, which this environment doesn't have).

- [ ] **Step 4: Final commit (only if any of the above surfaced fixes)**

If Steps 1-3 required any corrections, stage and commit them:

```bash
git add -A
git status
```

Review the diff, then commit with a message describing what was fixed. If Steps 1-3 all passed cleanly with no changes needed, skip this step — there's nothing to commit.

---

## Self-Review Notes

**Spec coverage:** All 10 sections of `docs/superpowers/specs/2026-08-07-basecamp5-mcp-update-design.md` map to a task: §1 Message Boards + §2 Messages → Task 1; §3 My Assignments → Task 2; §4 People → Task 3; §5 Projects → Task 4; §6 Todos → Task 5; §7 Todolists → Task 6; §8 Todosets → Task 7; §9 Search → Task 8; §10 Documentation → Task 9. Task 10 covers the spec's testing plan.

**Correction made during planning:** the design spec's Todolists route table listed `reposition_todolist` as an existing method to migrate, but it doesn't exist in the current codebase — only `reposition_todolist_group` does. Task 6 adds it as a new method/tool instead of a migration, consistent with the spec's stated goal (full Basecamp 5 Todolists coverage) and the docs' documented `PUT /todosets/todolists/{id}/position.json` endpoint.

**Type/signature consistency verified:** `create_message`'s new `status`/`subscriptions`/`visible_to_clients` kwargs (Task 1) match `update_message`'s independent kwarg set (different fields, no `status` since messages aren't re-drafted). `search()`'s kwarg names in `basecamp_client.py` (Task 8) match exactly what `search_basecamp`'s tool wrapper passes. `reposition_todolist`'s signature (`project_id, todolist_id, position`) matches the tool wrapper's parameter names and the `reposition_todolist_group`/`reposition_todo` precedent for position validation (`position < 1` guard in the tool layer, not the client layer).
