#!/usr/bin/env python3
"""Regression tests for the update_todo tool signature.

Prior to this fix, `update_todo` in basecamp_fastmcp_http.py had no type
hints at all, so FastMCP generated a schema with no "type" for any
parameter (not even "array" for assignee_ids). Calling clients had no type
information to go on and would sometimes send assignee_ids as a
comma-joined or bracket-literal string instead of a JSON array, which the
Basecamp API then mis-parsed -- keeping only the first ID, or dropping all
of them. This test locks in a properly typed schema and verifies a full
list of IDs passes through to the client unmodified.
"""

import os
import sys
import unittest
import asyncio
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import basecamp_fastmcp_http as server


def run(coro):
    return asyncio.run(coro)


class TestUpdateTodoToolSchema(unittest.TestCase):
    def test_assignee_ids_declared_as_integer_array(self):
        tools = run(server.mcp.list_tools())
        update_todo = next(t for t in tools if t.name == 'update_todo')
        schema = update_todo.parameters

        assignee_schema = schema['properties']['assignee_ids']
        variants = assignee_schema.get('anyOf', [assignee_schema])
        array_variant = next(v for v in variants if v.get('type') == 'array')
        self.assertEqual(array_variant['items']['type'], 'integer')

    def test_full_assignee_id_list_passed_through_unmodified(self):
        mock_client = MagicMock()
        mock_client.update_todo.return_value = {'id': 2}

        async def fake_get_client():
            return mock_client

        with patch.object(server, '_get_basecamp_client', fake_get_client):
            result = run(server.update_todo('999', '2', assignee_ids=[5, 9, 12]))

        self.assertEqual(result.get('status'), 'success')
        _, kwargs = mock_client.update_todo.call_args
        self.assertEqual(kwargs['assignee_ids'], [5, 9, 12])


if __name__ == '__main__':
    unittest.main()
