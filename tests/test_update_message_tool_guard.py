#!/usr/bin/env python3
"""Test for the update_message tool-layer 'at least one field' guard.

Covers Finding 4 from the 2026-08-07 basecamp5-mcp-update final review:
since `notify` is now always sent to the client layer, the client-layer
`ValueError("No fields provided to update")` guard in
`BasecampClient.update_message` can no longer trip on a no-op call. This
test verifies the tool layer in `basecamp_fastmcp_http.py` catches that
case itself, before calling the client at all.
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


class TestUpdateMessageToolGuard(unittest.TestCase):
    def test_no_fields_returns_error_without_calling_client(self):
        mock_client = MagicMock()

        async def fake_get_client():
            return mock_client

        with patch.object(server, '_get_basecamp_client', fake_get_client):
            result = run(server.update_message('123'))

        self.assertEqual(result.get('error'), 'Invalid input')
        mock_client.update_message.assert_not_called()

    def test_notify_alone_does_not_count_as_a_field(self):
        mock_client = MagicMock()

        async def fake_get_client():
            return mock_client

        with patch.object(server, '_get_basecamp_client', fake_get_client):
            result = run(server.update_message('123', notify=True))

        self.assertEqual(result.get('error'), 'Invalid input')
        mock_client.update_message.assert_not_called()

    def test_subject_provided_proceeds_to_client(self):
        mock_client = MagicMock()
        mock_client.update_message.return_value = {'id': 123, 'subject': 'New'}

        async def fake_get_client():
            return mock_client

        with patch.object(server, '_get_basecamp_client', fake_get_client):
            result = run(server.update_message('123', subject='New'))

        self.assertEqual(result.get('status'), 'success')
        mock_client.update_message.assert_called_once()


if __name__ == '__main__':
    unittest.main()
