#!/usr/bin/env python3
"""Test for the get_cloud_file tool's URL-parsing guard.

Basecamp's public API has no endpoint to list a vault's cloud files, so
this tool takes a Basecamp app URL the user pastes in (e.g.
https://app.basecamp.com/{account}/buckets/{project}/cloud_files/{id})
instead of a bare ID. This verifies the tool layer in
`basecamp_fastmcp_http.py` correctly extracts the cloud file ID from a
few real URL shapes, and rejects an unparseable URL before ever calling
the client.
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


class TestGetCloudFileUrlParsing(unittest.TestCase):
    def test_parses_id_from_app_basecamp_url(self):
        mock_client = MagicMock()
        mock_client.get_cloud_file.return_value = {"id": 5276167238, "title": "Brand assets"}

        async def fake_get_client():
            return mock_client

        with patch.object(server, '_get_basecamp_client', fake_get_client):
            result = run(server.get_cloud_file('https://app.basecamp.com/3405506/buckets/21048787/cloud_files/5276167238'))

        self.assertEqual(result.get('status'), 'success')
        mock_client.get_cloud_file.assert_called_once_with('5276167238')

    def test_parses_id_from_api_url_with_json_suffix(self):
        mock_client = MagicMock()
        mock_client.get_cloud_file.return_value = {"id": 5276167238, "title": "Brand assets"}

        async def fake_get_client():
            return mock_client

        with patch.object(server, '_get_basecamp_client', fake_get_client):
            result = run(server.get_cloud_file('https://3.basecampapi.com/3405506/cloud_files/5276167238.json'))

        self.assertEqual(result.get('status'), 'success')
        mock_client.get_cloud_file.assert_called_once_with('5276167238')

    def test_unparseable_url_returns_error_without_calling_client(self):
        mock_client = MagicMock()

        async def fake_get_client():
            return mock_client

        with patch.object(server, '_get_basecamp_client', fake_get_client):
            result = run(server.get_cloud_file('https://app.basecamp.com/3405506/buckets/21048787/documents/123'))

        self.assertEqual(result.get('error'), 'Invalid input')
        mock_client.get_cloud_file.assert_not_called()


if __name__ == '__main__':
    unittest.main()
