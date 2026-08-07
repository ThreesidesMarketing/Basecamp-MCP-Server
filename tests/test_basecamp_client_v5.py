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
