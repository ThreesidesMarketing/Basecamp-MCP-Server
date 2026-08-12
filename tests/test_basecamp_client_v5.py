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

    @patch.object(BasecampClient, 'post')
    def test_create_project_507_raises_distinct_error(self, mock_post):
        mock_post.return_value = make_response(507, {"error": "storage"})

        with self.assertRaises(Exception) as ctx:
            self.client.create_project('New')

        self.assertIn('free plan', str(ctx.exception))
        self.assertIn('project limit', str(ctx.exception))

    @patch.object(BasecampClient, 'put')
    def test_unarchive_project_507_raises_distinct_error(self, mock_put):
        mock_put.return_value = make_response(507, {"error": "storage"})

        with self.assertRaises(Exception) as ctx:
            self.client.unarchive_project('1')

        self.assertIn('project limit', str(ctx.exception))

    @patch.object(BasecampClient, 'delete')
    def test_trash_project(self, mock_delete):
        mock_delete.return_value = make_response(204)

        result = self.client.trash_project('1')

        self.assertTrue(result)
        mock_delete.assert_called_once_with('projects/1.json')


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

    @patch.object(BasecampClient, 'get')
    @patch.object(BasecampClient, 'put')
    def test_update_todo_uses_flat_route(self, mock_put, mock_get):
        mock_get.return_value = make_response(200, {
            "id": 2, "content": "Old", "description": None,
            "assignees": [], "completion_subscribers": [],
            "due_on": None, "starts_on": None,
        })
        mock_put.return_value = make_response(200, {"id": 2})

        result = self.client.update_todo('999', '2', content='Updated')

        self.assertEqual(result, {"id": 2})
        endpoint = mock_put.call_args[0][0]
        self.assertEqual(endpoint, 'todos/2.json')

    @patch.object(BasecampClient, 'get')
    @patch.object(BasecampClient, 'put')
    def test_update_todo_preserves_existing_assignees_when_omitted(self, mock_put, mock_get):
        # Regression test: PUT /todos/{id}.json is a full-replace endpoint --
        # Basecamp clears any field left out of the request body. Omitting
        # assignee_ids on an update must NOT wipe existing assignees.
        mock_get.return_value = make_response(200, {
            "id": 2, "content": "Old content", "description": "desc",
            "assignees": [{"id": 5}, {"id": 9}],
            "completion_subscribers": [{"id": 7}],
            "due_on": "2026-09-01", "starts_on": None,
        })
        mock_put.return_value = make_response(200, {"id": 2})

        self.client.update_todo('999', '2', due_on='2026-10-06')

        mock_get.assert_called_once_with('todos/2.json')
        data = mock_put.call_args[0][1]
        self.assertEqual(data['due_on'], '2026-10-06')
        self.assertEqual(data['content'], 'Old content')
        self.assertEqual(data['assignee_ids'], [5, 9])
        self.assertEqual(data['completion_subscriber_ids'], [7])

    @patch.object(BasecampClient, 'get')
    @patch.object(BasecampClient, 'put')
    def test_update_todo_allows_explicit_clear_of_assignees(self, mock_put, mock_get):
        mock_get.return_value = make_response(200, {
            "id": 2, "content": "Old content", "description": None,
            "assignees": [{"id": 5}], "completion_subscribers": [],
            "due_on": None, "starts_on": None,
        })
        mock_put.return_value = make_response(200, {"id": 2})

        self.client.update_todo('999', '2', assignee_ids=[])

        data = mock_put.call_args[0][1]
        self.assertEqual(data['assignee_ids'], [])

    @patch.object(BasecampClient, 'get')
    @patch.object(BasecampClient, 'put')
    def test_update_todo_raises_when_no_fields_given(self, mock_put, mock_get):
        with self.assertRaises(ValueError):
            self.client.update_todo('999', '2')

        mock_get.assert_not_called()
        mock_put.assert_not_called()

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


class TestTodosets(unittest.TestCase):
    def setUp(self):
        self.client = make_client()

    @patch.object(BasecampClient, 'get')
    def test_get_todoset_by_id_uses_flat_route(self, mock_get):
        mock_get.return_value = make_response(200, {"id": 3})

        result = self.client.get_todoset('999', todoset_id='3')

        self.assertEqual(result, {"id": 3})
        mock_get.assert_called_once_with('todosets/3.json')


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


if __name__ == '__main__':
    unittest.main()
