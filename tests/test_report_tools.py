#!/usr/bin/env python3
"""Tests for the agent-friendly report tools and compact todo output.

get_todos_due, get_overdue_todos and get_person_todos answer the common
"what's due / overdue / on X's plate" questions in one Basecamp call,
filtered and compacted server-side, so an LLM caller doesn't walk every
todolist and pull ~30-field raw records into its context. These verify
filtering, compaction, and that get_todos picks up todos inside groups.
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


def todo(id, title, due_on, project=("Threesides HQ", 3390646), list_=("Finance", 9866550417),
         assignees=(("Shannon Beagley", 1),), starts_on=None, completed=False, description=None):
    return {
        "id": id, "type": "todo", "content": title, "due_on": due_on, "starts_on": starts_on,
        "completed": completed, "description": description,
        "bucket": {"id": project[1], "name": project[0]},
        "parent": {"id": list_[1], "title": list_[0]},
        "assignees": [{"id": pid, "name": name, "avatar_url": "https://x"} for name, pid in assignees],
        "creator": {"id": 9, "name": "Someone", "avatar_url": "https://x"},
        "app_url": f"https://app.basecamp.com/3405506/buckets/{project[1]}/todos/{id}",
        "bookmark_url": "https://x", "subscription_url": "https://x",
    }


class ClientPatch:
    def __init__(self, client):
        self.client = client

    def __enter__(self):
        async def fake_get_client():
            return self.client
        self._p = patch.object(server, '_get_basecamp_client', fake_get_client)
        self._p.start()
        return self.client

    def __exit__(self, *exc):
        self._p.stop()


class TestGetTodosDue(unittest.TestCase):
    def setUp(self):
        self.client = MagicMock()
        self.client.get_upcoming_schedule.return_value = {
            "schedule_entries": [],
            "recurring_schedule_entry_occurrences": [],
            "assignables": [
                todo(1, "Payroll", "2026-10-06"),
                todo(2, "Goalfest", "2026-10-06", list_=("My HR and Admin: Bayley", 5), assignees=(("Bayley Kubara", 2),)),
                todo(3, "Anniversary", "2026-10-31", starts_on="2026-10-01", list_=("Rewards", 6)),
                todo(4, "Client ads", "2026-10-06", project=("The RUC", 3448009)),
                todo(5, "Done already", "2026-10-06", completed=True),
            ],
        }

    def test_filters_to_project_due_on_date_and_excludes_spanning_and_completed(self):
        with ClientPatch(self.client):
            result = run(server.get_todos_due("2026-10-06", project="threesides hq"))
        self.assertEqual([t["id"] for t in result["todos"]], [1, 2])
        self.client.get_upcoming_schedule.assert_called_once_with("2026-10-06", "2026-10-06")

    def test_include_spanning_adds_multi_day_items(self):
        with ClientPatch(self.client):
            result = run(server.get_todos_due("2026-10-06", project="3390646", include_spanning=True))
        self.assertIn(3, [t["id"] for t in result["todos"]])

    def test_assignee_filter_matches_name_fragment(self):
        with ClientPatch(self.client):
            result = run(server.get_todos_due("2026-10-06", assignee="bayley"))
        self.assertEqual([t["id"] for t in result["todos"]], [2])

    def test_output_is_compact(self):
        with ClientPatch(self.client):
            result = run(server.get_todos_due("2026-10-06", project="The RUC"))
        self.assertEqual(result["todos"][0], {
            "id": 4, "title": "Client ads", "project": "The RUC", "project_id": 3448009,
            "list": "Finance", "list_id": 9866550417, "assignees": ["Shannon Beagley"],
            "due_on": "2026-10-06", "app_url": "https://app.basecamp.com/3405506/buckets/3448009/todos/4",
        })

    def test_bad_date_returns_error_without_calling_api(self):
        with ClientPatch(self.client):
            result = run(server.get_todos_due("6 Oct"))
        self.assertEqual(result["error"], "Execution error")
        self.client.get_upcoming_schedule.assert_not_called()


class TestGetOverdueTodos(unittest.TestCase):
    def test_flattens_filters_sorts_and_truncates_with_counts(self):
        client = MagicMock()
        client.get_overdue_todos.return_value = {
            "under_a_week_late": [todo(1, "A", "2026-10-03"), todo(2, "B", "2026-10-05", project=("The RUC", 7))],
            "over_a_week_late": [todo(3, "C", "2026-09-20")],
            "over_a_month_late": [],
            "over_three_months_late": [todo(4, "D", "2026-01-01")],
        }
        with ClientPatch(client):
            result = run(server.get_overdue_todos(project="Threesides", max_results=2))
        self.assertEqual([t["id"] for t in result["todos"]], [1, 3])
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["returned"], 2)
        self.assertEqual(result["counts_by_project"], {"Threesides HQ": 3})


class TestGetPersonTodos(unittest.TestCase):
    def test_ambiguous_name_returns_candidates(self):
        client = MagicMock()
        client.get_assignable_people.return_value = [{"id": 1, "name": "Brad Gale"}, {"id": 2, "name": "Brad Rush"}]
        with ClientPatch(client):
            result = run(server.get_person_todos("brad"))
        self.assertEqual(result["error"], "Ambiguous person")
        self.assertEqual(len(result["candidates"]), 2)
        client.get_person_todos.assert_not_called()

    def test_resolves_name_and_notes_100_cap(self):
        client = MagicMock()
        client.get_assignable_people.return_value = [{"id": 11, "name": "Shannon Beagley"}]
        client.get_person_todos.return_value = {
            "person": {"id": 11, "name": "Shannon Beagley"},
            "todos": [todo(i, f"T{i}", None) for i in range(100)],
        }
        with ClientPatch(client):
            result = run(server.get_person_todos("shannon"))
        client.get_person_todos.assert_called_once_with("11")
        self.assertEqual(result["count"], 100)
        self.assertIn("note", result)


class TestCompactTodos(unittest.TestCase):
    def test_get_todos_includes_group_todos_and_plain_text_notes(self):
        client = MagicMock()
        client.get_todos.side_effect = lambda project_id, list_id: {
            "10": [todo(1, "Top", "2026-10-06", description="<div>Hello&nbsp;<b>there</b></div>")],
            "20": [todo(2, "In group", "2026-10-06")],
        }[list_id]
        client.get_todolist_groups.return_value = [{"id": 20, "title": "Group"}]
        with ClientPatch(client):
            result = run(server.get_todos("3390646", "10"))
        self.assertEqual([t["id"] for t in result["todos"]], [1, 2])
        self.assertEqual(result["todos"][0]["notes"], "Hello there")
        self.assertNotIn("creator", result["todos"][0])

    def test_get_todos_full_returns_raw_records(self):
        client = MagicMock()
        client.get_todos.return_value = [todo(1, "Top", None)]
        with ClientPatch(client):
            result = run(server.get_todos("3390646", "10", full=True))
        self.assertIn("creator", result["todos"][0])
        client.get_todolist_groups.assert_not_called()


class TestCompactSearch(unittest.TestCase):
    def test_search_results_are_compact_with_snippet(self):
        client = MagicMock()
        client.search.return_value = [{
            "id": 7, "type": "Todo", "title": "Process - Christmas Shutdown",
            "bucket": {"id": 3390646, "name": "Threesides HQ"},
            "parent": {"id": 8, "title": "Procedures and Templates"},
            "plain_text_content": "x" * 500, "content": "<div>…</div>",
            "creator": {"id": 1, "name": "Shannon Beagley", "avatar_url": "https://x"},
            "updated_at": "2026-05-07T07:19:27.069Z", "app_url": "https://app.basecamp.com/r/7",
            "bookmark_url": "https://x",
        }]
        with ClientPatch(client):
            result = run(server.search_basecamp("christmas shutdown"))
        hit = result["results"][0]
        self.assertEqual(set(hit), {"id", "type", "title", "project", "project_id", "parent", "snippet",
                                    "creator", "updated_at", "app_url"})
        self.assertEqual(len(hit["snippet"]), 201)
        self.assertEqual(hit["updated_at"], "2026-05-07")

    def test_search_caps_results_at_per_page(self):
        client = MagicMock()
        client.search.return_value = [{"id": i, "type": "Todo", "title": f"T{i}"} for i in range(50)]
        with ClientPatch(client):
            result = run(server.search_basecamp("x", per_page=5))
        self.assertEqual([r["id"] for r in result["results"]], [0, 1, 2, 3, 4])


class TestCompactSingleItems(unittest.TestCase):
    def test_get_todo_keeps_long_notes_and_drops_noise(self):
        client = MagicMock()
        client.get_todo.return_value = todo(1, "Process", None, description="<p>" + "word " * 300 + "</p>")
        with ClientPatch(client):
            result = run(server.get_todo("3390646", "1"))
        self.assertGreater(len(result["todo"]["notes"]), 1000)
        self.assertEqual(result["todo"]["creator"], "Someone")
        self.assertNotIn("bookmark_url", result["todo"])

    def test_get_comments_compact(self):
        client = MagicMock()
        client.get_comments.return_value = {
            "comments": [{"id": 5, "content": "<div>Moved due to <b>shutdown</b></div>",
                          "created_at": "2025-12-01T01:00:00Z", "creator": {"name": "Lauren"}, "url": "https://x"}],
            "total_count": 1, "next_page": None,
        }
        with ClientPatch(client):
            result = run(server.get_comments("1", "3390646"))
        self.assertEqual(result["comments"], [{"id": 5, "author": "Lauren", "created_at": "2025-12-01",
                                               "text": "Moved due to shutdown"}])


class TestTextOnlyResults(unittest.TestCase):
    """Over MCP, tools advertise no output schema and return text content only."""

    def test_list_and_call_over_mcp(self):
        from fastmcp import Client

        client = MagicMock()
        client.get_upcoming_schedule.return_value = {"assignables": [todo(1, "Payroll", "2026-10-06")]}

        async def scenario():
            async with Client(server.mcp) as mcp_client:
                tools = {t.name: t for t in await mcp_client.list_tools()}
                result = await mcp_client.call_tool("get_todos_due", {"start_date": "2026-10-06"})
                return tools, result

        with ClientPatch(client):
            tools, result = run(scenario())
        self.assertIsNone(tools["get_todos_due"].output_schema)
        self.assertIsNone(result.structured_content)
        self.assertIn('"Payroll"', result.content[0].text)


if __name__ == '__main__':
    unittest.main()
