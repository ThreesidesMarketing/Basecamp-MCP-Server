import os
import re

import requests
from dotenv import load_dotenv


class BasecampClient:
    """
    Client for interacting with Basecamp 3 API using Basic Authentication or OAuth 2.0.
    """

    def __init__(self, username=None, password=None, account_id=None, user_agent=None,
                 access_token=None, auth_mode="basic"):
        """
        Initialize the Basecamp client with credentials.

        Args:
            username (str, optional): Basecamp username (email) for Basic Auth
            password (str, optional): Basecamp password for Basic Auth
            account_id (str, optional): Basecamp account ID
            user_agent (str, optional): User agent for API requests
            access_token (str, optional): OAuth access token for OAuth Auth
            auth_mode (str, optional): Authentication mode ('basic' or 'oauth')
        """
        # Load environment variables if not provided directly
        load_dotenv()

        self.auth_mode = auth_mode.lower()
        self.account_id = account_id or os.getenv('BASECAMP_ACCOUNT_ID')
        self.user_agent = user_agent or os.getenv('USER_AGENT')

        # Set up authentication based on mode
        if self.auth_mode == 'basic':
            self.username = username or os.getenv('BASECAMP_USERNAME')
            self.password = password or os.getenv('BASECAMP_PASSWORD')

            if not all([self.username, self.password, self.account_id, self.user_agent]):
                raise ValueError("Missing required credentials for Basic Auth. Set them in .env file or pass them to the constructor.")

            self.auth = (self.username, self.password)
            self.headers = {
                "User-Agent": self.user_agent,
                "Content-Type": "application/json"
            }

        elif self.auth_mode == 'oauth':
            self.access_token = access_token or os.getenv('BASECAMP_ACCESS_TOKEN')

            if not all([self.access_token, self.account_id, self.user_agent]):
                raise ValueError("Missing required credentials for OAuth. Set them in .env file or pass them to the constructor.")

            self.auth = None  # No basic auth needed for OAuth
            self.headers = {
                "User-Agent": self.user_agent,
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.access_token}"
            }

        else:
            raise ValueError("Invalid auth_mode. Must be 'basic' or 'oauth'")

        # Basecamp 3 uses a different URL structure
        self.base_url = f"https://3.basecampapi.com/{self.account_id}"

    def test_connection(self):
        """Test the connection to Basecamp API."""
        response = self.get('projects.json')
        if response.status_code == 200:
            return True, "Connection successful"
        else:
            return False, f"Connection failed: {response.status_code} - {response.text}"

    def get(self, endpoint, params=None):
        """Make a GET request to the Basecamp API."""
        url = f"{self.base_url}/{endpoint}"
        return requests.get(url, auth=self.auth, headers=self.headers, params=params)

    def post(self, endpoint, data=None, allow_redirects=True):
        """Make a POST request to the Basecamp API."""
        url = f"{self.base_url}/{endpoint}"
        return requests.post(url, auth=self.auth, headers=self.headers, json=data, allow_redirects=allow_redirects)

    def put(self, endpoint, data=None):
        """Make a PUT request to the Basecamp API."""
        url = f"{self.base_url}/{endpoint}"
        return requests.put(url, auth=self.auth, headers=self.headers, json=data)

    def delete(self, endpoint):
        """Make a DELETE request to the Basecamp API."""
        url = f"{self.base_url}/{endpoint}"
        return requests.delete(url, auth=self.auth, headers=self.headers)

    def patch(self, endpoint, data=None):
        """Make a PATCH request to the Basecamp API."""
        url = f"{self.base_url}/{endpoint}"
        return requests.patch(url, auth=self.auth, headers=self.headers, json=data)

    # Project methods
    def get_projects(self):
        """Get all projects, handling pagination.

        Basecamp paginates list endpoints (commonly 15 items per page). This
        implementation follows pagination via the `page` query parameter and
        the HTTP `Link` header if present, aggregating all pages before
        returning the combined list. Only returns id, status, name, description,
        and dock properties.
        """
        endpoint = 'projects.json'

        all_projects = []
        page = 1

        while True:
            response = self.get(endpoint, params={"page": page})
            if response.status_code != 200:
                raise Exception(f"Failed to get projects: {response.status_code} - {response.text}")

            page_items = response.json() or []
            # Filter to only include specified properties
            filtered_items = [
                {
                    key: project[key]
                    for key in ['id', 'status', 'name', 'description']
                    if key in project
                }
                for project in page_items
            ]
            all_projects.extend(filtered_items)

            # Check for next page using Link header or by empty result
            link_header = response.headers.get("Link", "")
            has_next = 'rel="next"' in link_header if link_header else False

            if not page_items or not has_next:
                break

            page += 1

        return all_projects

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
        elif response.status_code == 507:
            raise Exception(f"Cannot create project: account is on a free plan and has reached its project limit. Upgrade the subscription to create more projects. {response.text}")
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
        elif response.status_code == 507:
            raise Exception(f"Cannot unarchive project: account is on a free plan and has reached its project limit. Upgrade the subscription or trash another project to restore this one. {response.text}")
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
    def get_todosets(self, project_id):
        """Get all todosets for a project.

        Basecamp 3 supports multiple todosets per project. This method retrieves
        all todosets from the project's dock.

        Args:
            project_id (str): The project ID

        Returns:
            list: List of todosets for the project
        """
        project = self.get_project(project_id)
        todosets = [item for item in project.get("dock", []) if item.get("name") == "todoset"]
        return todosets

    def get_todoset(self, project_id, todoset_id=None):
        """Get a specific todoset for a project.

        If todoset_id is not provided, returns the first todoset found in the project's dock
        (for backward compatibility).

        Args:
            project_id (str): The project ID
            todoset_id (str, optional): Specific todoset ID to retrieve

        Returns:
            dict: The todoset object
        """
        if todoset_id:
            # Get specific todoset by ID
            endpoint = f'todosets/{todoset_id}.json'
            response = self.get(endpoint)
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"Failed to get todoset {todoset_id}: {response.status_code} - {response.text}")
        else:
            # Backward compatibility: get first todoset from project dock
            project = self.get_project(project_id)
            try:
                return next(_ for _ in project["dock"] if _["name"] == "todoset")
            except (IndexError, TypeError, StopIteration):
                raise Exception(f"Failed to get todoset for project: {project_id}. Project response: {project}")

    def get_todolists(self, project_id, todoset_id=None):
        """Get all todolists for a project, optionally filtered by todoset.

        Args:
            project_id (str): The project ID
            todoset_id (str, optional): Specific todoset ID to get todolists from.
                                      If not provided, gets todolists from all todosets.

        Returns:
            list: List of todolists
        """
        if todoset_id:
            # Get todolists from specific todoset
            endpoint = f'todosets/{todoset_id}/todolists.json'
            response = self.get(endpoint)
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"Failed to get todolists: {response.status_code} - {response.text}")
        else:
            # Get todolists from all todosets
            todosets = self.get_todosets(project_id)
            all_todolists = []
            
            for todoset in todosets:
                todoset_id = todoset['id']
                endpoint = f'todosets/{todoset_id}/todolists.json'
                response = self.get(endpoint)
                if response.status_code == 200:
                    todolists = response.json()
                    # Add todoset metadata to each todolist
                    for todolist in todolists:
                        todolist['todoset'] = {
                            'id': todoset_id, 
                            'name': todoset.get('title', todoset.get('name', 'Unknown'))
                        }
                    all_todolists.extend(todolists)
                else:
                    # Log warning but continue with other todosets
                    print(f"Warning: Failed to get todolists for todoset {todoset_id}: {response.status_code} - {response.text}")
            
            return all_todolists

    def get_todolist(self, project_id, todolist_id):
        """Get a specific todolist."""
        response = self.get(f'todolists/{todolist_id}.json')
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get todolist: {response.status_code} - {response.text}")

    def create_todolist(self, project_id, name, description=None):
        """Create a new todolist in a project.

        Args:
            project_id (str): Project ID
            name (str): Todolist name (required)
            description (str, optional): HTML description

        Returns:
            dict: The created todolist object
        """
        todoset = self.get_todoset(project_id)
        todoset_id = todoset['id']
        endpoint = f'todosets/{todoset_id}/todolists.json'
        data = {'name': name}
        if description is not None:
            data['description'] = description
        response = self.post(endpoint, data)
        if response.status_code == 201:
            return response.json()
        else:
            raise Exception(f"Failed to create todolist: {response.status_code} - {response.text}")

    def update_todolist(self, project_id, todolist_id, name, description=None):
        """Update an existing todolist.

        Args:
            project_id (str): Project ID
            todolist_id (str): Todolist ID
            name (str): New name (required by API)
            description (str, optional): New HTML description

        Returns:
            dict: The updated todolist object
        """
        endpoint = f'todolists/{todolist_id}.json'
        data = {'name': name}
        if description is not None:
            data['description'] = description
        response = self.put(endpoint, data)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to update todolist: {response.status_code} - {response.text}")

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

    # To-do methods
    def get_todos(self, project_id, todolist_id):
        """Get all todos in a todolist, handling pagination.

        Basecamp paginates list endpoints (commonly 15 items per page). This
        implementation follows pagination via the `page` query parameter and
        the HTTP `Link` header if present, aggregating all pages before
        returning the combined list.
        """
        endpoint = f'todolists/{todolist_id}/todos.json'

        all_todos = []
        page = 1

        while True:
            response = self.get(endpoint, params={"page": page})
            if response.status_code != 200:
                raise Exception(f"Failed to get todos: {response.status_code} - {response.text}")

            page_items = response.json() or []
            all_todos.extend(page_items)

            # Check for next page using Link header or by empty result
            link_header = response.headers.get("Link", "")
            has_next = 'rel="next"' in link_header if link_header else False

            if not page_items or not has_next:
                break

            page += 1

        return all_todos

    def get_todo(self, project_id, todo_id):
        """Get a specific todo.

        Args:
            project_id (str): Project ID (bucket)
            todo_id (str): Todo ID

        Returns:
            dict: The todo object
        """
        endpoint = f'todos/{todo_id}.json'
        response = self.get(endpoint)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get todo: {response.status_code} - {response.text}")

    def create_todo(self, project_id, todolist_id, content, description=None, assignee_ids=None,
                    completion_subscriber_ids=None, notify=False, due_on=None, starts_on=None):
        """
        Create a new todo item in a todolist.
        
        Args:
            project_id (str): Project ID
            todolist_id (str): Todolist ID
            content (str): The todo item's text (required)
            description (str, optional): HTML description
            assignee_ids (list, optional): List of person IDs to assign
            completion_subscriber_ids (list, optional): List of person IDs to notify on completion
            notify (bool, optional): Whether to notify assignees
            due_on (str, optional): Due date in YYYY-MM-DD format
            starts_on (str, optional): Start date in YYYY-MM-DD format
            
        Returns:
            dict: The created todo
        """
        endpoint = f'todolists/{todolist_id}/todos.json'
        data = {'content': content}
        
        if description is not None:
            data['description'] = description
        if assignee_ids is not None:
            data['assignee_ids'] = assignee_ids
        if completion_subscriber_ids is not None:
            data['completion_subscriber_ids'] = completion_subscriber_ids
        if notify is not None:
            data['notify'] = notify
        if due_on is not None:
            data['due_on'] = due_on
        if starts_on is not None:
            data['starts_on'] = starts_on
            
        response = self.post(endpoint, data)
        if response.status_code == 201:
            return response.json()
        else:
            raise Exception(f"Failed to create todo: {response.status_code} - {response.text}")

    def update_todo(self, project_id, todo_id, content=None, description=None, assignee_ids=None,
                    completion_subscriber_ids=None, notify=None, due_on=None, starts_on=None):
        """
        Update an existing todo item.

        `PUT /todos/{id}.json` is a full-replace endpoint: any parameter left
        out of the request body clears its existing value server-side (e.g.
        missing assignee_ids clears all assignees). To give callers real
        partial-update semantics, any parameter left as None here is
        backfilled from the todo's current state before the request is sent.
        Pass an empty list/string explicitly to clear a field instead of
        leaving it unset.

        Args:
            project_id (str): Project ID
            todo_id (str): Todo ID
            content (str, optional): The todo item's text
            description (str, optional): HTML description
            assignee_ids (list, optional): List of person IDs to assign
            completion_subscriber_ids (list, optional): List of person IDs to notify on completion
            notify (bool, optional): Whether to notify assignees
            due_on (str, optional): Due date in YYYY-MM-DD format
            starts_on (str, optional): Start date in YYYY-MM-DD format

        Returns:
            dict: The updated todo
        """
        if all(v is None for v in [content, description, assignee_ids,
                                    completion_subscriber_ids, notify, due_on, starts_on]):
            raise ValueError("No fields provided to update")

        endpoint = f'todos/{todo_id}.json'

        if (content is None or description is None or assignee_ids is None
                or completion_subscriber_ids is None or due_on is None or starts_on is None):
            current = self.get_todo(project_id, todo_id)
            if content is None:
                content = current.get('content')
            if description is None:
                description = current.get('description')
            if assignee_ids is None:
                assignee_ids = [person['id'] for person in current.get('assignees') or []]
            if completion_subscriber_ids is None:
                completion_subscriber_ids = [person['id'] for person in current.get('completion_subscribers') or []]
            if due_on is None:
                due_on = current.get('due_on')
            if starts_on is None:
                starts_on = current.get('starts_on')

        data = {
            'content': content,
            'description': description,
            'assignee_ids': assignee_ids,
            'completion_subscriber_ids': completion_subscriber_ids,
            'due_on': due_on,
            'starts_on': starts_on,
        }
        if notify is not None:
            data['notify'] = notify

        response = self.put(endpoint, data)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to update todo: {response.status_code} - {response.text}")

    def delete_todo(self, project_id, todo_id):
        """
        Move a todo item to the trash.

        Args:
            project_id (str): Project ID
            todo_id (str): Todo ID

        Returns:
            bool: True if successful
        """
        endpoint = f'recordings/{todo_id}/status/trashed.json'
        response = self.put(endpoint)
        if response.status_code == 204:
            return True
        else:
            raise Exception(f"Failed to trash todo: {response.status_code} - {response.text}")

    def archive_todo(self, project_id, todo_id):
        """
        Archive a todo item.

        Args:
            project_id (str): Project ID
            todo_id (str): Todo ID

        Returns:
            bool: True if successful
        """
        endpoint = f'recordings/{todo_id}/status/archived.json'
        response = self.put(endpoint)
        if response.status_code == 204:
            return True
        else:
            raise Exception(f"Failed to archive todo: {response.status_code} - {response.text}")

    def reposition_todo(self, project_id, todo_id, position, parent_id=None):
        """
        Reposition a todo within its list, or move it to another list/group.

        Args:
            project_id (str): Project ID
            todo_id (str): Todo ID
            position (int): New 1-based position
            parent_id (str, optional): ID of the target todolist or group to
                move the todo into. Omit to keep the todo in its current list.

        Returns:
            bool: True if successful
        """
        endpoint = f'todos/{todo_id}/position.json'
        data = {'position': position}
        if parent_id is not None:
            data['parent_id'] = parent_id
        response = self.put(endpoint, data)
        if response.status_code == 204:
            return True
        else:
            raise Exception(f"Failed to reposition todo: {response.status_code} - {response.text}")

    def complete_todo(self, project_id, todo_id):
        """
        Mark a todo as complete.
        
        Args:
            project_id (str): Project ID
            todo_id (str): Todo ID
            
        Returns:
            dict: Completion details
        """
        endpoint = f'todos/{todo_id}/completion.json'
        response = self.post(endpoint)
        if response.status_code == 201:
            return response.json()
        else:
            raise Exception(f"Failed to complete todo: {response.status_code} - {response.text}")

    def uncomplete_todo(self, project_id, todo_id):
        """
        Mark a todo as incomplete.
        
        Args:
            project_id (str): Project ID
            todo_id (str): Todo ID
            
        Returns:
            bool: True if successful
        """
        endpoint = f'todos/{todo_id}/completion.json'
        response = self.delete(endpoint)
        if response.status_code == 204:
            return True
        else:
            raise Exception(f"Failed to uncomplete todo: {response.status_code} - {response.text}")

    # Todolist group methods
    def get_todolist_groups(self, project_id, todolist_id):
        """Get all groups in a todolist.

        Args:
            project_id (str): Project ID
            todolist_id (str): Todolist ID

        Returns:
            list: List of group objects
        """
        endpoint = f'todolists/{todolist_id}/groups.json'
        all_groups = []
        page = 1
        while True:
            response = self.get(endpoint, params={"page": page})
            if response.status_code != 200:
                raise Exception(f"Failed to get todolist groups: {response.status_code} - {response.text}")
            page_items = response.json() or []
            all_groups.extend(page_items)
            link_header = response.headers.get("Link", "")
            if not page_items or 'rel="next"' not in link_header:
                break
            page += 1
        return all_groups

    def create_todolist_group(self, project_id, todolist_id, name, color=None):
        """Create a new group inside a todolist.

        Args:
            project_id (str): Project ID
            todolist_id (str): Todolist ID
            name (str): Group name (required)
            color (str, optional): One of: white, red, orange, yellow, green,
                blue, aqua, purple, gray, pink, brown

        Returns:
            dict: The created group object
        """
        endpoint = f'todolists/{todolist_id}/groups.json'
        data = {'name': name}
        if color is not None:
            data['color'] = color
        response = self.post(endpoint, data)
        if response.status_code == 201:
            return response.json()
        else:
            raise Exception(f"Failed to create todolist group: {response.status_code} - {response.text}")

    def reposition_todolist_group(self, project_id, group_id, position):
        """Reposition a todolist group.

        Args:
            project_id (str): Project ID
            group_id (str): Group ID
            position (int): New 1-based position

        Returns:
            bool: True if successful
        """
        endpoint = f'todolists/groups/{group_id}/position.json'
        response = self.put(endpoint, {'position': position})
        if response.status_code == 204:
            return True
        else:
            raise Exception(f"Failed to reposition todolist group: {response.status_code} - {response.text}")

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

    def get_campfire_lines(self, project_id, campfire_id):
        """Get chat lines from a campfire."""
        response = self.get(f'buckets/{project_id}/chats/{campfire_id}/lines.json')
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get campfire lines: {response.status_code} - {response.text}")

    # Message board methods
    def get_message_board(self, project_id):
        """Get the message board for a project.

        The message board ID is discovered from the project's dock array,
        following the same pattern as get_todoset().

        Args:
            project_id: Project/bucket ID

        Returns:
            dict: Message board details including id, title, messages_count, etc.
        """
        project = self.get_project(project_id)
        try:
            dock_item = next(_ for _ in project["dock"] if _["name"] == "message_board")
            board_id = dock_item['id']
            response = self.get(f'message_boards/{board_id}.json')
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"Failed to get message board: {response.status_code} - {response.text}")
        except (IndexError, TypeError, StopIteration):
            raise Exception(f"No message board found for project: {project_id}")

    def get_messages(self, project_id, message_board_id=None):
        """Get all messages from a message board, handling pagination.

        Basecamp paginates list endpoints (commonly 15 items per page). This
        implementation follows pagination via the `page` query parameter and
        the HTTP `Link` header if present, aggregating all pages before
        returning the combined list.

        Args:
            project_id: Project/bucket ID
            message_board_id: Optional message board ID. If not provided,
                will be discovered from the project's dock.

        Returns:
            list: All messages from the message board
        """
        if not message_board_id:
            message_board = self.get_message_board(project_id)
            message_board_id = message_board['id']

        endpoint = f'message_boards/{message_board_id}/messages.json'

        all_messages = []
        page = 1

        while True:
            response = self.get(endpoint, params={"page": page})
            if response.status_code != 200:
                raise Exception(f"Failed to get messages: {response.status_code} - {response.text}")

            page_items = response.json() or []
            all_messages.extend(page_items)

            # Check for next page using Link header
            link_header = response.headers.get("Link", "")
            has_next = 'rel="next"' in link_header if link_header else False

            if not page_items or not has_next:
                break

            page += 1

        return all_messages

    def get_message(self, project_id, message_id):
        """Get a specific message.

        Args:
            project_id: Project/bucket ID
            message_id: Message ID

        Returns:
            dict: Message details including title, content, creator, etc.
        """
        endpoint = f'messages/{message_id}.json'
        response = self.get(endpoint)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get message: {response.status_code} - {response.text}")

    def get_message_categories(self, project_id):
        """Get message categories (types) for a project.

        Args:
            project_id: Project/bucket ID

        Returns:
            list: Message categories with id, name, and icon
        """
        endpoint = f'buckets/{project_id}/categories.json'
        response = self.get(endpoint)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get message categories: {response.status_code} - {response.text}")

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

    # Inbox methods (Email Forwards)
    def get_inbox(self, project_id):
        """Get the inbox for a project (email forwards container).

        The inbox ID is discovered from the project's dock array,
        following the same pattern as get_message_board().

        Args:
            project_id: Project/bucket ID

        Returns:
            dict: Inbox details including forwards_count, forwards_url, etc.
        """
        project = self.get_project(project_id)
        try:
            dock_item = next(_ for _ in project["dock"] if _["name"] == "inbox")
            inbox_id = dock_item['id']
            response = self.get(f'buckets/{project_id}/inboxes/{inbox_id}.json')
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"Failed to get inbox: {response.status_code} - {response.text}")
        except (IndexError, TypeError, StopIteration):
            raise Exception(f"No inbox found for project: {project_id}")

    def get_forwards(self, project_id, inbox_id=None):
        """Get all forwards from an inbox, handling pagination.

        Args:
            project_id: Project/bucket ID
            inbox_id: Optional inbox ID. If not provided,
                will be discovered from the project's dock.

        Returns:
            list: All forwards from the inbox
        """
        if not inbox_id:
            inbox = self.get_inbox(project_id)
            inbox_id = inbox['id']

        endpoint = f'buckets/{project_id}/inboxes/{inbox_id}/forwards.json'

        all_forwards = []
        page = 1

        while True:
            response = self.get(endpoint, params={"page": page})
            if response.status_code != 200:
                raise Exception(f"Failed to get forwards: {response.status_code} - {response.text}")

            page_items = response.json() or []
            all_forwards.extend(page_items)

            # Check for next page using Link header
            link_header = response.headers.get("Link", "")
            has_next = 'rel="next"' in link_header if link_header else False

            if not page_items or not has_next:
                break

            page += 1

        return all_forwards

    def get_forward(self, project_id, forward_id):
        """Get a specific forward.

        Args:
            project_id: Project/bucket ID
            forward_id: Forward ID

        Returns:
            dict: Forward details including content, subject, from, replies_count, etc.
        """
        endpoint = f'buckets/{project_id}/inbox_forwards/{forward_id}.json'
        response = self.get(endpoint)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get forward: {response.status_code} - {response.text}")

    def get_inbox_replies(self, project_id, forward_id):
        """Get all replies to a forward, handling pagination.

        Args:
            project_id: Project/bucket ID
            forward_id: Forward ID

        Returns:
            list: All replies to the forward
        """
        endpoint = f'buckets/{project_id}/inbox_forwards/{forward_id}/replies.json'

        all_replies = []
        page = 1

        while True:
            response = self.get(endpoint, params={"page": page})
            if response.status_code != 200:
                raise Exception(f"Failed to get inbox replies: {response.status_code} - {response.text}")

            page_items = response.json() or []
            all_replies.extend(page_items)

            # Check for next page using Link header
            link_header = response.headers.get("Link", "")
            has_next = 'rel="next"' in link_header if link_header else False

            if not page_items or not has_next:
                break

            page += 1

        return all_replies

    def get_inbox_reply(self, project_id, forward_id, reply_id):
        """Get a specific inbox reply.

        Args:
            project_id: Project/bucket ID
            forward_id: Forward ID
            reply_id: Reply ID

        Returns:
            dict: Reply details including content, creator, etc.
        """
        endpoint = f'buckets/{project_id}/inbox_forwards/{forward_id}/replies/{reply_id}.json'
        response = self.get(endpoint)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get inbox reply: {response.status_code} - {response.text}")

    def trash_forward(self, project_id, forward_id):
        """Trash a forward.

        Uses the generic recordings trash endpoint, same pattern as trash_document.

        Args:
            project_id: Project/bucket ID
            forward_id: Forward ID

        Returns:
            bool: True if successful
        """
        endpoint = f"buckets/{project_id}/recordings/{forward_id}/status/trashed.json"
        response = self.put(endpoint)
        if response.status_code == 204:
            return True
        else:
            raise Exception(f"Failed to trash forward: {response.status_code} - {response.text}")

    # Schedule methods
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

    # Comments methods
    def get_comments(self, project_id, recording_id, page=1):
        """
        Get comments for a recording (todos, message, etc.).

        Args:
            project_id (int): Project/bucket ID.
            recording_id (int): ID of the recording (todos, message, etc.)
            page (int): Page number for pagination (default: 1).
                        Basecamp uses geared pagination: page 1 has 15 results,
                        page 2 has 30, page 3 has 50, page 4+ has 100.

        Returns:
            dict: Contains 'comments' list and pagination metadata:
                  - comments: list of comments
                  - total_count: total number of comments (from X-Total-Count header)
                  - next_page: next page number if available, None otherwise
        """
        if page < 1:
            raise ValueError("page must be >= 1")
        endpoint = f"buckets/{project_id}/recordings/{recording_id}/comments.json"
        response = self.get(endpoint, params={"page": page})
        if response.status_code == 200:
            # Parse pagination headers
            total_count = response.headers.get('X-Total-Count')
            total_count = int(total_count) if total_count else None

            # Parse Link header for next page
            next_page = None
            link_header = response.headers.get('Link', '')
            # Split by comma to handle multiple links (e.g., rel="prev", rel="next")
            for link in link_header.split(','):
                if 'rel="next"' in link:
                    match = re.search(r'page=(\d+)', link)
                    if match:
                        next_page = int(match.group(1))
                    break

            return {
                "comments": response.json(),
                "total_count": total_count,
                "next_page": next_page
            }
        else:
            raise Exception(f"Failed to get comments: {response.status_code} - {response.text}")

    def create_comment(self, recording_id, bucket_id, content):
        """
        Create a comment on a recording.

        Args:
            recording_id (int): ID of the recording to comment on
            bucket_id (int): Project/bucket ID
            content (str): Content of the comment in HTML format

        Returns:
            dict: The created comment
        """
        endpoint = f"buckets/{bucket_id}/recordings/{recording_id}/comments.json"
        data = {"content": content}
        response = self.post(endpoint, data)
        if response.status_code == 201:
            return response.json()
        else:
            raise Exception(f"Failed to create comment: {response.status_code} - {response.text}")

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

    def get_daily_check_ins(self, project_id, page=1):
        project = self.get_project(project_id)
        questionnaire = next(_ for _ in project["dock"] if _["name"] == "questionnaire")
        endpoint = f"buckets/{project_id}/questionnaires/{questionnaire['id']}/questions.json"
        response = self.get(endpoint, params={"page": page})
        if response.status_code != 200:
            raise Exception("Failed to read questions")
        return response.json()

    def get_question_answers(self, project_id, question_id, page=1):
        endpoint = f"buckets/{project_id}/questions/{question_id}/answers.json"
        response = self.get(endpoint, params={"page": page})
        if response.status_code != 200:
            raise Exception("Failed to read question answers")
        return response.json()

    # Card Table methods
    def get_card_tables(self, project_id):
        """Get all card tables for a project."""
        project = self.get_project(project_id)
        try:
            return [item for item in project["dock"] if item.get("name") in ("kanban_board", "card_table")]
        except (IndexError, TypeError):
            return []

    def get_card_table(self, project_id):
        """Get the first card table for a project (Basecamp 3 can have multiple card tables per project)."""
        card_tables = self.get_card_tables(project_id)
        if not card_tables:
            raise Exception(f"No card tables found for project: {project_id}")
        return card_tables[0]  # Return the first card table
    
    def get_card_table_details(self, project_id, card_table_id):
        """Get details for a specific card table."""
        response = self.get(f'buckets/{project_id}/card_tables/{card_table_id}.json')
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 204:
            # 204 means "No Content" - return an empty structure
            return {"lists": [], "id": card_table_id, "status": "empty"}
        else:
            raise Exception(f"Failed to get card table: {response.status_code} - {response.text}")

    # Card Table Column methods
    def get_columns(self, project_id, card_table_id):
        """Get all columns in a card table."""
        # Get the card table details which includes the lists (columns)
        card_table_details = self.get_card_table_details(project_id, card_table_id)
        return card_table_details.get('lists', [])

    def get_column(self, project_id, column_id):
        """Get a specific column."""
        response = self.get(f'buckets/{project_id}/card_tables/columns/{column_id}.json')
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get column: {response.status_code} - {response.text}")

    def create_column(self, project_id, card_table_id, title):
        """Create a new column in a card table."""
        data = {"title": title}
        response = self.post(f'buckets/{project_id}/card_tables/{card_table_id}/columns.json', data)
        if response.status_code == 201:
            return response.json()
        else:
            raise Exception(f"Failed to create column: {response.status_code} - {response.text}")

    def update_column(self, project_id, column_id, title):
        """Update a column title."""
        data = {"title": title}
        response = self.put(f'buckets/{project_id}/card_tables/columns/{column_id}.json', data)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to update column: {response.status_code} - {response.text}")

    def move_column(self, project_id, column_id, position, card_table_id):
        """Move a column to a new position."""
        data = {
            "source_id": column_id, 
            "target_id": card_table_id,
            "position": position
        }
        response = self.post(f'buckets/{project_id}/card_tables/{card_table_id}/moves.json', data)
        if response.status_code == 204:
            return True
        else:
            raise Exception(f"Failed to move column: {response.status_code} - {response.text}")

    def update_column_color(self, project_id, column_id, color):
        """Update a column color."""
        data = {"color": color}
        response = self.patch(f'buckets/{project_id}/card_tables/columns/{column_id}/color.json', data)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to update column color: {response.status_code} - {response.text}")

    def put_column_on_hold(self, project_id, column_id):
        """Put a column on hold."""
        response = self.post(f'buckets/{project_id}/card_tables/columns/{column_id}/on_hold.json')
        if response.status_code == 204:
            return True
        else:
            raise Exception(f"Failed to put column on hold: {response.status_code} - {response.text}")

    def remove_column_hold(self, project_id, column_id):
        """Remove hold from a column."""
        response = self.delete(f'buckets/{project_id}/card_tables/columns/{column_id}/on_hold.json')
        if response.status_code == 204:
            return True
        else:
            raise Exception(f"Failed to remove column hold: {response.status_code} - {response.text}")

    def watch_column(self, project_id, column_id):
        """Subscribe to column notifications."""
        response = self.post(f'buckets/{project_id}/card_tables/lists/{column_id}/subscription.json')
        if response.status_code == 204:
            return True
        else:
            raise Exception(f"Failed to watch column: {response.status_code} - {response.text}")

    def unwatch_column(self, project_id, column_id):
        """Unsubscribe from column notifications."""
        response = self.delete(f'buckets/{project_id}/card_tables/lists/{column_id}/subscription.json')
        if response.status_code == 204:
            return True
        else:
            raise Exception(f"Failed to unwatch column: {response.status_code} - {response.text}")

    # Card Table Card methods
    def get_cards(self, project_id, column_id):
        """Get all cards in a column."""
        response = self.get(f'buckets/{project_id}/card_tables/lists/{column_id}/cards.json')
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get cards: {response.status_code} - {response.text}")

    def get_card(self, project_id, card_id):
        """Get a specific card."""
        response = self.get(f'buckets/{project_id}/card_tables/cards/{card_id}.json')
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get card: {response.status_code} - {response.text}")

    def create_card(self, project_id, column_id, title, content=None, due_on=None, notify=False):
        """Create a new card in a column."""
        data = {"title": title}
        if content:
            data["content"] = content
        if due_on:
            data["due_on"] = due_on
        if notify:
            data["notify"] = notify
        response = self.post(f'buckets/{project_id}/card_tables/lists/{column_id}/cards.json', data)
        if response.status_code == 201:
            return response.json()
        else:
            raise Exception(f"Failed to create card: {response.status_code} - {response.text}")

    def update_card(self, project_id, card_id, title=None, content=None, due_on=None, assignee_ids=None):
        """Update a card."""
        data = {}
        if title:
            data["title"] = title
        if content:
            data["content"] = content
        if due_on:
            data["due_on"] = due_on
        if assignee_ids:
            data["assignee_ids"] = assignee_ids
        response = self.put(f'buckets/{project_id}/card_tables/cards/{card_id}.json', data)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to update card: {response.status_code} - {response.text}")

    def move_card(self, project_id, card_id, column_id):
        """Move a card to a new column."""
        data = {"column_id": column_id}
        response = self.post(f'buckets/{project_id}/card_tables/cards/{card_id}/moves.json', data)
        if response.status_code == 204:
            return True
        else:
            raise Exception(f"Failed to move card: {response.status_code} - {response.text}")

    def complete_card(self, project_id, card_id):
        """Mark a card as complete."""
        response = self.post(f'buckets/{project_id}/todos/{card_id}/completion.json')
        if response.status_code == 201:
            return response.json()
        else:
            raise Exception(f"Failed to complete card: {response.status_code} - {response.text}")

    def uncomplete_card(self, project_id, card_id):
        """Mark a card as incomplete."""
        response = self.delete(f'buckets/{project_id}/todos/{card_id}/completion.json')
        if response.status_code == 204:
            return True
        else:
            raise Exception(f"Failed to uncomplete card: {response.status_code} - {response.text}")

    # Card Steps methods
    def get_card_steps(self, project_id, card_id):
        """Get all steps (sub-tasks) for a card."""
        card = self.get_card(project_id, card_id)
        return card.get('steps', [])

    def create_card_step(self, project_id, card_id, title, due_on=None, assignee_ids=None):
        """Create a new step (sub-task) for a card."""
        data = {"title": title}
        if due_on:
            data["due_on"] = due_on
        if assignee_ids:
            data["assignee_ids"] = assignee_ids
        response = self.post(f'buckets/{project_id}/card_tables/cards/{card_id}/steps.json', data)
        if response.status_code == 201:
            return response.json()
        else:
            raise Exception(f"Failed to create card step: {response.status_code} - {response.text}")

    def get_card_step(self, project_id, step_id):
        """Get a specific card step."""
        response = self.get(f'buckets/{project_id}/card_tables/steps/{step_id}.json')
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get card step: {response.status_code} - {response.text}")

    def update_card_step(self, project_id, step_id, title=None, due_on=None, assignee_ids=None):
        """Update a card step."""
        data = {}
        if title:
            data["title"] = title
        if due_on:
            data["due_on"] = due_on
        if assignee_ids:
            data["assignee_ids"] = assignee_ids
        response = self.put(f'buckets/{project_id}/card_tables/steps/{step_id}.json', data)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to update card step: {response.status_code} - {response.text}")

    def delete_card_step(self, project_id, step_id):
        """Delete a card step."""
        response = self.delete(f'buckets/{project_id}/card_tables/steps/{step_id}.json')
        if response.status_code == 204:
            return True
        else:
            raise Exception(f"Failed to delete card step: {response.status_code} - {response.text}")

    def complete_card_step(self, project_id, step_id):
        """Mark a card step as complete."""
        response = self.post(f'buckets/{project_id}/todos/{step_id}/completion.json')
        if response.status_code == 201:
            return response.json()
        else:
            raise Exception(f"Failed to complete card step: {response.status_code} - {response.text}")

    def uncomplete_card_step(self, project_id, step_id):
        """Mark a card step as incomplete."""
        response = self.delete(f'buckets/{project_id}/todos/{step_id}/completion.json')
        if response.status_code == 204:
            return True
        else:
            raise Exception(f"Failed to uncomplete card step: {response.status_code} - {response.text}")

    # New methods for additional Basecamp API functionality
    def create_attachment(self, file_path, name, content_type="application/octet-stream"):
        """Upload an attachment and return the attachable sgid."""
        with open(file_path, "rb") as f:
            data = f.read()

        headers = self.headers.copy()
        headers["Content-Type"] = content_type
        headers["Content-Length"] = str(len(data))

        endpoint = f"attachments.json?name={name}"
        response = requests.post(f"{self.base_url}/{endpoint}", auth=self.auth, headers=headers, data=data)
        if response.status_code == 201:
            return response.json()
        else:
            raise Exception(f"Failed to create attachment: {response.status_code} - {response.text}")

    def get_events(self, project_id, recording_id):
        """Get events for a recording."""
        endpoint = f"buckets/{project_id}/recordings/{recording_id}/events.json"
        response = self.get(endpoint)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get events: {response.status_code} - {response.text}")

    def get_webhooks(self, project_id):
        """List webhooks for a project."""
        endpoint = f"buckets/{project_id}/webhooks.json"
        response = self.get(endpoint)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get webhooks: {response.status_code} - {response.text}")

    def create_webhook(self, project_id, payload_url, types=None):
        """Create a webhook for a project."""
        data = {"payload_url": payload_url}
        if types:
            data["types"] = types
        endpoint = f"buckets/{project_id}/webhooks.json"
        response = self.post(endpoint, data)
        if response.status_code == 201:
            return response.json()
        else:
            raise Exception(f"Failed to create webhook: {response.status_code} - {response.text}")

    def delete_webhook(self, project_id, webhook_id):
        """Delete a webhook."""
        endpoint = f"buckets/{project_id}/webhooks/{webhook_id}.json"
        response = self.delete(endpoint)
        if response.status_code == 204:
            return True
        else:
            raise Exception(f"Failed to delete webhook: {response.status_code} - {response.text}")

    def get_documents(self, project_id, vault_id):
        """List documents in a vault."""
        endpoint = f"buckets/{project_id}/vaults/{vault_id}/documents.json"
        response = self.get(endpoint)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get documents: {response.status_code} - {response.text}")

    def get_document(self, project_id, document_id):
        """Get a single document."""
        endpoint = f"buckets/{project_id}/documents/{document_id}.json"
        response = self.get(endpoint)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get document: {response.status_code} - {response.text}")

    def create_document(self, project_id, vault_id, title, content, status="active"):
        """Create a document in a vault."""
        data = {"title": title, "content": content, "status": status}
        endpoint = f"buckets/{project_id}/vaults/{vault_id}/documents.json"
        response = self.post(endpoint, data)
        if response.status_code == 201:
            return response.json()
        else:
            raise Exception(f"Failed to create document: {response.status_code} - {response.text}")

    def update_document(self, project_id, document_id, title=None, content=None):
        """Update a document's title or content."""
        data = {}
        if title:
            data["title"] = title
        if content:
            data["content"] = content
        endpoint = f"buckets/{project_id}/documents/{document_id}.json"
        response = self.put(endpoint, data)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to update document: {response.status_code} - {response.text}")

    def trash_document(self, project_id, document_id):
        """Trash a document."""
        endpoint = f"buckets/{project_id}/recordings/{document_id}/status/trashed.json"
        response = self.put(endpoint)
        if response.status_code == 204:
            return True
        else:
            raise Exception(f"Failed to trash document: {response.status_code} - {response.text}")

    # Upload methods
    def get_uploads(self, project_id, vault_id=None):
        """List uploads in a project or vault."""
        if vault_id:
            endpoint = f"buckets/{project_id}/vaults/{vault_id}/uploads.json"
        else:
            endpoint = f"buckets/{project_id}/uploads.json"
        response = self.get(endpoint)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get uploads: {response.status_code} - {response.text}")

    def get_upload(self, project_id, upload_id):
        """Get a single upload."""
        endpoint = f"buckets/{project_id}/uploads/{upload_id}.json"
        response = self.get(endpoint)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get upload: {response.status_code} - {response.text}")

    # Vault methods
    def get_vault(self, project_id, vault_id=None):
        """Get a vault.

        If vault_id is not provided, returns the project's root vault,
        discovered from the project's dock array (name == "vault"),
        following the same pattern as get_todoset().

        Args:
            project_id (str): The project ID
            vault_id (str, optional): Specific vault ID to retrieve. If
                omitted, the project's root vault is returned.

        Returns:
            dict: The vault object
        """
        if not vault_id:
            project = self.get_project(project_id)
            try:
                vault_id = next(_ for _ in project["dock"] if _["name"] == "vault")["id"]
            except (IndexError, TypeError, StopIteration):
                raise Exception(f"No vault found for project: {project_id}")

        endpoint = f"vaults/{vault_id}.json"
        response = self.get(endpoint)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get vault: {response.status_code} - {response.text}")

    def get_vault_children(self, project_id, vault_id=None):
        """Get everything filed directly under a vault, handling pagination.

        Unlike the documented `vaults/{id}/vaults.json` endpoint (which
        only returns nested sub-vaults), this uses the vault's own
        `children_url` route, which returns every child type Basecamp
        allows directly under a vault: nested Vaults (sub-folders) and
        CloudFiles (embedded Google Drive/Dropbox/Box/etc. links) at
        minimum. Each item's own "type" field tells you which.

        If vault_id is not provided, lists the children of the project's
        root vault.

        Args:
            project_id (str): The project ID
            vault_id (str, optional): Parent vault ID. If omitted, the
                project's root vault is used as the parent.

        Returns:
            list: Child items (mixed types; see item["type"])
        """
        if not vault_id:
            vault_id = self.get_vault(project_id)['id']

        endpoint = f"buckets/{project_id}/vaults/{vault_id}/children.json"

        all_children = []
        page = 1

        while True:
            response = self.get(endpoint, params={"page": page})
            if response.status_code != 200:
                raise Exception(f"Failed to get vault children: {response.status_code} - {response.text}")

            page_items = response.json() or []
            all_children.extend(page_items)

            link_header = response.headers.get("Link", "")
            has_next = 'rel="next"' in link_header if link_header else False

            if not page_items or not has_next:
                break

            page += 1

        return all_children

    def create_vault(self, project_id, title, vault_id=None):
        """Create a child vault nested under a vault.

        If vault_id is not provided, the new vault is created directly
        under the project's root vault.

        Args:
            project_id (str): The project ID
            title (str): Name of the new vault
            vault_id (str, optional): Parent vault ID to nest the new
                vault under. If omitted, the project's root vault is used.

        Returns:
            dict: The created vault
        """
        if not vault_id:
            vault_id = self.get_vault(project_id)['id']

        endpoint = f"vaults/{vault_id}/vaults.json"
        data = {"title": title}
        response = self.post(endpoint, data)
        if response.status_code == 201:
            return response.json()
        else:
            raise Exception(f"Failed to create vault: {response.status_code} - {response.text}")

    def update_vault(self, project_id, vault_id, title):
        """Rename a vault.

        Args:
            project_id (str): The project ID. Kept for interface
                consistency with the rest of the codebase; the flat
                route scopes by vault_id alone.
            vault_id (str): Vault ID to rename
            title (str): New title for the vault

        Returns:
            dict: The updated vault
        """
        endpoint = f"vaults/{vault_id}.json"
        data = {"title": title}
        response = self.put(endpoint, data)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to update vault: {response.status_code} - {response.text}")

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

    # External link methods (dock tools historically called "doors")
    def get_external_links(self, project_id=None, status=None):
        """
        List external links via the flat, generic recordings endpoint.

        This is the only endpoint that returns an external link's full
        shape (url, service, description) -- get_external_link() omits them.

        `projects/recordings.json` is paginated (commonly 15 items per
        page); this follows pagination via the `page` query parameter
        and the HTTP `Link` header, aggregating all pages before
        returning the combined list.

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

        all_links = []
        page = 1

        while True:
            page_params = dict(params)
            page_params['page'] = page
            response = self.get('projects/recordings.json', params=page_params)
            if response.status_code != 200:
                raise Exception(f"Failed to get external links: {response.status_code} - {response.text}")

            page_items = response.json() or []
            all_links.extend(page_items)

            link_header = response.headers.get("Link", "")
            has_next = 'rel="next"' in link_header if link_header else False

            if not page_items or not has_next:
                break

            page += 1

        return all_links

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
        link. The redirect target is a different host
        (3.basecamp.com vs the API host 3.basecampapi.com), and
        `requests` strips the Authorization header when following a
        cross-host redirect -- so this call disables redirect-following
        to see the raw 302 (the actual success signal) instead of the
        followed response, which would often come back unauthenticated.
        """
        door = {'service': service, 'url': url}
        if title is not None:
            door['title'] = title
        if description is not None:
            door['description'] = description
        endpoint = f'buckets/{project_id}/dock/doors.json'
        response = self.post(endpoint, {'door': door}, allow_redirects=False)
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
