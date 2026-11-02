#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import os
import ssl

from pathlib import Path
from typing import Any

from aiohttp import web
from toolathlon_mock_utils import load_case_token_values, mutate_store, slugify, utc_now_iso


STORE_PATH_ENV = 'NEXUS_TOOLATHLON_CANVAS_STORE_PATH'
DEFAULT_PORTS = '10001,20001'
DEFAULT_TLS_PORTS = {20001}
CERT_PATH = Path(__file__).with_name('toolathlon_canvas_localhost.crt')
KEY_PATH = Path(__file__).with_name('toolathlon_canvas_localhost.key')
CASE_TOKENS = load_case_token_values()


def _store_path() -> str:
    value = os.getenv(STORE_PATH_ENV)
    if not value:
        raise RuntimeError(f'{STORE_PATH_ENV} is not set')
    return value


def _case_root() -> Path | None:
    value = os.getenv('NEXUS_TOOLATHLON_CASE_ROOT')
    return Path(value) if value else None


def _default_state() -> dict[str, Any]:
    return {
        'version': 1,
        'seeded': False,
        'next_ids': {
            'user': 1,
            'course': 1,
            'enrollment': 1,
            'assignment': 1,
            'submission': 1,
            'file': 1,
            'discussion': 1,
            'entry': 1,
            'quiz': 1,
            'question': 1,
            'quiz_submission': 1,
            'conversation': 1,
            'message': 1,
            'page': 1,
        },
        'token_map': {},
        'users': {},
        'courses': {},
        'enrollments': {},
        'assignments': {},
        'files': {},
        'discussions': {},
        'quizzes': {},
        'conversations': {},
        'pages': {},
        'module_progress': {},
    }


def _next_id(state: dict[str, Any], key: str) -> int:
    value = int(state['next_ids'][key])
    state['next_ids'][key] = value + 1
    return value


def _name_from_email(email: str) -> str:
    local = email.split('@', 1)[0].replace('.', ' ').replace('_', ' ').replace('-', ' ').strip()
    return ' '.join(part.capitalize() for part in local.split()) or email


def _looks_like_email(value: str) -> bool:
    if '@' not in value:
        return False
    _, _, domain = value.partition('@')
    return '.' in domain


def _iter_case_users() -> list[dict[str, str]]:
    case_root = _case_root()
    if case_root is None or not case_root.exists():
        return []

    rows: list[dict[str, str]] = []
    for csv_path in sorted(case_root.rglob('*.csv')):
        try:
            with csv_path.open('r', encoding='utf-8', newline='') as handle:
                reader = csv.DictReader(handle)
                fieldnames = [field for field in (reader.fieldnames or []) if field]
                email_fields = [field for field in fieldnames if 'email' in field.lower()]
                if not email_fields:
                    continue
                name_fields = [
                    field for field in fieldnames if field.lower() == 'name' or field.lower().endswith('_name')
                ]
                token_fields = [field for field in fieldnames if 'canvas' in field.lower() and 'token' in field.lower()]
                for row in reader:
                    email = ''
                    for field in email_fields:
                        candidate = str(row.get(field, '')).strip()
                        if _looks_like_email(candidate):
                            email = candidate
                            break
                    if not email:
                        continue
                    name = ''
                    for field in name_fields:
                        candidate = str(row.get(field, '')).strip()
                        if candidate:
                            name = candidate
                            break
                    token = ''
                    for field in token_fields:
                        candidate = str(row.get(field, '')).strip()
                        if candidate:
                            token = candidate
                            break
                    rows.append({'email': email, 'name': name, 'token': token})
        except Exception:
            continue
    return rows


def _create_user(
    state: dict[str, Any],
    *,
    email: str,
    name: str | None = None,
    token: str | None = None,
    admin: bool = False,
) -> dict[str, Any]:
    for user in state['users'].values():
        if user.get('email') == email:
            if token:
                state['token_map'][token] = user['id']
            return user

    user_id = _next_id(state, 'user')
    full_name = name or _name_from_email(email)
    user = {
        'id': user_id,
        'name': full_name,
        'short_name': full_name.split()[0] if full_name.split() else full_name,
        'sortable_name': full_name,
        'email': email,
        'login_id': email,
        'role': 'admin' if admin else 'user',
        'created_at': utc_now_iso(),
    }
    state['users'][str(user_id)] = user
    if token:
        state['token_map'][token] = user_id
    return user


def _create_page(state: dict[str, Any], *, course_id: int, title: str, body: str) -> dict[str, Any]:
    page_id = _next_id(state, 'page')
    page_url = 'front_page' if slugify(title) == 'front-page' else f'{slugify(title)}-{page_id}'
    page = {
        'id': page_id,
        'course_id': course_id,
        'title': title,
        'url': page_url,
        'body': body,
        'published': True,
        'updated_at': utc_now_iso(),
    }
    state['pages'][str(page_id)] = page
    return page


def _ensure_seeded(state: dict[str, Any]) -> None:
    if state.get('seeded'):
        return

    admin_tokens = [
        ('mcpcanvasadmintoken1', CASE_TOKENS.get('admin_email_address') or 'mcpcanvasadmin1@mcp.com'),
        ('mcpcanvasadmintoken2', CASE_TOKENS.get('admin_email_address') or 'mcpcanvasadmin2@mcp.com'),
        ('mcpcanvasadmintoken3', CASE_TOKENS.get('admin_email_address') or 'mcpcanvasadmin3@mcp.com'),
    ]
    for token, email in admin_tokens:
        _create_user(state, email=email, token=token, admin=True)

    if token := CASE_TOKENS.get('admin_canvas_token'):
        _create_user(
            state,
            email=CASE_TOKENS.get('admin_email_address') or 'canvas-admin@mcp.com',
            token=str(token),
            admin=True,
        )
    if token := CASE_TOKENS.get('admin_canvas_api_token'):
        _create_user(
            state,
            email=CASE_TOKENS.get('admin_email_address') or 'canvas-admin@mcp.com',
            token=str(token),
            admin=True,
        )
    if token := CASE_TOKENS.get('canvas_api_token'):
        hinted_email = (
            CASE_TOKENS.get('canvas_user_name')
            or CASE_TOKENS.get('teacher_email')
            or CASE_TOKENS.get('admin_email_address')
            or 'canvas-user@mcp.com'
        )
        _create_user(state, email=str(hinted_email), token=str(token), admin=str(token).startswith('mcpcanvasadmin'))

    for row in _iter_case_users():
        _create_user(
            state,
            email=row['email'],
            name=row['name'] or None,
            token=row['token'] or None,
        )

    state['seeded'] = True


def _user_for_token(state: dict[str, Any], token: str) -> dict[str, Any]:
    _ensure_seeded(state)
    if token in state['token_map']:
        return state['users'][str(state['token_map'][token])]
    email = f'{slugify(token) or "canvas-user"}@canvas.mock'
    return _create_user(state, email=email, token=token, admin=token.startswith('mcpcanvasadmin'))


def _auth_token(request: web.Request) -> str:
    header = request.headers.get('Authorization', '')
    if header.startswith('Bearer '):
        return header[len('Bearer ') :].strip()
    return header.strip()


def _current_user(state: dict[str, Any], request: web.Request) -> dict[str, Any]:
    token = _auth_token(request)
    if not token:
        return _create_user(state, email='anonymous@canvas.mock')
    return _user_for_token(state, token)


def _json(data: Any, *, status: int = 200) -> web.Response:
    return web.json_response(data, status=status)


def _course_payload(course: dict[str, Any]) -> dict[str, Any]:
    return {
        'id': course['id'],
        'name': course['name'],
        'course_code': course.get('course_code', ''),
        'workflow_state': course.get('workflow_state', 'unpublished'),
        'account_id': course.get('account_id', 1),
        'is_public': bool(course.get('is_public', False)),
        'is_public_to_auth_users': bool(course.get('is_public_to_auth_users', False)),
        'public_syllabus': bool(course.get('public_syllabus', False)),
        'public_syllabus_to_auth': bool(course.get('public_syllabus_to_auth', False)),
        'syllabus_body': course.get('syllabus_body', ''),
    }


def _find_course(state: dict[str, Any], course_id: int) -> dict[str, Any] | None:
    return state['courses'].get(str(int(course_id)))


def _assignment_payload(assignment: dict[str, Any], *, submission: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        'id': assignment['id'],
        'course_id': assignment['course_id'],
        'name': assignment['name'],
        'description': assignment.get('description', ''),
        'points_possible': assignment.get('points_possible', 100),
        'published': bool(assignment.get('published', False)),
        'due_at': assignment.get('due_at'),
        'submission_types': assignment.get('submission_types', []),
        'allowed_extensions': assignment.get('allowed_extensions', []),
    }
    if submission is not None:
        payload['submission'] = submission
    return payload


def _submission_payload(submission: dict[str, Any], assignment: dict[str, Any]) -> dict[str, Any]:
    return {
        'id': submission['id'],
        'user_id': submission['user_id'],
        'assignment_id': assignment['id'],
        'workflow_state': submission.get('workflow_state', 'unsubmitted'),
        'score': submission.get('score'),
        'grade': submission.get('grade'),
        'body': submission.get('body', ''),
        'submission_type': submission.get('submission_type'),
        'submitted_at': submission.get('submitted_at'),
        'attempt': submission.get('attempt', 0),
        'attachments': submission.get('attachments', []),
        'submission_body': submission.get('body', ''),
        'submission_attachments': submission.get('attachments', []),
    }


def _file_payload(file_obj: dict[str, Any]) -> dict[str, Any]:
    return {
        'id': file_obj['id'],
        'display_name': file_obj['name'],
        'filename': file_obj['name'],
        'size': file_obj.get('size', 0),
        'content-type': file_obj.get('content_type', 'application/octet-stream'),
        'url': file_obj['url'],
        'folder_id': file_obj.get('folder_id'),
    }


def _discussion_payload(topic: dict[str, Any]) -> dict[str, Any]:
    return {
        'id': topic['id'],
        'title': topic['title'],
        'message': topic.get('message', ''),
        'course_id': topic['course_id'],
        'is_announcement': bool(topic.get('is_announcement', True)),
        'published': bool(topic.get('published', True)),
        'delayed_post_at': topic.get('delayed_post_at'),
        'entries': topic.get('entries', []),
    }


def _quiz_payload(quiz: dict[str, Any]) -> dict[str, Any]:
    return {
        'id': quiz['id'],
        'course_id': quiz['course_id'],
        'title': quiz['title'],
        'description': quiz.get('description', ''),
        'quiz_type': quiz.get('quiz_type', 'assignment'),
        'time_limit': quiz.get('time_limit'),
        'shuffle_answers': bool(quiz.get('shuffle_answers', False)),
        'allowed_attempts': quiz.get('allowed_attempts', 1),
        'scoring_policy': quiz.get('scoring_policy', 'keep_highest'),
        'published': bool(quiz.get('published', False)),
        'points_possible': quiz.get('points_possible', 0),
        'due_at': quiz.get('due_at'),
        'unlock_at': quiz.get('unlock_at'),
        'lock_at': quiz.get('lock_at'),
        'questions': quiz.get('questions', []),
    }


def _quiz_submission_payload(submission: dict[str, Any], quiz: dict[str, Any]) -> dict[str, Any]:
    return {
        'id': submission['id'],
        'quiz_id': quiz['id'],
        'user_id': submission['user_id'],
        'attempt': submission.get('attempt', 1),
        'score': submission.get('score'),
        'workflow_state': submission.get('workflow_state', 'untaken'),
        'submitted_at': submission.get('submitted_at'),
        'submission_id': submission['id'],
    }


def _conversation_summary(conversation: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    messages = conversation.get('messages', [])
    participants = [
        state['users'][str(user_id)]
        for user_id in conversation.get('participant_ids', [])
        if str(user_id) in state['users']
    ]
    return {
        'id': conversation['id'],
        'subject': conversation.get('subject', ''),
        'participants': [{'id': user['id'], 'name': user['name'], 'email': user['email']} for user in participants],
        'messages': messages,
        'last_message': messages[-1]['body'] if messages else '',
    }


def _find_user_by_any(state: dict[str, Any], recipient: str) -> dict[str, Any] | None:
    if recipient.isdigit():
        return state['users'].get(str(int(recipient)))
    for user in state['users'].values():
        if user.get('email') == recipient:
            return user
    return None


async def _request_json(request: web.Request) -> dict[str, Any]:
    if request.content_type.startswith('application/json'):
        try:
            data = await request.json()
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    try:
        post = await request.post()
        return dict(post)
    except Exception:
        return {}


async def health(_: web.Request) -> web.Response:
    return _json({'status': 'ok'})


async def list_courses(request: web.Request) -> web.Response:
    def build(state: dict[str, Any]) -> list[dict[str, Any]]:
        _ensure_seeded(state)
        include_deleted = any('deleted' in value for value in request.query.getall('include', []))
        result = []
        for course in state['courses'].values():
            if course.get('workflow_state') == 'deleted' and not include_deleted:
                continue
            result.append(_course_payload(course))
        return result

    return _json(mutate_store(_store_path(), _default_state, build))


async def accounts_courses(request: web.Request) -> web.Response:
    account_id = int(request.match_info['account_id'])
    if request.method == 'POST':
        payload = await _request_json(request)

        def create(state: dict[str, Any]) -> dict[str, Any]:
            _ensure_seeded(state)
            course_data = dict(payload.get('course', {}))
            course_id = _next_id(state, 'course')
            course = {
                'id': course_id,
                'account_id': account_id,
                'name': course_data.get('name', f'Course {course_id}'),
                'course_code': course_data.get('course_code', ''),
                'workflow_state': 'unpublished',
                'is_public': bool(course_data.get('is_public', False)),
                'is_public_to_auth_users': bool(course_data.get('is_public_to_auth_users', False)),
                'public_syllabus': bool(course_data.get('public_syllabus', False)),
                'public_syllabus_to_auth': bool(course_data.get('public_syllabus_to_auth', False)),
                'syllabus_body': course_data.get('syllabus_body', ''),
                'created_at': utc_now_iso(),
                'teacher_ids': [],
                'student_ids': [],
                'assignment_ids': [],
                'discussion_ids': [],
                'quiz_ids': [],
            }
            state['courses'][str(course_id)] = course
            _create_page(state, course_id=course_id, title='Front Page', body=course['syllabus_body'])
            return _course_payload(course)

        return _json(mutate_store(_store_path(), _default_state, create))

    search_term = request.query.get('search_term', '').strip().lower()

    def list_for_account(state: dict[str, Any]) -> list[dict[str, Any]]:
        _ensure_seeded(state)
        rows = []
        for course in state['courses'].values():
            if int(course.get('account_id', 1)) != account_id:
                continue
            if (
                search_term
                and search_term not in course['name'].lower()
                and search_term not in course.get('course_code', '').lower()
            ):
                continue
            rows.append(_course_payload(course))
        return rows

    return _json(mutate_store(_store_path(), _default_state, list_for_account))


async def course_item(request: web.Request) -> web.Response:
    course_id = int(request.match_info['course_id'])
    if request.method == 'GET':

        def get_course(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            _ensure_seeded(state)
            course = _find_course(state, course_id)
            if course is None:
                return 404, {'message': 'course not found'}
            return 200, _course_payload(course)

        status, payload = mutate_store(_store_path(), _default_state, get_course)
        return _json(payload, status=status)

    if request.method == 'DELETE':
        event = request.query.get('event', 'delete')
        payload = {'course': {'event': event}}
    else:
        payload = await _request_json(request)

    def update_course(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        _ensure_seeded(state)
        course = _find_course(state, course_id)
        if course is None:
            return 404, {'message': 'course not found'}
        course_update = dict(payload.get('course', {}))
        if not course_update:
            course_update = {key: value for key, value in payload.items() if key != 'course'}
        event = course_update.get('event') or payload.get('event')
        workflow_state = course_update.get('workflow_state') or payload.get('workflow_state')
        if event == 'offer':
            course['workflow_state'] = 'available'
        elif event == 'delete':
            course['workflow_state'] = 'deleted'
            return 200, {'delete': True, 'success': True}
        elif event == 'conclude':
            course['workflow_state'] = 'completed'
            return 200, {'conclude': True, 'success': True}
        elif workflow_state in {'available', 'deleted', 'completed', 'unpublished'}:
            course['workflow_state'] = workflow_state
        elif course_update.get('offer') is False:
            course['workflow_state'] = 'unpublished'
        for key in ('name', 'course_code', 'syllabus_body', 'default_view', 'public_description'):
            if key in course_update:
                course[key] = course_update[key]
        if 'is_public' in course_update:
            course['is_public'] = bool(course_update['is_public'])
        return 200, _course_payload(course)

    status, response = mutate_store(_store_path(), _default_state, update_course)
    return _json(response, status=status)


async def users_self(request: web.Request) -> web.Response:
    def current(state: dict[str, Any]) -> dict[str, Any]:
        return _current_user(state, request)

    return _json(mutate_store(_store_path(), _default_state, current))


async def accounts_users(request: web.Request) -> web.Response:
    int(request.match_info['account_id'])
    if request.method == 'POST':
        payload = await _request_json(request)

        def create(state: dict[str, Any]) -> dict[str, Any]:
            _ensure_seeded(state)
            user_data = payload.get('user', {})
            pseudonym = payload.get('pseudonym', {})
            email = (
                pseudonym.get('unique_id') or user_data.get('email') or f'user-{_next_id(state, "user")}@canvas.mock'
            )
            name = user_data.get('name') or _name_from_email(email)
            return _create_user(state, email=email, name=name)

        return _json(mutate_store(_store_path(), _default_state, create))

    search_term = request.query.get('search_term', '').strip().lower()

    def list_users(state: dict[str, Any]) -> list[dict[str, Any]]:
        _ensure_seeded(state)
        rows = []
        for user in state['users'].values():
            if search_term and search_term not in user['email'].lower() and search_term not in user['name'].lower():
                continue
            rows.append(user)
        return rows

    return _json(mutate_store(_store_path(), _default_state, list_users))


async def course_users(request: web.Request) -> web.Response:
    course_id = int(request.match_info['course_id'])

    def build(state: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
        _ensure_seeded(state)
        course = _find_course(state, course_id)
        if course is None:
            return 404, []
        user_ids = set(course.get('teacher_ids', [])) | set(course.get('student_ids', []))
        if not user_ids:
            user_ids = {user['id'] for user in state['users'].values()}
        return 200, [state['users'][str(user_id)] for user_id in sorted(user_ids) if str(user_id) in state['users']]

    status, payload = mutate_store(_store_path(), _default_state, build)
    return _json(payload, status=status)


async def enrollments(request: web.Request) -> web.Response:
    course_id = int(request.match_info['course_id'])
    if request.method == 'POST':
        payload = await _request_json(request)

        def create(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            _ensure_seeded(state)
            course = _find_course(state, course_id)
            if course is None:
                return 404, {'message': 'course not found'}
            enrollment_data = dict(payload.get('enrollment', {}))
            user_id = int(enrollment_data.get('user_id'))
            if str(user_id) not in state['users']:
                return 404, {'message': 'user not found'}
            enrollment_id = _next_id(state, 'enrollment')
            enrollment = {
                'id': enrollment_id,
                'course_id': course_id,
                'user_id': user_id,
                'type': enrollment_data.get('type', 'StudentEnrollment'),
                'enrollment_state': enrollment_data.get('enrollment_state', 'active'),
                'user': state['users'][str(user_id)],
            }
            state['enrollments'][str(enrollment_id)] = enrollment
            if enrollment['type'] == 'TeacherEnrollment':
                if user_id not in course['teacher_ids']:
                    course['teacher_ids'].append(user_id)
            else:
                if user_id not in course['student_ids']:
                    course['student_ids'].append(user_id)
            return 200, enrollment

        status, payload_out = mutate_store(_store_path(), _default_state, create)
        return _json(payload_out, status=status)

    def list_enrollments(state: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
        _ensure_seeded(state)
        course = _find_course(state, course_id)
        if course is None:
            return 404, []
        rows = [enrollment for enrollment in state['enrollments'].values() if int(enrollment['course_id']) == course_id]
        return 200, rows

    status, payload = mutate_store(_store_path(), _default_state, list_enrollments)
    return _json(payload, status=status)


async def enrollment_item(request: web.Request) -> web.Response:
    course_id = int(request.match_info['course_id'])
    enrollment_id = int(request.match_info['enrollment_id'])

    def handle(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        _ensure_seeded(state)
        course = _find_course(state, course_id)
        enrollment = state['enrollments'].get(str(enrollment_id))
        if course is None or enrollment is None or int(enrollment.get('course_id', 0)) != course_id:
            return 404, {'message': 'enrollment not found'}
        if request.method == 'DELETE':
            state['enrollments'].pop(str(enrollment_id), None)
            user_id = int(enrollment.get('user_id', 0))
            if enrollment.get('type') == 'TeacherEnrollment':
                course['teacher_ids'] = [value for value in course.get('teacher_ids', []) if int(value) != user_id]
            else:
                course['student_ids'] = [value for value in course.get('student_ids', []) if int(value) != user_id]
            return 200, {'success': True, 'deleted': True}
        return 200, enrollment

    status, payload = mutate_store(_store_path(), _default_state, handle)
    return _json(payload, status=status)


async def assignments(request: web.Request) -> web.Response:
    course_id = int(request.match_info['course_id'])
    if request.method == 'POST':
        payload = await _request_json(request)

        def create(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            _ensure_seeded(state)
            course = _find_course(state, course_id)
            if course is None:
                return 404, {'message': 'course not found'}
            assignment_data = dict(payload.get('assignment', {}))
            assignment_id = _next_id(state, 'assignment')
            assignment = {
                'id': assignment_id,
                'course_id': course_id,
                'name': assignment_data.get('name', f'Assignment {assignment_id}'),
                'description': assignment_data.get('description', ''),
                'points_possible': assignment_data.get('points_possible', 100),
                'due_at': assignment_data.get('due_at'),
                'published': bool(assignment_data.get('published', False)),
                'submission_types': assignment_data.get('submission_types', ['online_text_entry']),
                'allowed_extensions': assignment_data.get('allowed_extensions', []),
                'submissions': {},
            }
            state['assignments'][str(assignment_id)] = assignment
            course['assignment_ids'].append(assignment_id)
            return 200, _assignment_payload(assignment)

        status, payload_out = mutate_store(_store_path(), _default_state, create)
        return _json(payload_out, status=status)

    include_submission = request.query.get('include_submission') in {'1', 'true', 'True'}

    def list_assignments(state: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
        _ensure_seeded(state)
        course = _find_course(state, course_id)
        if course is None:
            return 404, []
        current_user = _current_user(state, request)
        rows = []
        for assignment_id in course.get('assignment_ids', []):
            assignment = state['assignments'].get(str(assignment_id))
            if assignment is None:
                continue
            submission = assignment['submissions'].get(str(current_user['id'])) if include_submission else None
            rows.append(
                _assignment_payload(
                    assignment, submission=None if submission is None else _submission_payload(submission, assignment)
                )
            )
        return 200, rows

    status, payload = mutate_store(_store_path(), _default_state, list_assignments)
    return _json(payload, status=status)


async def assignment_item(request: web.Request) -> web.Response:
    course_id = int(request.match_info['course_id'])
    assignment_id = int(request.match_info['assignment_id'])
    include_submission = request.query.get('include_submission') in {'1', 'true', 'True'}

    if request.method == 'GET':

        def get_assignment(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            _ensure_seeded(state)
            assignment = state['assignments'].get(str(assignment_id))
            if assignment is None or int(assignment['course_id']) != course_id:
                return 404, {'message': 'assignment not found'}
            current_user = _current_user(state, request)
            submission = assignment['submissions'].get(str(current_user['id'])) if include_submission else None
            return 200, _assignment_payload(
                assignment, submission=None if submission is None else _submission_payload(submission, assignment)
            )

        status, payload = mutate_store(_store_path(), _default_state, get_assignment)
        return _json(payload, status=status)

    payload = await _request_json(request)

    def update_assignment(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        _ensure_seeded(state)
        assignment = state['assignments'].get(str(assignment_id))
        if assignment is None or int(assignment['course_id']) != course_id:
            return 404, {'message': 'assignment not found'}
        if request.method == 'DELETE':
            state['assignments'].pop(str(assignment_id), None)
            course = _find_course(state, course_id)
            if course is not None:
                course['assignment_ids'] = [
                    value for value in course.get('assignment_ids', []) if int(value) != assignment_id
                ]
            return 200, {'success': True}
        assignment_data = dict(payload.get('assignment', {}))
        for key in ('name', 'description', 'due_at'):
            if key in assignment_data:
                assignment[key] = assignment_data[key]
        if 'points_possible' in assignment_data:
            assignment['points_possible'] = assignment_data['points_possible']
        if 'published' in assignment_data:
            assignment['published'] = bool(assignment_data['published'])
        return 200, _assignment_payload(assignment)

    status, payload_out = mutate_store(_store_path(), _default_state, update_assignment)
    return _json(payload_out, status=status)


async def assignment_submissions(request: web.Request) -> web.Response:
    course_id = int(request.match_info['course_id'])
    assignment_id = int(request.match_info['assignment_id'])
    if request.method == 'POST':
        payload = await _request_json(request)

        def submit(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            _ensure_seeded(state)
            assignment = state['assignments'].get(str(assignment_id))
            if assignment is None or int(assignment['course_id']) != course_id:
                return 404, {'message': 'assignment not found'}
            user = _current_user(state, request)
            submission_data = dict(payload.get('submission', {}))
            current = assignment['submissions'].get(str(user['id']))
            attempt = int(current.get('attempt', 0)) + 1 if current else 1
            attachments = []
            for file_id in submission_data.get('file_ids', []) or []:
                file_obj = state['files'].get(str(int(file_id)))
                if file_obj:
                    attachments.append({'id': file_obj['id'], 'filename': file_obj['name'], 'url': file_obj['url']})
            submission = {
                'id': current['id'] if current else _next_id(state, 'submission'),
                'user_id': user['id'],
                'submission_type': submission_data.get('submission_type', 'online_text_entry'),
                'body': submission_data.get('body') or '',
                'url': submission_data.get('url') or '',
                'attachments': attachments,
                'workflow_state': 'submitted',
                'submitted_at': utc_now_iso(),
                'attempt': attempt,
                'score': current.get('score') if current else None,
                'grade': current.get('grade') if current else None,
            }
            assignment['submissions'][str(user['id'])] = submission
            return 200, _submission_payload(submission, assignment)

        status, payload_out = mutate_store(_store_path(), _default_state, submit)
        return _json(payload_out, status=status)

    def list_items(state: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
        _ensure_seeded(state)
        assignment = state['assignments'].get(str(assignment_id))
        if assignment is None or int(assignment['course_id']) != course_id:
            return 404, []
        return 200, [_submission_payload(submission, assignment) for submission in assignment['submissions'].values()]

    status, payload = mutate_store(_store_path(), _default_state, list_items)
    return _json(payload, status=status)


async def assignment_submission_item(request: web.Request) -> web.Response:
    course_id = int(request.match_info['course_id'])
    assignment_id = int(request.match_info['assignment_id'])
    user_key = request.match_info['user_key']
    payload = await _request_json(request) if request.method == 'PUT' else {}

    def build(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        _ensure_seeded(state)
        assignment = state['assignments'].get(str(assignment_id))
        if assignment is None or int(assignment['course_id']) != course_id:
            return 404, {'message': 'assignment not found'}
        if user_key == 'self':
            user = _current_user(state, request)
            submission = assignment['submissions'].get(str(user['id']))
        else:
            submission = None
            for candidate in assignment['submissions'].values():
                if str(candidate['user_id']) == user_key or str(candidate['id']) == user_key:
                    submission = candidate
                    break
        if submission is None:
            return 404, {'message': 'submission not found'}
        if request.method == 'PUT':
            submission_update = dict(payload.get('submission', {}))
            if 'posted_grade' in submission_update:
                try:
                    submission['score'] = float(submission_update['posted_grade'])
                except Exception:
                    submission['score'] = submission_update['posted_grade']
                submission['grade'] = str(submission_update['posted_grade'])
            comment = submission_update.get('comment', {})
            if comment:
                submission.setdefault('comments', []).append(comment)
            submission['workflow_state'] = 'graded'
        return 200, _submission_payload(submission, assignment)

    status, payload_out = mutate_store(_store_path(), _default_state, build)
    return _json(payload_out, status=status)


async def course_files(request: web.Request) -> web.Response:
    course_id = int(request.match_info['course_id'])
    if request.method == 'POST':
        payload = await _request_json(request)

        def create_file(state: dict[str, Any]) -> dict[str, Any]:
            _ensure_seeded(state)
            file_id = _next_id(state, 'file')
            name = payload.get('name') or f'file-{file_id}.txt'
            file_obj = {
                'id': file_id,
                'course_id': course_id,
                'name': name,
                'size': payload.get('size', 0),
                'content_type': payload.get('content_type', 'application/octet-stream'),
                'content': payload.get('content', ''),
                'url': f'http://127.0.0.1:{request.url.port or 10001}/canvas/files/{file_id}',
            }
            state['files'][str(file_id)] = file_obj
            return _file_payload(file_obj)

        return _json(mutate_store(_store_path(), _default_state, create_file))

    def list_files(state: dict[str, Any]) -> list[dict[str, Any]]:
        _ensure_seeded(state)
        return [
            _file_payload(file_obj)
            for file_obj in state['files'].values()
            if int(file_obj.get('course_id', 0)) == course_id
        ]

    return _json(mutate_store(_store_path(), _default_state, list_files))


async def file_item(request: web.Request) -> web.Response:
    file_id = int(request.match_info['file_id'])

    def get_file(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        _ensure_seeded(state)
        file_obj = state['files'].get(str(file_id))
        if file_obj is None:
            return 404, {'message': 'file not found'}
        return 200, _file_payload(file_obj)

    status, payload = mutate_store(_store_path(), _default_state, get_file)
    return _json(payload, status=status)


async def file_download(request: web.Request) -> web.Response:
    file_id = int(request.match_info['file_id'])

    def get_bytes(state: dict[str, Any]) -> tuple[int, bytes, str]:
        _ensure_seeded(state)
        file_obj = state['files'].get(str(file_id))
        if file_obj is None:
            return 404, b'', 'text/plain'
        raw = file_obj.get('content', '')
        if isinstance(raw, str):
            return 200, raw.encode('utf-8'), file_obj.get('content_type', 'application/octet-stream')
        return 200, bytes(raw), file_obj.get('content_type', 'application/octet-stream')

    status, payload, content_type = mutate_store(_store_path(), _default_state, get_bytes)
    return web.Response(status=status, body=payload, content_type=content_type)


async def folders(request: web.Request) -> web.Response:
    course_id = int(request.match_info['course_id'])
    return _json([{'id': course_id * 1000 + 1, 'name': 'course files', 'parent_folder_id': None}])


async def pages(request: web.Request) -> web.Response:
    course_id = int(request.match_info['course_id'])

    def list_pages(state: dict[str, Any]) -> list[dict[str, Any]]:
        _ensure_seeded(state)
        rows = []
        for page in state['pages'].values():
            if int(page['course_id']) != course_id:
                continue
            rows.append({'page_id': page['id'], 'title': page['title'], 'url': page['url'], 'body': page['body']})
        return rows

    return _json(mutate_store(_store_path(), _default_state, list_pages))


async def page_item(request: web.Request) -> web.Response:
    course_id = int(request.match_info['course_id'])
    page_url = request.match_info['page_url']

    def get_page(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        _ensure_seeded(state)
        for page in state['pages'].values():
            if int(page['course_id']) == course_id and page['url'] == page_url:
                return 200, {'page_id': page['id'], 'title': page['title'], 'url': page['url'], 'body': page['body']}
        return 404, {'message': 'page not found'}

    status, payload = mutate_store(_store_path(), _default_state, get_page)
    return _json(payload, status=status)


async def discussion_topics(request: web.Request) -> web.Response:
    course_id = int(request.match_info['course_id'])
    if request.method == 'POST':
        payload = await _request_json(request)

        def create(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            _ensure_seeded(state)
            course = _find_course(state, course_id)
            if course is None:
                return 404, {'message': 'course not found'}
            topic_id = _next_id(state, 'discussion')
            topic = {
                'id': topic_id,
                'course_id': course_id,
                'title': payload.get('title', f'Topic {topic_id}'),
                'message': payload.get('message', ''),
                'is_announcement': bool(payload.get('is_announcement', True)),
                'published': bool(payload.get('published', True)),
                'delayed_post_at': payload.get('delayed_post_at'),
                'entries': [],
            }
            state['discussions'][str(topic_id)] = topic
            course['discussion_ids'].append(topic_id)
            return 200, _discussion_payload(topic)

        status, payload_out = mutate_store(_store_path(), _default_state, create)
        return _json(payload_out, status=status)

    only_announcements = request.query.get('only_announcements') == 'true'

    def list_topics(state: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
        _ensure_seeded(state)
        course = _find_course(state, course_id)
        if course is None:
            return 404, []
        rows = []
        for topic_id in course.get('discussion_ids', []):
            topic = state['discussions'].get(str(topic_id))
            if topic is None:
                continue
            if only_announcements and not topic.get('is_announcement', False):
                continue
            rows.append(_discussion_payload(topic))
        return 200, rows

    status, payload = mutate_store(_store_path(), _default_state, list_topics)
    return _json(payload, status=status)


async def discussion_item(request: web.Request) -> web.Response:
    course_id = int(request.match_info['course_id'])
    topic_id = int(request.match_info['topic_id'])
    payload = await _request_json(request) if request.method == 'PUT' else {}

    def handle(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        _ensure_seeded(state)
        topic = state['discussions'].get(str(topic_id))
        if topic is None or int(topic['course_id']) != course_id:
            return 404, {'message': 'topic not found'}
        if request.method == 'DELETE':
            state['discussions'].pop(str(topic_id), None)
            course = _find_course(state, course_id)
            if course is not None:
                course['discussion_ids'] = [
                    value for value in course.get('discussion_ids', []) if int(value) != topic_id
                ]
            return 200, {'success': True}
        if request.method == 'PUT':
            for key in ('title', 'message'):
                if key in payload:
                    topic[key] = payload[key]
        return 200, _discussion_payload(topic)

    status, payload_out = mutate_store(_store_path(), _default_state, handle)
    return _json(payload_out, status=status)


async def discussion_entries(request: web.Request) -> web.Response:
    course_id = int(request.match_info['course_id'])
    topic_id = int(request.match_info['topic_id'])
    payload = await _request_json(request)

    def create_entry(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        _ensure_seeded(state)
        topic = state['discussions'].get(str(topic_id))
        if topic is None or int(topic['course_id']) != course_id:
            return 404, {'message': 'topic not found'}
        user = _current_user(state, request)
        entry = {
            'id': _next_id(state, 'entry'),
            'user_id': user['id'],
            'message': payload.get('message') or payload.get('body') or '',
            'created_at': utc_now_iso(),
        }
        topic['entries'].append(entry)
        return 200, entry

    status, payload_out = mutate_store(_store_path(), _default_state, create_entry)
    return _json(payload_out, status=status)


async def quizzes(request: web.Request) -> web.Response:
    course_id = int(request.match_info['course_id'])
    if request.method == 'POST':
        payload = await _request_json(request)

        def create(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            _ensure_seeded(state)
            course = _find_course(state, course_id)
            if course is None:
                return 404, {'message': 'course not found'}
            quiz_data = dict(payload.get('quiz', {}))
            quiz_id = _next_id(state, 'quiz')
            quiz = {
                'id': quiz_id,
                'course_id': course_id,
                'title': quiz_data.get('title', f'Quiz {quiz_id}'),
                'description': quiz_data.get('description', ''),
                'quiz_type': quiz_data.get('quiz_type', 'assignment'),
                'time_limit': quiz_data.get('time_limit'),
                'shuffle_answers': bool(quiz_data.get('shuffle_answers', False)),
                'allowed_attempts': quiz_data.get('allowed_attempts', 1),
                'scoring_policy': quiz_data.get('scoring_policy', 'keep_highest'),
                'published': bool(quiz_data.get('published', False)),
                'points_possible': quiz_data.get('points_possible', 0),
                'due_at': quiz_data.get('due_at'),
                'unlock_at': quiz_data.get('unlock_at'),
                'lock_at': quiz_data.get('lock_at'),
                'questions': [],
                'submissions': {},
            }
            state['quizzes'][str(quiz_id)] = quiz
            course['quiz_ids'].append(quiz_id)
            return 200, _quiz_payload(quiz)

        status, payload_out = mutate_store(_store_path(), _default_state, create)
        return _json(payload_out, status=status)

    def list_items(state: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
        _ensure_seeded(state)
        course = _find_course(state, course_id)
        if course is None:
            return 404, []
        rows = []
        for quiz_id in course.get('quiz_ids', []):
            quiz = state['quizzes'].get(str(quiz_id))
            if quiz is None:
                continue
            rows.append(_quiz_payload(quiz))
        return 200, rows

    status, payload = mutate_store(_store_path(), _default_state, list_items)
    return _json(payload, status=status)


async def quiz_item(request: web.Request) -> web.Response:
    course_id = int(request.match_info['course_id'])
    quiz_id = int(request.match_info['quiz_id'])
    payload = await _request_json(request) if request.method == 'PUT' else {}

    def handle(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        _ensure_seeded(state)
        quiz = state['quizzes'].get(str(quiz_id))
        if quiz is None or int(quiz['course_id']) != course_id:
            return 404, {'message': 'quiz not found'}
        if request.method == 'DELETE':
            state['quizzes'].pop(str(quiz_id), None)
            course = _find_course(state, course_id)
            if course is not None:
                course['quiz_ids'] = [value for value in course.get('quiz_ids', []) if int(value) != quiz_id]
            return 200, {'success': True}
        if request.method == 'PUT':
            quiz_data = dict(payload.get('quiz', {}))
            for key in ('title', 'description', 'due_at', 'unlock_at', 'lock_at'):
                if key in quiz_data:
                    quiz[key] = quiz_data[key]
            for key in ('time_limit', 'points_possible', 'allowed_attempts'):
                if key in quiz_data:
                    quiz[key] = quiz_data[key]
            if 'published' in quiz_data:
                quiz['published'] = bool(quiz_data['published'])
        return 200, _quiz_payload(quiz)

    status, payload_out = mutate_store(_store_path(), _default_state, handle)
    return _json(payload_out, status=status)


async def quiz_questions(request: web.Request) -> web.Response:
    course_id = int(request.match_info['course_id'])
    quiz_id = int(request.match_info['quiz_id'])
    if request.method == 'POST':
        payload = await _request_json(request)

        def create_question(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            _ensure_seeded(state)
            quiz = state['quizzes'].get(str(quiz_id))
            if quiz is None or int(quiz['course_id']) != course_id:
                return 404, {'message': 'quiz not found'}
            question_data = dict(payload.get('question', {}))
            question = {
                'id': _next_id(state, 'question'),
                'question_name': question_data.get('question_name', ''),
                'question_text': question_data.get('question_text', ''),
                'question_type': question_data.get('question_type', 'multiple_choice_question'),
                'points_possible': question_data.get('points_possible', 1),
                'answers': question_data.get('answers', []),
            }
            quiz['questions'].append(question)
            return 200, question

        status, payload_out = mutate_store(_store_path(), _default_state, create_question)
        return _json(payload_out, status=status)

    def list_questions(state: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
        _ensure_seeded(state)
        quiz = state['quizzes'].get(str(quiz_id))
        if quiz is None or int(quiz['course_id']) != course_id:
            return 404, []
        return 200, list(quiz.get('questions', []))

    status, payload = mutate_store(_store_path(), _default_state, list_questions)
    return _json(payload, status=status)


async def quiz_question_item(request: web.Request) -> web.Response:
    course_id = int(request.match_info['course_id'])
    quiz_id = int(request.match_info['quiz_id'])
    question_id = int(request.match_info['question_id'])

    def get_question(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        _ensure_seeded(state)
        quiz = state['quizzes'].get(str(quiz_id))
        if quiz is None or int(quiz['course_id']) != course_id:
            return 404, {'message': 'quiz not found'}
        for question in quiz.get('questions', []):
            if int(question['id']) == question_id:
                return 200, question
        return 404, {'message': 'question not found'}

    status, payload = mutate_store(_store_path(), _default_state, get_question)
    return _json(payload, status=status)


def _score_quiz(quiz: dict[str, Any], answers: Any) -> float:
    if isinstance(answers, list):
        answers_by_question = {
            str(item.get('question_id')): item.get('answer') or item.get('answer_id')
            for item in answers
            if isinstance(item, dict)
        }
    elif isinstance(answers, dict):
        answers_by_question = {str(key): value for key, value in answers.items()}
    else:
        answers_by_question = {}
    score = 0.0
    for question in quiz.get('questions', []):
        correct_ids = {
            str(answer.get('id')) for answer in question.get('answers', []) if float(answer.get('weight', 0) or 0) > 0
        }
        user_answer = answers_by_question.get(str(question['id']))
        if user_answer is None:
            continue
        if isinstance(user_answer, list):
            if {str(item) for item in user_answer} == correct_ids:
                score += float(question.get('points_possible', 1))
        elif str(user_answer) in correct_ids:
            score += float(question.get('points_possible', 1))
    return score


async def quiz_submissions(request: web.Request) -> web.Response:
    course_id = int(request.match_info['course_id'])
    quiz_id = int(request.match_info['quiz_id'])
    if request.method == 'POST':

        def start_attempt(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            _ensure_seeded(state)
            quiz = state['quizzes'].get(str(quiz_id))
            if quiz is None or int(quiz['course_id']) != course_id:
                return 404, {'message': 'quiz not found'}
            user = _current_user(state, request)
            existing = quiz['submissions'].get(str(user['id']))
            attempt = int(existing.get('attempt', 0)) + 1 if existing else 1
            submission = {
                'id': existing['id'] if existing else _next_id(state, 'quiz_submission'),
                'user_id': user['id'],
                'attempt': attempt,
                'workflow_state': 'untaken',
                'score': None,
                'submitted_at': None,
            }
            quiz['submissions'][str(user['id'])] = submission
            return 200, {'quiz_submissions': [_quiz_submission_payload(submission, quiz)]}

        status, payload = mutate_store(_store_path(), _default_state, start_attempt)
        return _json(payload, status=status)

    def list_items(state: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
        _ensure_seeded(state)
        quiz = state['quizzes'].get(str(quiz_id))
        if quiz is None or int(quiz['course_id']) != course_id:
            return 404, []
        return 200, [_quiz_submission_payload(submission, quiz) for submission in quiz['submissions'].values()]

    status, payload = mutate_store(_store_path(), _default_state, list_items)
    return _json(payload, status=status)


async def quiz_submission_self(request: web.Request) -> web.Response:
    course_id = int(request.match_info['course_id'])
    quiz_id = int(request.match_info['quiz_id'])

    def get_self(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        _ensure_seeded(state)
        quiz = state['quizzes'].get(str(quiz_id))
        if quiz is None or int(quiz['course_id']) != course_id:
            return 404, {'message': 'quiz not found'}
        user = _current_user(state, request)
        submission = quiz['submissions'].get(str(user['id']))
        if submission is None:
            return 404, {'message': 'submission not found'}
        return 200, _quiz_submission_payload(submission, quiz)

    status, payload = mutate_store(_store_path(), _default_state, get_self)
    return _json(payload, status=status)


def _extract_quiz_answers(payload: dict[str, Any]) -> dict[str, Any]:
    questions: dict[str, Any] = {}
    quiz_submission = payload.get('quiz_submission')
    if isinstance(quiz_submission, dict):
        nested_questions = quiz_submission.get('questions')
        if isinstance(nested_questions, dict):
            questions.update({str(key): value for key, value in nested_questions.items()})
        for key, value in quiz_submission.items():
            key_str = str(key)
            if key_str.startswith('question_'):
                questions[key_str.split('_', 1)[1]] = value
    quiz_submissions = payload.get('quiz_submissions')
    if isinstance(quiz_submissions, list) and quiz_submissions:
        first = quiz_submissions[0]
        if isinstance(first, dict) and isinstance(first.get('questions'), dict):
            questions.update({str(key): value for key, value in first['questions'].items()})
    return questions


async def quiz_submission_item(request: web.Request) -> web.Response:
    course_id = int(request.match_info['course_id'])
    quiz_id = int(request.match_info['quiz_id'])
    submission_id = int(request.match_info['submission_id'])
    payload = await _request_json(request) if request.method in {'PUT', 'POST'} else {}

    def handle(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        _ensure_seeded(state)
        quiz = state['quizzes'].get(str(quiz_id))
        if quiz is None or int(quiz['course_id']) != course_id:
            return 404, {'message': 'quiz not found'}
        submission = None
        for candidate in quiz['submissions'].values():
            if int(candidate['id']) == submission_id:
                submission = candidate
                break
        if submission is None:
            return 404, {'message': 'submission not found'}
        if request.method == 'GET':
            return 200, _quiz_submission_payload(submission, quiz)
        questions = _extract_quiz_answers(payload)
        submission['score'] = _score_quiz(quiz, questions)
        submission['workflow_state'] = 'complete'
        submission['submitted_at'] = utc_now_iso()
        submission['answers'] = questions
        return 200, _quiz_submission_payload(submission, quiz)

    status, payload_out = mutate_store(_store_path(), _default_state, handle)
    return _json(payload_out, status=status)


async def quiz_statistics(request: web.Request) -> web.Response:
    course_id = int(request.match_info['course_id'])
    quiz_id = int(request.match_info['quiz_id'])

    def stats(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        _ensure_seeded(state)
        quiz = state['quizzes'].get(str(quiz_id))
        if quiz is None or int(quiz['course_id']) != course_id:
            return 404, {'message': 'quiz not found'}
        scores = [
            float(submission['score'])
            for submission in quiz['submissions'].values()
            if submission.get('score') is not None
        ]
        return 200, {
            'quiz_id': quiz_id,
            'submission_count': len(scores),
            'average_score': sum(scores) / len(scores) if scores else None,
        }

    status, payload = mutate_store(_store_path(), _default_state, stats)
    return _json(payload, status=status)


async def conversations(request: web.Request) -> web.Response:
    if request.method == 'POST':
        payload = await _request_json(request)

        def create(state: dict[str, Any]) -> list[dict[str, Any]]:
            _ensure_seeded(state)
            sender = _current_user(state, request)
            raw_recipients = payload.get('recipients') or payload.get('recipients[]') or []
            recipients = raw_recipients if isinstance(raw_recipients, list) else [raw_recipients]
            participant_ids = {sender['id']}
            for recipient in recipients:
                recipient_user = _find_user_by_any(state, str(recipient))
                if recipient_user is None and '@' in str(recipient):
                    recipient_user = _create_user(state, email=str(recipient))
                if recipient_user is not None:
                    participant_ids.add(recipient_user['id'])
            conversation = {
                'id': _next_id(state, 'conversation'),
                'subject': payload.get('subject', ''),
                'participant_ids': sorted(participant_ids),
                'messages': [
                    {
                        'id': _next_id(state, 'message'),
                        'author_id': sender['id'],
                        'body': payload.get('body', ''),
                        'created_at': utc_now_iso(),
                        'subject': payload.get('subject', ''),
                    }
                ],
            }
            state['conversations'][str(conversation['id'])] = conversation
            return [_conversation_summary(conversation, state)]

        return _json(mutate_store(_store_path(), _default_state, create))

    def list_items(state: dict[str, Any]) -> list[dict[str, Any]]:
        _ensure_seeded(state)
        user = _current_user(state, request)
        rows = []
        for conversation in state['conversations'].values():
            if user['id'] in conversation.get('participant_ids', []):
                rows.append(_conversation_summary(conversation, state))
        return rows

    return _json(mutate_store(_store_path(), _default_state, list_items))


async def conversation_item(request: web.Request) -> web.Response:
    conversation_id = int(request.match_info['conversation_id'])

    def get_item(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        _ensure_seeded(state)
        conversation = state['conversations'].get(str(conversation_id))
        if conversation is None:
            return 404, {'message': 'conversation not found'}
        return 200, _conversation_summary(conversation, state)

    status, payload = mutate_store(_store_path(), _default_state, get_item)
    return _json(payload, status=status)


async def conversation_add_message(request: web.Request) -> web.Response:
    conversation_id = int(request.match_info['conversation_id'])
    payload = await _request_json(request)

    def add_message(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        _ensure_seeded(state)
        conversation = state['conversations'].get(str(conversation_id))
        if conversation is None:
            return 404, {'message': 'conversation not found'}
        user = _current_user(state, request)
        conversation['messages'].append({
            'id': _next_id(state, 'message'),
            'author_id': user['id'],
            'body': payload.get('body', ''),
            'created_at': utc_now_iso(),
            'subject': conversation.get('subject', ''),
        })
        return 200, _conversation_summary(conversation, state)

    status, payload_out = mutate_store(_store_path(), _default_state, add_message)
    return _json(payload_out, status=status)


async def modules(_: web.Request) -> web.Response:
    return _json([])


async def module_item(request: web.Request) -> web.Response:
    if request.method == 'DELETE':
        return _json({'success': True, 'deleted': True})
    return _json({'message': 'module not found'}, status=404)


async def module_done(_: web.Request) -> web.Response:
    return _json({'completed': True})


async def rubrics(_: web.Request) -> web.Response:
    return _json([])


async def rubric_item(_: web.Request) -> web.Response:
    return _json({'message': 'rubric not found'}, status=404)


async def upcoming_assignments(request: web.Request) -> web.Response:
    def build(state: dict[str, Any]) -> list[dict[str, Any]]:
        _ensure_seeded(state)
        rows: list[dict[str, Any]] = []
        for assignment in state['assignments'].values():
            course = _find_course(state, int(assignment['course_id']))
            rows.append({
                'course_id': assignment['course_id'],
                'course_name': course['name'] if course else '',
                **_assignment_payload(assignment),
            })
        return rows

    return _json(mutate_store(_store_path(), _default_state, build))


async def dashboard(request: web.Request) -> web.Response:
    return await list_courses(request)


async def dashboard_cards(request: web.Request) -> web.Response:
    return await list_courses(request)


async def course_grades(request: web.Request) -> web.Response:
    return _json({})


async def notifications(_: web.Request) -> web.Response:
    return _json([])


async def account_item(request: web.Request) -> web.Response:
    account_id = int(request.match_info['account_id'])
    return _json({'id': account_id, 'name': f'Canvas Mock Account {account_id}'})


async def account_scopes(request: web.Request) -> web.Response:
    account_id = int(request.match_info['account_id'])
    return _json({'id': account_id, 'features': []})


async def account_sub_accounts(request: web.Request) -> web.Response:
    return _json([])


async def account_reports(request: web.Request) -> web.Response:
    return _json([])


async def account_report_item(request: web.Request) -> web.Response:
    return _json({'status': 'complete'})


def _build_app() -> web.Application:
    app = web.Application()
    app.router.add_get('/healthz', health)
    app.router.add_get('/api/v1/courses', list_courses)
    app.router.add_get('/api/v1/accounts/{account_id}/courses', accounts_courses)
    app.router.add_post('/api/v1/accounts/{account_id}/courses', accounts_courses)
    app.router.add_get('/api/v1/courses/{course_id}', course_item)
    app.router.add_put('/api/v1/courses/{course_id}', course_item)
    app.router.add_delete('/api/v1/courses/{course_id}', course_item)
    app.router.add_get('/api/v1/users/self', users_self)
    app.router.add_get('/api/v1/users/self/profile', users_self)
    app.router.add_put('/api/v1/users/self', users_self)
    app.router.add_get('/api/v1/accounts/{account_id}/users', accounts_users)
    app.router.add_post('/api/v1/accounts/{account_id}/users', accounts_users)
    app.router.add_get('/api/v1/courses/{course_id}/users', course_users)
    app.router.add_get('/api/v1/courses/{course_id}/enrollments', enrollments)
    app.router.add_post('/api/v1/courses/{course_id}/enrollments', enrollments)
    app.router.add_get('/api/v1/courses/{course_id}/enrollments/{enrollment_id}', enrollment_item)
    app.router.add_delete('/api/v1/courses/{course_id}/enrollments/{enrollment_id}', enrollment_item)
    app.router.add_get('/api/v1/courses/{course_id}/assignments', assignments)
    app.router.add_post('/api/v1/courses/{course_id}/assignments', assignments)
    app.router.add_get('/api/v1/courses/{course_id}/assignments/{assignment_id}', assignment_item)
    app.router.add_put('/api/v1/courses/{course_id}/assignments/{assignment_id}', assignment_item)
    app.router.add_delete('/api/v1/courses/{course_id}/assignments/{assignment_id}', assignment_item)
    app.router.add_get('/api/v1/courses/{course_id}/assignments/{assignment_id}/submissions', assignment_submissions)
    app.router.add_post('/api/v1/courses/{course_id}/assignments/{assignment_id}/submissions', assignment_submissions)
    app.router.add_get(
        '/api/v1/courses/{course_id}/assignments/{assignment_id}/submissions/{user_key}', assignment_submission_item
    )
    app.router.add_put(
        '/api/v1/courses/{course_id}/assignments/{assignment_id}/submissions/{user_key}', assignment_submission_item
    )
    app.router.add_get('/api/v1/courses/{course_id}/files', course_files)
    app.router.add_post('/api/v1/courses/{course_id}/files', course_files)
    app.router.add_get('/api/v1/files/{file_id}', file_item)
    app.router.add_get('/canvas/files/{file_id}', file_download)
    app.router.add_get('/api/v1/courses/{course_id}/folders', folders)
    app.router.add_get('/api/v1/courses/{course_id}/pages', pages)
    app.router.add_get('/api/v1/courses/{course_id}/pages/{page_url}', page_item)
    app.router.add_get('/api/v1/courses/{course_id}/discussion_topics', discussion_topics)
    app.router.add_post('/api/v1/courses/{course_id}/discussion_topics', discussion_topics)
    app.router.add_get('/api/v1/courses/{course_id}/discussion_topics/{topic_id}', discussion_item)
    app.router.add_put('/api/v1/courses/{course_id}/discussion_topics/{topic_id}', discussion_item)
    app.router.add_delete('/api/v1/courses/{course_id}/discussion_topics/{topic_id}', discussion_item)
    app.router.add_post('/api/v1/courses/{course_id}/discussion_topics/{topic_id}/entries', discussion_entries)
    app.router.add_get('/api/v1/courses/{course_id}/quizzes', quizzes)
    app.router.add_post('/api/v1/courses/{course_id}/quizzes', quizzes)
    app.router.add_get('/api/v1/courses/{course_id}/quizzes/{quiz_id}', quiz_item)
    app.router.add_put('/api/v1/courses/{course_id}/quizzes/{quiz_id}', quiz_item)
    app.router.add_delete('/api/v1/courses/{course_id}/quizzes/{quiz_id}', quiz_item)
    app.router.add_get('/api/v1/courses/{course_id}/quizzes/{quiz_id}/questions', quiz_questions)
    app.router.add_post('/api/v1/courses/{course_id}/quizzes/{quiz_id}/questions', quiz_questions)
    app.router.add_get('/api/v1/courses/{course_id}/quizzes/{quiz_id}/questions/{question_id}', quiz_question_item)
    app.router.add_get('/api/v1/courses/{course_id}/quizzes/{quiz_id}/submissions', quiz_submissions)
    app.router.add_post('/api/v1/courses/{course_id}/quizzes/{quiz_id}/submissions', quiz_submissions)
    app.router.add_get('/api/v1/courses/{course_id}/quizzes/{quiz_id}/submissions/self', quiz_submission_self)
    app.router.add_get(
        '/api/v1/courses/{course_id}/quizzes/{quiz_id}/submissions/{submission_id}', quiz_submission_item
    )
    app.router.add_put(
        '/api/v1/courses/{course_id}/quizzes/{quiz_id}/submissions/{submission_id}', quiz_submission_item
    )
    app.router.add_post(
        '/api/v1/courses/{course_id}/quizzes/{quiz_id}/submissions/{submission_id}/complete', quiz_submission_item
    )
    app.router.add_get('/api/v1/courses/{course_id}/quizzes/{quiz_id}/statistics', quiz_statistics)
    app.router.add_get('/api/v1/conversations', conversations)
    app.router.add_post('/api/v1/conversations', conversations)
    app.router.add_get('/api/v1/conversations/{conversation_id}', conversation_item)
    app.router.add_post('/api/v1/conversations/{conversation_id}/add_message', conversation_add_message)
    app.router.add_get('/api/v1/courses/{course_id}/modules', modules)
    app.router.add_get('/api/v1/courses/{course_id}/modules/{module_id}', module_item)
    app.router.add_delete('/api/v1/courses/{course_id}/modules/{module_id}', module_item)
    app.router.add_get('/api/v1/courses/{course_id}/modules/{module_id}/items', modules)
    app.router.add_get('/api/v1/courses/{course_id}/modules/{module_id}/items/{item_id}', module_item)
    app.router.add_delete('/api/v1/courses/{course_id}/modules/{module_id}/items/{item_id}', module_item)
    app.router.add_put('/api/v1/courses/{course_id}/modules/{module_id}/items/{item_id}/done', module_done)
    app.router.add_get('/api/v1/courses/{course_id}/rubrics', rubrics)
    app.router.add_get('/api/v1/courses/{course_id}/rubrics/{rubric_id}', rubric_item)
    app.router.add_get('/api/v1/users/self/upcoming_events', upcoming_assignments)
    app.router.add_get('/api/v1/users/self/dashboard_positions', dashboard)
    app.router.add_get('/api/v1/dashboard/dashboard_cards', dashboard_cards)
    app.router.add_get('/api/v1/courses/{course_id}/grades', course_grades)
    app.router.add_get('/api/v1/users/{user_id}/grades', course_grades)
    app.router.add_get('/api/v1/users/self/communication_channels', notifications)
    app.router.add_get('/api/v1/accounts/{account_id}', account_item)
    app.router.add_get('/api/v1/accounts/{account_id}/scopes', account_scopes)
    app.router.add_get('/api/v1/accounts/{account_id}/sub_accounts', account_sub_accounts)
    app.router.add_get('/api/v1/accounts/{account_id}/reports', account_reports)
    app.router.add_post('/api/v1/accounts/{account_id}/reports/{report}', account_report_item)
    app.router.add_get('/api/v1/accounts/{account_id}/reports/{report_type}/{report_id}', account_report_item)
    return app


def _ssl_context() -> ssl.SSLContext | None:
    if not CERT_PATH.exists() or not KEY_PATH.exists():
        return None
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain(str(CERT_PATH), str(KEY_PATH))
    return context


async def _run(host: str, ports: list[int]) -> None:
    runner = web.AppRunner(_build_app())
    await runner.setup()
    tls_context = _ssl_context()
    sites = [
        web.TCPSite(runner, host, port, ssl_context=tls_context if port in DEFAULT_TLS_PORTS else None)
        for port in ports
    ]
    for site in sites:
        await site.start()
    await web._run_app(asyncio.Future())  # type: ignore[attr-defined]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--ports', default=DEFAULT_PORTS)
    args = parser.parse_args()
    ports = [int(item) for item in str(args.ports).split(',') if item.strip()]
    if not ports:
        raise SystemExit('no ports configured')
    runner = web.AppRunner(_build_app())
    await runner.setup()
    tls_context = _ssl_context()
    sites = [
        web.TCPSite(runner, args.host, port, ssl_context=tls_context if port in DEFAULT_TLS_PORTS else None)
        for port in ports
    ]
    for site in sites:
        await site.start()
    while True:
        await asyncio.sleep(3600)


if __name__ == '__main__':
    asyncio.run(main())
