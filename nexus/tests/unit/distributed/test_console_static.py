from pathlib import Path


CONSOLE_SOURCE = Path(__file__).parents[3] / 'nexus/distributed/console/src/App.jsx'
CONSOLE_SOURCE_INDEX = Path(__file__).parents[3] / 'nexus/distributed/console/index.html'
CONSOLE_BUILD_INDEX = Path(__file__).parents[3] / 'nexus/distributed/static/console/index.html'
CONSOLE_ASSETS = Path(__file__).parents[3] / 'nexus/distributed/static/console/assets'
CONSOLE_PACKAGE = Path(__file__).parents[3] / 'nexus/distributed/console/package.json'


def test_console_is_built_from_react_vite_static_assets():
    package_json = CONSOLE_PACKAGE.read_text(encoding='utf-8')
    index_html = CONSOLE_BUILD_INDEX.read_text(encoding='utf-8')

    assert '"vite"' in package_json
    assert '"react"' in package_json
    assert '/v1/console/assets/' in index_html
    assert any(path.suffix == '.js' for path in CONSOLE_ASSETS.iterdir())
    assert any(path.suffix == '.css' for path in CONSOLE_ASSETS.iterdir())


def test_console_uses_summary_execution_list_and_reuses_live_event_source():
    source = CONSOLE_SOURCE.read_text(encoding='utf-8')

    assert 'summary: "1"' in source
    assert 'eventSourceExecutionId' in source
    assert 'eventSourceExecutionId === item.execution_id' in source
    assert '/live/events?cursor=' in source
    assert '/v1/executions/${encodeURIComponent(item.execution_id)}/events?cursor=' not in source


def test_console_ignores_stale_execution_detail_and_event_stream_updates():
    source = CONSOLE_SOURCE.read_text(encoding='utf-8')
    query = (CONSOLE_SOURCE.parent / 'resourceQuery.js').read_text(encoding='utf-8')

    assert 'selectedExecutionIdRef' in source
    assert 'signal?.aborted || selectedExecutionIdRef.current !== executionId' in source
    assert 'if (selectedExecutionIdRef.current === executionId) return;' in source
    assert 'if (selectedExecutionIdRef.current !== executionId) return;' in source
    assert 'eventSourceRef.current !== source || selectedExecutionIdRef.current !== item.execution_id' in source
    assert '!signal.aborted && id === generation.current' in query
    assert 'state.key === key && enabled' in query
    assert 'generation.current++;' in query
    assert 'poll.stop();' in query


def test_console_uses_backend_cursor_pagination_and_panel_scrollers():
    source = CONSOLE_SOURCE.read_text(encoding='utf-8')
    pager = (CONSOLE_SOURCE.parent / 'ResourceUI.jsx').read_text(encoding='utf-8')

    assert 'limit: q.page_size || "50"' in source
    assert '["q", "status", "cursor", "mode"]' in source
    assert 'top_level: "1"' in source
    assert 'nextCursor={query.data?.next_cursor}' in source
    assert 'cursorNavigation(route, nextCursor)' in pager
    assert 'pagination-footer' in pager
    assert 'liveEventsRef.current.scrollTop = liveEventsRef.current.scrollHeight' in source
    assert 'eventsBottomRef.current?.scrollIntoView' not in source


def test_console_stops_reconnecting_terminal_live_event_sources():
    source = CONSOLE_SOURCE.read_text(encoding='utf-8')

    assert 'terminalEventSourceIdsRef' in source
    assert 'terminalEventSourceIdsRef.current.has(item.execution_id)' in source
    assert 'terminalEventSourceIdsRef.current.add(item.execution_id)' in source
    assert 'isTerminalLiveEvent(type)' in source


def test_console_sends_browser_credentials_to_cross_origin_application_gateway():
    source = CONSOLE_SOURCE.read_text(encoding='utf-8')

    assert source.count('credentials: "include"') == 2
    assert '{ withCredentials: true }' in source


def test_console_shows_submitted_live_turns_and_refreshes_command_status():
    source = CONSOLE_SOURCE.read_text(encoding='utf-8')

    assert '"turn.submitted"' in source
    assert '"turn.started"' in source
    assert 'await refreshSelectedExecution(currentExecution.execution_id)' in source
    assert 'await refreshControlEvents(currentExecution.execution_id)' in source


def test_console_only_builds_debug_links_from_structured_runtime_entries():
    source = CONSOLE_SOURCE.read_text(encoding='utf-8')

    assert 'const runtimeNames = Object.keys(runtimeInfo.runtimes || runtimeInfo || {})' not in source
    assert 'getRuntimeEntries(runtimeInfo)' in source
    assert 'isRuntimeDebuggable' in source
    assert '/proxy/debug/gui`' not in source
    assert '/runtimes/${encodeURIComponent(name)}/proxy`' in source


def test_console_has_typed_submission_form_and_advanced_json_entry():
    source = CONSOLE_SOURCE.read_text(encoding='utf-8')

    assert 'id="new-submission"' in source
    assert 'New Submission' in source
    assert 'buildSubmission(draft)' in source
    assert '/v1/submissions/${encodeURIComponent(type)}' in source
    assert 'Raw submission JSON' in source
    assert 'Advanced overrides' in source
    assert 'Host environment (one KEY=VALUE per line)' in source
    assert 'value={draft.hostEnv}' in source


def test_console_exposes_stop_events_manager_groups_and_system_status():
    source = CONSOLE_SOURCE.read_text(encoding='utf-8')

    assert 'navigationItems(isAdmin)' in source
    assert 'admin-tab' in source
    assert 'data-tab={key}' in source
    assert 'id="system-view"' in source
    overview = (CONSOLE_SOURCE.parent / 'PlatformOverview.jsx').read_text(encoding='utf-8')
    assert 'PlatformOverview' in source
    assert '/v1/system/summary' in overview
    assert '/v1/system/status' in overview
    assert 'usePoll(loadLive, 2000, active)' in overview
    assert 'usePoll(loadSnapshot, 30000, active)' in overview
    assert 'request("PUT", controlPath("/v1/system/config-overrides")' in source
    assert 'request("DELETE", controlPath("/v1/system/config-overrides")' in source
    assert 'Save configuration' in source
    assert 'Secret values are write-only' in source
    assert 'Pending changes' in source
    assert 'diffConfigValues' in source
    assert 'No changes from the loaded version.' in source
    assert 'Snapshot · current work' in overview
    assert 'MetricHistory' in overview

    assert 'Stop Run' in source
    assert 'stopExecution(' in source
    assert '/stop`' in source
    assert 'canStopExecution' in source
    assert 'row-action' in source

    assert 'data-detail-tab' in source
    assert 'ControlEventsTab' in source
    assert '["control", "Events"]' in source
    assert 'Live Events' in source
    assert 'event-type-filter' in source
    assert 'toggle-event-stream' in source
    assert 'clear-live-events' in source
    assert 'className="event-item"' in source
    assert 'trace.event' in source

    managers = (CONSOLE_SOURCE.parent / 'ManagerInventory.jsx').read_text(encoding='utf-8')
    assert 'Healthy' in managers
    assert 'Lost / stale' in managers


def test_console_uses_manager_dropdown_filters_and_single_stale_delete():
    source = (CONSOLE_SOURCE.parent / 'ManagerInventory.jsx').read_text(encoding='utf-8')

    assert 'All environments' in source
    assert 'list="manager-pools"' in source
    assert 'Lost / stale' in source
    assert 'value === "lost" ? "lost"' in source
    assert (
        '/v1/managers/${encodeURIComponent(row.manager_id)}?session_id=${encodeURIComponent(row.session_id)}' in source
    )
    assert 'Delete stale record' in source
    assert '!row.healthy && row.reserved_slots > 0' in source
    assert 'clean stale' not in source.lower()


def test_console_can_drain_and_resume_current_manager_session():
    source = (CONSOLE_SOURCE.parent / 'ManagerInventory.jsx').read_text(encoding='utf-8')

    assert '"PATCH"' in source
    assert 'controlPath(`/v1/managers/${encodeURIComponent(row.manager_id)}`)' in source
    assert '{ draining, expected_session_id: row.session_id }' in source
    assert '"Resume"' in source
    assert '"Drain"' in source


def test_console_exposes_manager_log_tail_viewer():
    source = (CONSOLE_SOURCE.parent / 'ManagerInventory.jsx').read_text(encoding='utf-8')

    assert '<h3>Manager logs</h3>' in source
    assert 'manager-log-viewer' in source
    assert 'q.manager_id && session' in source
    assert '/v1/managers/${encodeURIComponent(q.manager_id)}/logs?' in source
    assert 'session_id: session, lines: q.lines || "200"' in source
    assert 'enabled: Boolean(logsPath)' in source


def test_console_exposes_execution_manager_logs_tab():
    source = CONSOLE_SOURCE.read_text(encoding='utf-8')

    assert '["manager-logs", "Logs"]' in source
    assert '["manager-logs", "Manager Logs"]' not in source
    assert 'data-detail-tab={key}' in source
    assert '/v1/executions/${encodeURIComponent(item.execution_id)}/logs?lines=' in source
    assert 'executionLogSourceLabel' in source
    assert 'function ExecutionManagerLogsTab(' in source


def test_console_build_contains_current_execution_logs_label():
    built_scripts = '\n'.join(path.read_text(encoding='utf-8') for path in CONSOLE_ASSETS.glob('*.js'))

    assert '["manager-logs","Logs"]' in built_scripts
    assert '["manager-logs","Manager Logs"]' not in built_scripts


def test_console_exposes_cloud_submit_access_resource_and_batch_detail_views():
    source = CONSOLE_SOURCE.read_text(encoding='utf-8')
    allocations = (CONSOLE_SOURCE.parent / 'AllocationViews.jsx').read_text(encoding='utf-8')

    assert '<BatchWorkspace' in source
    assert 'closeItem={() => updateRouteQuery(route, { item_id: undefined })}' in source
    assert 'fromBatch' in source
    assert '["trace-backends", "Trace backends"]' in source
    assert '["database-accounts", "Database accounts"]' in source
    assert '["managers", "Managers"], ["allocations", "Allocations"]' in source
    assert '["dynamic", "predefined", "all"]' in allocations
    assert '/v1/my-environments' in source
    assert '/v1/environment-grants' in source
    assert '/v1/allocations' in allocations
    assert '/force-cleanup?expected_version=' in allocations
    assert '/v1/environment-catalog' in source
    assert '/v1/my-environments/${encodeURIComponent(environmentId)}/quota`' in source
    assert 'Claim quota' in source
    assert 'TOKEN_PRESETS' not in source
    assert '<OneTimeToken token={createdToken}' not in source
    assert '(item.attempts || []).map' in source
    assert '/v1/executions/${encodeURIComponent(item.execution_id)}/result`' in source
    assert '/batch?include_status_counts=true' in source
    assert 'item_status_counts' in source
    assert 'Copy full principal' in source
    assert 'Unknown user' in source
    assert 'return item.phase || item.status || "unknown";' in source
    assert 'dependency_build?.status' in allocations
    assert '<th>Dependency build</th>' in allocations


def test_console_exposes_principal_run_trace_backend_grants_and_user_catalog():
    source = CONSOLE_SOURCE.read_text(encoding='utf-8')

    assert '["trace-backends", "Trace backends"]' in source
    assert '/v1/access/run-trace-backends' in source
    assert '/v1/admin/run-trace-backends' in source
    assert '/v1/admin/principals/${encodeURIComponent(principalId)}/run-trace-backends' in source
    assert 'function RunTraceBackendGrantsView(' in source
    assert 'function MyRunTraceBackendsTable(' in source
    assert 'buildRunTraceBackendGrantPayload' in source


def test_console_shared_access_uses_user_picker_and_hides_incoming_grants():
    source = CONSOLE_SOURCE.read_text(encoding='utf-8')

    assert 'visiblePrincipalGrants(grants, me?.principal_id || "", isAdmin)' in source
    assert 'placeholder="grantee principal_id"' not in source
    assert '<option value="">Select grantee</option>' in source
    assert 'const ownerUser = usersById.get(ownerPrincipalId);' in source
    assert 'value={ownerUser ? `${ownerUser.display_name || ownerUser.subject} · ${ownerPrincipalId}`' in source


def test_console_claims_default_environment_quota_with_the_implicit_system_token():
    source = CONSOLE_SOURCE.read_text(encoding='utf-8')
    compact_source = ' '.join(source.split())

    assert 'claimQuota={(environmentId) => claimDefaultQuota(environmentId).catch(showError)}' in compact_source


def test_console_does_not_present_the_implicit_system_token_as_a_credential():
    source = CONSOLE_SOURCE.read_text(encoding='utf-8')

    assert 'Credential:' not in source
    assert 'systemTokenLabel' not in source
    assert 'setSystemTokenLabel' not in source


def test_console_reuses_server_projected_execution_phase():
    source = CONSOLE_SOURCE.read_text(encoding='utf-8')

    assert 'return item.phase || item.status || "unknown";' in source
    assert 'dependencyBuildStatus' not in source
    assert 'unreleased = allocations.some' not in source


def test_console_declares_inline_favicon():
    source_index = CONSOLE_SOURCE_INDEX.read_text(encoding='utf-8')

    assert '<link rel="icon" href="data:image/svg+xml,' in source_index
