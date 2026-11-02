"""CodeAct-agent implementation for nexus framework.

This agent implements the CodeAct pattern from OpenHands for solving software engineering tasks.
The agent loop follows:
1. Initialize with system prompt and tools
2. Query LLM for next action (tool calls)
3. Execute tools and collect observations
4. Check for exit conditions (finish tool, max iterations)
5. Continue until done
"""

import json
import logging

from enum import Enum

from pydantic import Field

from nexus.core import (
    AgentRunInput,
    AgentRunOutput,
    Registry,
    ToolExecutor,
    WithUserPrompt,
    agent_registry,
    require_context,
    tool_registry,
)
from nexus.extensions.agents.codeact.tools import get_file_editor_description


logger = logging.getLogger(__name__)


def _allows_tool_call_retry(client: object) -> bool:
    """Whether *client* can re-roll a malformed tool call instead of returning it again.

    HyTuner is the only one, and importing it costs numpy, which codeact does not
    otherwise need -- so the check stays lazy and tolerates the module being absent.
    """
    try:
        from nexus.extensions.openai_clients.hytuner import HyTunerClient
    except ImportError:
        return False
    return isinstance(client, HyTunerClient)


# Import tools to register them
Registry.import_module('nexus.extensions.agents.codeact.tools')


# ============== Execution Status ==============


class ExecutionStatus(str, Enum):
    """Execution status for CodeAct agent."""

    RUNNING = 'running'
    """Agent is actively running"""

    FINISHED = 'finished'
    """Agent finished normally (via finish tool or no tool calls)"""

    MAX_ITERATIONS = 'max_iterations'
    """Agent reached maximum iterations limit"""

    TOOL_ERROR = 'tool_error'
    """Abnormal exit - tool call failed (e.g., invalid JSON arguments)"""

    ERROR = 'error'
    """Agent encountered an error"""


# ============== Input/Output Models ==============


class CodeActAgentInput(AgentRunInput, WithUserPrompt):
    """Input configuration for CodeAct agent."""

    max_iterations: int = Field(default=100, description='Maximum number of agent steps')
    completion_retry_on_tool_error: int = Field(
        default=0,
        description='Number of retries when model output is invalid (no tool calls or invalid JSON). '
        '0 means no retry, fail immediately on first error.',
    )


class CodeActAgentOutput(AgentRunOutput):
    """Output from CodeAct agent run."""

    execution_status: ExecutionStatus = Field(description='Final execution status')
    iterations: int = Field(default=0, description='Number of iterations completed')


# ============== Default System Prompt ==============

# Security policy content (from security_policy.j2)
SECURITY_POLICY = """# 🔐 Security Policy

## OK to do without Explicit User Consent

- Download and run code from a repository specified by a user
- Open pull requests on the original repositories where the code is stored
- Install and run popular packages from pypi, npm, or other package managers
- Use APIs to work with GitHub or other platforms, unless the user asks otherwise or your task requires browsing

## Do only with Explicit User Consent

- Upload code to anywhere other than the location where it was obtained from
- Upload API keys or tokens anywhere, except when using them to authenticate with the appropriate service

## Never Do

- Never perform any illegal activities, such as circumventing security to access a system that is not under your control or performing denial-of-service attacks on external servers
- Never run software to mine cryptocurrency

## General Security Guidelines

- Only use GITHUB_TOKEN and other credentials in ways the user has explicitly requested and would expect"""

# Security risk assessment content (from security_risk_assessment.j2, cli_mode=True)
SECURITY_RISK_ASSESSMENT = """# Security Risk Policy
When using tools that support the security_risk parameter, assess the safety risk of your actions:


- **LOW**: Safe, read-only actions.
  - Viewing/summarizing content, reading project files, simple in-memory calculations.
- **MEDIUM**: Project-scoped edits or execution.
  - Modify user project files, run project scripts/tests, install project-local packages.
- **HIGH**: System-level or untrusted operations.
  - Changing system settings, global installs, elevated (`sudo`) commands, deleting critical files, downloading & executing untrusted code, or sending local secrets/data out.


**Global Rules**
- Always escalate to **HIGH** if sensitive data leaves the environment."""

DEFAULT_SYSTEM_PROMPT = f"""You are OpenHands agent, a helpful AI assistant that can interact with a computer to solve tasks.

<ROLE>
* Your primary role is to assist users by executing commands, modifying code, and solving technical problems effectively. You should be thorough, methodical, and prioritize quality over speed.
* If the user asks a question, like "why is X happening", don't try to fix the problem. Just give an answer to the question.
</ROLE>

<EFFICIENCY>
* Each action you take is somewhat expensive. Wherever possible, combine multiple actions into a single action, e.g. combine multiple bash commands into one, using sed and grep to edit/view multiple files at once.
* When exploring the codebase, use efficient tools like find, grep, and git commands with appropriate filters to minimize unnecessary operations.
</EFFICIENCY>

<FILE_SYSTEM_GUIDELINES>
* When a user provides a file path, do NOT assume it's relative to the current working directory. First explore the file system to locate the file before working on it.
* If asked to edit a file, edit the file directly, rather than creating a new file with a different filename.
* For global search-and-replace operations, consider using `sed` instead of opening file editors multiple times.
* NEVER create multiple versions of the same file with different suffixes (e.g., file_test.py, file_fix.py, file_simple.py). Instead:
  - Always modify the original file directly when making changes
  - If you need to create a temporary file for testing, delete it once you've confirmed your solution works
  - If you decide a file you created is no longer useful, delete it instead of creating a new version
* Do NOT include documentation files explaining your changes in version control unless the user explicitly requests it
* When reproducing bugs or implementing fixes, use a single file rather than creating multiple files with different versions
</FILE_SYSTEM_GUIDELINES>

<CODE_QUALITY>
* Write clean, efficient code with minimal comments. Avoid redundancy in comments: Do not repeat information that can be easily inferred from the code itself.
* When implementing solutions, focus on making the minimal changes needed to solve the problem.
* Before implementing any changes, first thoroughly understand the codebase through exploration.
* If you are adding a lot of code to a function or file, consider splitting the function or file into smaller pieces when appropriate.
* Place all imports at the top of the file unless explicitly requested otherwise or if placing imports at the top would cause issues (e.g., circular imports, conditional imports, or imports that need to be delayed for specific reasons).
</CODE_QUALITY>

<VERSION_CONTROL>
* If there are existing git user credentials already configured, use them and add Co-authored-by: openhands <openhands@all-hands.dev> to any commits messages you make. if a git config doesn't exist use "openhands" as the user.name and "openhands@all-hands.dev" as the user.email by default, unless explicitly instructed otherwise.
* Exercise caution with git operations. Do NOT make potentially dangerous changes (e.g., pushing to main, deleting repositories) unless explicitly asked to do so.
* When committing changes, use `git status` to see all modified files, and stage all files necessary for the commit. Use `git commit -a` whenever possible.
* Do NOT commit files that typically shouldn't go into version control (e.g., node_modules/, .env files, build directories, cache files, large binaries) unless explicitly instructed by the user.
* If unsure about committing certain files, check for the presence of .gitignore files or ask the user for clarification.
</VERSION_CONTROL>

<PULL_REQUESTS>
* **Important**: Do not push to the remote branch and/or start a pull request unless explicitly asked to do so.
* When creating pull requests, create only ONE per session/issue unless explicitly instructed otherwise.
* When working with an existing PR, update it with new commits rather than creating additional PRs for the same issue.
* When updating a PR, preserve the original PR title and purpose, updating description only when necessary.
</PULL_REQUESTS>

<PROBLEM_SOLVING_WORKFLOW>
1. EXPLORATION: Thoroughly explore relevant files and understand the context before proposing solutions
2. ANALYSIS: Consider multiple approaches and select the most promising one
3. TESTING:
   * For bug fixes: Create tests to verify issues before implementing fixes
   * For new features: Consider test-driven development when appropriate
   * Do NOT write tests for documentation changes, README updates, configuration files, or other non-functionality changes
   * If the repository lacks testing infrastructure and implementing tests would require extensive setup, consult with the user before investing time in building testing infrastructure
   * If the environment is not set up to run tests, consult with the user first before investing time to install all dependencies
4. IMPLEMENTATION:
   * Make focused, minimal changes to address the problem
   * Always modify existing files directly rather than creating new versions with different suffixes
   * If you create temporary files for testing, delete them after confirming your solution works
5. VERIFICATION: If the environment is set up to run tests, test your implementation thoroughly, including edge cases. If the environment is not set up to run tests, consult with the user first before investing time to run tests.
</PROBLEM_SOLVING_WORKFLOW>

<SECURITY>
{SECURITY_POLICY}
</SECURITY>


<SECURITY_RISK_ASSESSMENT>
{SECURITY_RISK_ASSESSMENT}
</SECURITY_RISK_ASSESSMENT>


<EXTERNAL_SERVICES>
* When interacting with external services like GitHub, GitLab, or Bitbucket, use their respective APIs instead of browser-based interactions whenever possible.
* Only resort to browser-based interactions with these services if specifically requested by the user or if the required operation cannot be performed via API.
</EXTERNAL_SERVICES>

<ENVIRONMENT_SETUP>
* When user asks you to run an application, don't stop if the application is not installed. Instead, please install the application and run the command again.
* If you encounter missing dependencies:
  1. First, look around in the repository for existing dependency files (requirements.txt, pyproject.toml, package.json, Gemfile, etc.)
  2. If dependency files exist, use them to install all dependencies at once (e.g., `pip install -r requirements.txt`, `npm install`, etc.)
  3. Only install individual packages directly if no dependency files are found or if only specific packages are needed
* Similarly, if you encounter missing dependencies for essential tools requested by the user, install them when possible.
</ENVIRONMENT_SETUP>

<TROUBLESHOOTING>
* If you've made repeated attempts to solve a problem but tests still fail or the user reports it's still broken:
  1. Step back and reflect on 5-7 different possible sources of the problem
  2. Assess the likelihood of each possible cause
  3. Methodically address the most likely causes, starting with the highest probability
  4. Explain your reasoning process in your response to the user
* When you run into any major issue while executing a plan from the user, please don't try to directly work around it. Instead, propose a new plan and confirm with the user before proceeding.
</TROUBLESHOOTING>

<PROCESS_MANAGEMENT>
* When terminating processes:
  - Do NOT use general keywords with commands like `pkill -f server` or `pkill -f python` as this might accidentally kill other important servers or processes
  - Always use specific keywords that uniquely identify the target process
  - Prefer using `ps aux` to find the exact process ID (PID) first, then kill that specific PID
  - When possible, use more targeted approaches like finding the PID from a pidfile or using application-specific shutdown commands
</PROCESS_MANAGEMENT>"""

# Default instance template for SWE-bench tasks (from default.j2)
DEFAULT_INSTANCE_TEMPLATE = """I have access to a python code repository in the directory {working_dir} . You can explore and modify files using the available tools. Consider the following issue description:

<issue_description>
{problem_statement}
</issue_description>

Can you help me implement the necessary changes to the repository so that the requirements specified in the <issue_description> are met?
I've already taken care of all changes to any of the test files described in the <issue_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!
Also the development Python environment is already set up for you (i.e., all dependencies already installed), so you don't need to install other packages.
Your task is to make the minimal changes to non-test files in the {working_dir} directory to ensure the <issue_description> is satisfied.

Follow these phases to resolve the issue:

Phase 1. READING: read the problem and reword it in clearer terms
   1.1 If there are code or config snippets. Express in words any best practices or conventions in them.
   1.2 Hightlight message errors, method names, variables, file names, stack traces, and technical details.
   1.3 Explain the problem in clear terms.
   1.4 Enumerate the steps to reproduce the problem.
   1.5 Hightlight any best practices to take into account when testing and fixing the issue

Phase 2. RUNNING: install and run the tests on the repository
   2.1 Activate the environment by running
       ./opt/miniconda3/etc/profile.d/conda.sh ; conda activate testbed
   2.2 Follow the readme
   2.3 Install the environment and anything needed
   2.4 Iterate and figure out how to run the tests

Phase 3. EXPLORATION: find the files that are related to the problem and possible solutions
   3.1 Use `grep` to search for relevant methods, classes, keywords and error messages.
   3.2 Identify all files related to the problem statement.
   3.3 Propose the methods and files to fix the issue and explain why.
   3.4 From the possible file locations, select the most likely location to fix the issue.

Phase 4. TEST CREATION: before implementing any fix, create a script to reproduce and verify the issue.
   4.1 Look at existing test files in the repository to understand the test format/structure.
   4.2 Create a minimal reproduction script that reproduces the located issue.
   4.3 Run the reproduction script to confirm you are reproducing the issue.
   4.4 Adjust the reproduction script as necessary.

Phase 5. FIX ANALYSIS: state clearly the problem and how to fix it
   5.1 State clearly what the problem is.
   5.2 State clearly where the problem is located.
   5.3 State clearly how the test reproduces the issue.
   5.4 State clearly the best practices to take into account in the fix.
   5.5 State clearly how to fix the problem.

Phase 6. FIX IMPLEMENTATION: Edit the source code to implement your chosen solution.
   6.1 Make minimal, focused changes to fix the issue.

Phase 7. VERIFICATION: Test your implementation thoroughly.
   7.1 Run your reproduction script to verify the fix works.
   7.2 Add edge cases to your test script to ensure comprehensive coverage.
   7.3 Run existing tests related to the modified code to ensure you haven't broken anything.

8. FINAL REVIEW: Carefully re-read the problem description and compare your changes with the base commit {base_commit}.
   8.1 Ensure you've fully addressed all requirements.
   8.2 Run any tests in the repository related to:
     8.2.1 The issue you are fixing
     8.2.2 The files you modified
     8.2.3 The functions you changed
   8.3 If any tests fail, revise your implementation until all tests pass

Be thorough in your exploration, testing, and reasoning. It's fine if your thinking process is lengthy - quality and completeness are more important than brevity.
"""


# ============== Agent Implementation ==============


@agent_registry.register('codeact')
async def run_codeact_agent(agent_input: CodeActAgentInput) -> CodeActAgentOutput:
    """Run CodeAct agent on a task.

    Implements the CodeAct agent loop:
    1. Initialize conversation with system prompt
    2. Add user message
    3. Loop:
       a. Call LLM with history and tools
       b. If no tool calls -> agent wants to respond/finish
       c. For each tool call:
          - Execute tool
          - Add observation to history
          - Check for finish condition
    4. Exit when: finish tool called, no tool calls, max iterations, or error

    When task_name is 'swe_task', the agent will use SWE-bench specific
    configuration from task_config:
    - repo: Repository name in "owner/repo" format
    - problem_statement: The issue description
    - base_commit: The base commit hash
    - workspace_base: (optional) Base workspace directory, default "/workspace"
    """
    ctx = require_context()
    client = ctx.openai_client

    # Create tool executor with CodeAct agent tools
    executor = ToolExecutor([
        tool_registry.get('codeact/terminal'),
        tool_registry.get('codeact/file_editor'),
        tool_registry.get('codeact/task_tracker'),
        tool_registry.get('codeact/think'),
        tool_registry.get('codeact/finish'),
    ])
    # Override from task_config if running as swe_task
    if ctx.task_name == 'swe_task':
        working_dir = ctx.task_config.get('working_dir')
        if not working_dir:
            raise ValueError('working_dir is required in task_config for swe_task')
        problem_statement = ctx.task_config.get('problem_statement')
        if not problem_statement:
            raise ValueError('problem_statement is required in task_config for swe_task')
        base_commit = ctx.task_config.get('base_commit')
        if not base_commit:
            raise ValueError('base_commit is required in task_config for swe_task')

        # Build instruction using SWE-bench template
        agent_input.user_prompt = DEFAULT_INSTANCE_TEMPLATE.format(
            working_dir=working_dir, problem_statement=problem_statement, base_commit=base_commit
        )

        # Dynamically set file_editor description with working directory
        executor.set_tool_override(
            'codeact/file_editor',
            description=get_file_editor_description(working_dir),
        )

    # Build initial history
    system_prompt = DEFAULT_SYSTEM_PROMPT
    history: list[dict] = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': agent_input.user_prompt},
    ]

    # Get tool schemas for LLM
    tools = executor.get_tool_schemas()

    execution_status: ExecutionStatus = ExecutionStatus.RUNNING
    iteration = 0

    try:
        while iteration < agent_input.max_iterations:
            logger.debug(f'=== CodeAct Agent Step {iteration + 1} ===')

            # Retry loop for handling model output errors (no tool calls or invalid JSON)
            max_retries = agent_input.completion_retry_on_tool_error
            for retry_attempt in range(max_retries + 1):
                # Make LLM completion call
                completion_kwargs = {}
                if max_retries > 0 and retry_attempt < max_retries and _allows_tool_call_retry(client):
                    completion_kwargs['allow_retry_for_tool_call'] = True
                response = await client.create_completion(
                    messages=history,
                    tools=tools or None,
                    **completion_kwargs,
                )
                message = response.choices[0].message

                # Log assistant content if present
                if message.content:
                    logger.debug(
                        f'Assistant: {message.content[:500]}...'
                        if len(message.content) > 500
                        else f'Assistant: {message.content}'
                    )

                # Check for tool error: no tool calls
                if not message.tool_calls:
                    if retry_attempt < max_retries:
                        logger.warning(f'No tool calls - model output error, retry {retry_attempt + 1}/{max_retries}')
                        continue  # Retry without adding to history
                    logger.error('No tool calls - model did not call any tool, all retries exhausted')
                    execution_status = ExecutionStatus.TOOL_ERROR
                    break

                # Check for tool error: invalid JSON in any tool call
                json_parse_error = None
                parsed_tool_args = []
                for tool_call in message.tool_calls:
                    try:
                        parsed_tool_args.append(json.loads(tool_call.function.arguments))
                    except json.JSONDecodeError as e:
                        json_parse_error = e
                        break

                if json_parse_error is not None:
                    if retry_attempt < max_retries:
                        logger.warning(
                            f'Failed to parse tool arguments: {json_parse_error}, '
                            f'retry {retry_attempt + 1}/{max_retries}'
                        )
                        continue  # Retry without adding to history
                    logger.error(f'Failed to parse tool arguments: {json_parse_error}, all retries exhausted')
                    execution_status = ExecutionStatus.TOOL_ERROR
                    break

                # Success: valid response with parseable tool calls
                break

            # If execution_status is set from retry loop, break main loop
            if execution_status != ExecutionStatus.RUNNING:
                break

            # Add assistant message to history (only after successful validation)
            history.append(message.model_dump(exclude_none=True))

            # Process tool calls
            for idx, tool_call in enumerate(message.tool_calls):
                tool_name = tool_call.function.name
                tool_args = parsed_tool_args[idx]

                logger.debug(f'Tool call: {tool_name}')
                logger.debug(f'Tool args: {json.dumps(tool_args, ensure_ascii=False)[:500]}')

                # Execute tool
                try:
                    tool_response = await executor.call(tool_name, tool_args)
                except Exception as e:
                    logger.error(f'Tool execution error: {e}')
                    history.append({
                        'role': 'tool',
                        'tool_call_id': tool_call.id,
                        'content': f'Error executing tool: {e}',
                    })
                    continue

                # Extract output text
                output_text = ''
                if tool_response.output:
                    for part in tool_response.output:
                        if isinstance(part, dict) and part.get('type') == 'text':
                            output_text += part.get('text', '')

                logger.debug(
                    f'Tool output: {output_text[:500]}...' if len(output_text) > 500 else f'Tool output: {output_text}'
                )

                # Add tool response to history
                history.append({
                    'role': 'tool',
                    'tool_call_id': tool_call.id,
                    'content': output_text or 'Tool executed successfully with no output.',
                })

                # Check for finish condition from tool
                tool_exit_status = ctx.tool_state.get('exit_status')
                if tool_exit_status == 'finished':
                    logger.info('Finish tool called - agent completed task')
                    execution_status = ExecutionStatus.FINISHED
                    break

            # Check if we should exit the main loop
            if execution_status != ExecutionStatus.RUNNING:
                break

            iteration += 1

    except Exception as e:
        logger.exception(f'Agent encountered error: {e}')
        execution_status = ExecutionStatus.ERROR

    # Check if we hit max iterations
    if execution_status == ExecutionStatus.RUNNING:
        execution_status = ExecutionStatus.MAX_ITERATIONS
        logger.warning(f'Agent reached max iterations ({agent_input.max_iterations})')

    logger.info(f'CodeAct-agent finished with status: {execution_status.value}, iterations: {iteration + 1}')

    return CodeActAgentOutput(
        execution_status=execution_status,
        iterations=iteration + 1,
    )
