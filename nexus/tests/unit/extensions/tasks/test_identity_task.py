import pytest

from nexus.extensions.tasks.identity.task import IdentityTaskInput, identity_task


@pytest.mark.asyncio
async def test_identity_task_provides_scripted_user_inputs():
    seen: list[str | None] = []

    async def run_agent(**kwargs):
        get_user_input = kwargs['get_user_input']
        seen.append(await get_user_input('first hint'))
        seen.append(await get_user_input('second hint'))
        seen.append(await get_user_input('done hint'))

    await identity_task(IdentityTaskInput(run_agent=run_agent, user_inputs=['first', 'second']))

    assert seen == ['first', 'second', None]
