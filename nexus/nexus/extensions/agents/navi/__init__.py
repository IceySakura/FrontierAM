"""Navi agent integration for WindowsAgentArena.

This module ports the WAA Navi agent into Nexus with the requested fixed
operating mode:

- ``obs_view=screen``: plan from full-screen observations.
- ``som_origin=a11y``: build Set-of-Marks from the accessibility tree.
- ``use_last_screen=True``: compare the previous raw screen with the current SoM image.
- ``action_space=code_block``: execute planner Python code through WAA's computer API.

Usage:
    from nexus.core import Registry
    Registry.import_module('nexus.extensions.agents.navi')

    agent_fn = agent_registry.get('navi_agent')
"""

from .agent import NaviAgentInput, NaviAgentOutput, run_navi_agent


__all__ = ['NaviAgentInput', 'NaviAgentOutput', 'run_navi_agent']
