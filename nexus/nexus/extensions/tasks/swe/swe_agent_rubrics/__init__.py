"""SWE-agent Rubrics implementation for nexus framework.

This module provides a rubric-writing agent that analyzes code repositories
and gold patches to generate evaluation rubrics in YAML format.

Usage:
    from nexus.core import Registry
    Registry.import_module('nexus.extensions.tasks.swe.swe_agent_rubrics')

    agent_fn = agent_registry.get('swe_agent_rubrics')
"""
