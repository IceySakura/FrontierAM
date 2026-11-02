"""HITL recipes.

Recipes are ordinary Python modules that compose task-specific prompts,
handlers, actors, and loop helpers. They do not define a separate registry or
schema; concrete Nexus tasks import the recipe module they need.
"""
