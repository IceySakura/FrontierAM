"""Lightweight text maze environment focused on collecting many keys."""

from .env import (
    TreasureMazeConfig,
    TreasureMazeEnv,
    make_env,
    make_ordered_env,
    make_unordered_env,
)


__all__ = [
    'TreasureMazeConfig',
    'TreasureMazeEnv',
    'make_env',
    'make_ordered_env',
    'make_unordered_env',
]
