"""Stateful local stdio MCP server for Atari Pong."""

from __future__ import annotations

import io
import json
import os

from collections import deque
from dataclasses import dataclass
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp import Image as MCPImage


os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

DEFAULT_FRAME_WIDTH = 480
DEFAULT_FRAME_HEIGHT = 630
DEFAULT_INITIAL_NOOP_STEPS = 3
DEFAULT_MAX_DECISIONS = 1000

ACTION_MEANINGS = {
    0: 'NOOP',
    1: 'FIRE',
    2: 'move the right-side paddle UP',
    3: 'move the right-side paddle DOWN',
    4: 'move UP while pressing FIRE',
    5: 'move DOWN while pressing FIRE',
}


@dataclass(frozen=True)
class PongSnapshot:
    state: dict[str, Any]
    image_png: bytes


def _positive_int_from_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = int(raw)
    if value <= 0:
        raise ValueError(f'{name} must be positive, got {value}')
    return value


def _non_negative_int_from_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = int(raw)
    if value < 0:
        raise ValueError(f'{name} must be non-negative, got {value}')
    return value


def _create_pong_env() -> Any:
    import ale_py
    import gymnasium as gym

    gym.register_envs(ale_py)
    return gym.make(
        'ALE/Pong-v5',
        render_mode=None,
        repeat_action_probability=0.0,
    )


def make_contact_sheet(frames: list[Any], frame_size: tuple[int, int]) -> bytes:
    """Create one 2x2 PNG ordered from oldest to newest."""
    from PIL import Image as PILImage

    images = [
        PILImage.fromarray(frame).resize(
            frame_size,
            PILImage.Resampling.NEAREST,
        )
        for frame in frames
    ]
    frame_width, frame_height = frame_size
    sheet = PILImage.new('RGB', (frame_width * 2, frame_height * 2))
    positions = [
        (0, 0),
        (frame_width, 0),
        (0, frame_height),
        (frame_width, frame_height),
    ]
    for image, position in zip(images, positions, strict=False):
        sheet.paste(image, position)

    buffer = io.BytesIO()
    sheet.save(buffer, format='PNG')
    return buffer.getvalue()


class PongGame:
    """Own one isolated Pong episode inside one MCP process."""

    def __init__(
        self,
        *,
        env: Any | None = None,
        frame_size: tuple[int, int] = (DEFAULT_FRAME_WIDTH, DEFAULT_FRAME_HEIGHT),
        initial_noop_steps: int = DEFAULT_INITIAL_NOOP_STEPS,
        max_decisions: int = DEFAULT_MAX_DECISIONS,
    ) -> None:
        self.env = env if env is not None else _create_pong_env()
        self.frame_size = frame_size
        self.initial_noop_steps = initial_noop_steps
        self.max_decisions = max_decisions
        self.frames: deque[Any] = deque(maxlen=4)
        self.episode_return = 0.0
        self.decision = 0
        self.done = True
        self.terminated = False
        self.truncated = False
        self.done_reason: str | None = 'not_reset'

    def reset(self, seed: int = 42) -> PongSnapshot:
        observation, _ = self.env.reset(seed=seed)
        self.frames.clear()
        self.frames.append(observation.copy())
        self.episode_return = 0.0
        self.decision = 0
        self.done = False
        self.terminated = False
        self.truncated = False
        self.done_reason = None

        for _ in range(self.initial_noop_steps):
            if self.done:
                break
            observation, reward, terminated, truncated, _ = self.env.step(0)
            self.frames.append(observation.copy())
            self.episode_return += float(reward)
            self.terminated = bool(terminated)
            self.truncated = bool(truncated)
            self.done = self.terminated or self.truncated
            if self.done:
                self.done_reason = 'environment'

        while len(self.frames) < 4:
            self.frames.append(self.frames[-1].copy())
        return self._snapshot(reward=0.0)

    def step(self, action: int) -> PongSnapshot:
        if action not in ACTION_MEANINGS:
            raise ValueError('Action must be an integer from 0 through 5')
        if self.done:
            raise RuntimeError('Episode is finished or has not started; call pong_reset')

        observation, reward, terminated, truncated, _ = self.env.step(action)
        self.frames.append(observation.copy())
        self.episode_return += float(reward)
        self.decision += 1
        self.terminated = bool(terminated)
        self.truncated = bool(truncated)
        self.done = self.terminated or self.truncated
        if self.done:
            self.done_reason = 'environment'
        elif self.decision >= self.max_decisions:
            self.done = True
            self.truncated = True
            self.done_reason = 'decision_limit'

        return self._snapshot(reward=float(reward), action=action)

    def close(self) -> None:
        self.env.close()

    def _snapshot(self, *, reward: float, action: int | None = None) -> PongSnapshot:
        state = {
            'reward': reward,
            'episode_return': self.episode_return,
            'decision': self.decision,
            'done': self.done,
            'terminated': self.terminated,
            'truncated': self.truncated,
            'done_reason': self.done_reason,
            'last_action': action,
            'max_decisions': self.max_decisions,
            'action_meanings': ACTION_MEANINGS,
            'frame_order': [
                'top-left: oldest',
                'top-right',
                'bottom-left',
                'bottom-right: newest',
            ],
        }
        return PongSnapshot(
            state=state,
            image_png=make_contact_sheet(list(self.frames), self.frame_size),
        )


def _game_from_environment() -> PongGame:
    return PongGame(
        frame_size=(
            _positive_int_from_env('NEXUS_PONG_FRAME_WIDTH', DEFAULT_FRAME_WIDTH),
            _positive_int_from_env('NEXUS_PONG_FRAME_HEIGHT', DEFAULT_FRAME_HEIGHT),
        ),
        initial_noop_steps=_non_negative_int_from_env(
            'NEXUS_PONG_INITIAL_NOOP_STEPS',
            DEFAULT_INITIAL_NOOP_STEPS,
        ),
        max_decisions=_positive_int_from_env('NEXUS_PONG_MAX_DECISIONS', DEFAULT_MAX_DECISIONS),
    )


mcp = FastMCP('pong')
_game: PongGame | None = None


def _require_game() -> PongGame:
    global _game
    if _game is None:
        _game = _game_from_environment()
    return _game


def _tool_result(snapshot: PongSnapshot) -> list[Any]:
    return [
        json.dumps(snapshot.state, ensure_ascii=False, sort_keys=True),
        MCPImage(data=snapshot.image_png, format='png'),
    ]


@mcp.tool(name='reset')
def pong_reset(seed: int = 42):
    """Start a Pong episode and return state plus a 2x2 sheet of four consecutive frames."""
    return _tool_result(_require_game().reset(seed=seed))


@mcp.tool(name='step')
def pong_step(action: int):
    """Execute one action (0-5) and return state plus the latest four frames."""
    return _tool_result(_require_game().step(action=action))


def main() -> None:
    global _game
    try:
        mcp.run(transport='stdio')
    finally:
        if _game is not None:
            _game.close()
            _game = None


if __name__ == '__main__':
    main()
