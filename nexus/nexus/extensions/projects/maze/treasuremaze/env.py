"""A tiny, pure-Python text maze environment with many keys and wordy observations.

This module is intentionally self-contained. It avoids heavy abstractions and
keeps nearly all logic in one file:

- a small config dataclass;
- a room dataclass;
- iterative maze generation on a grid;
- simple item placement for keys and distractors;
- very wordy template-based observations;
- a compact ASCII map renderer;
- a tiny command parser and environment class.
"""

from __future__ import annotations

import random

from collections import deque
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass, field

from .observation_templates import (
    ObservationTemplateContext,
    build_observation_paragraphs,
)


DISTRACTOR_WORDS: list[str] = [
    'apron',
    'apple',
    'ashtray',
    'badge',
    'bag',
    'ball',
    'banana',
    'bandage',
    'basket',
    'battery',
    'beaker',
    'bell',
    'belt',
    'bench',
    'blanket',
    'block',
    'board',
    'bolt',
    'book',
    'bookmark',
    'boot',
    'bottle',
    'bowl',
    'box',
    'bracelet',
    'brick',
    'broom',
    'brush',
    'bucket',
    'button',
    'cable',
    'calendar',
    'camera',
    'candle',
    'cap',
    'carrot',
    'cart',
    'case',
    'chain',
    'chair',
    'chalk',
    'charger',
    'chart',
    'cloth',
    'coat',
    'coin',
    'comb',
    'compass',
    'cork',
    'crate',
    'cup',
    'cushion',
    'diary',
    'dish',
    'doll',
    'drawer',
    'drill',
    'drum',
    'envelope',
    'eraser',
    'fan',
    'file',
    'flask',
    'folder',
    'fork',
    'frame',
    'funnel',
    'gamepad',
    'garland',
    'glove',
    'goggles',
    'grater',
    'hammer',
    'hanger',
    'hat',
    'helmet',
    'hook',
    'horn',
    'hose',
    'jar',
    'jug',
    'kettle',
    'ladder',
    'ladle',
    'lamp',
    'lantern',
    'leaflet',
    'lemon',
    'lid',
    'locket',
    'magnet',
    'marker',
    'mask',
    'mat',
    'medal',
    'mirror',
    'mitt',
    'mug',
    'nail',
    'napkin',
    'needle',
    'net',
    'newspaper',
    'notebook',
    'olive',
    'orange',
    'ovenmitt',
    'package',
    'pad',
    'paintbrush',
    'paperclip',
    'pebble',
    'pen',
    'pencil',
    'pepper',
    'phone',
    'photo',
    'pillow',
    'pin',
    'pipe',
    'pitcher',
    'plate',
    'pliers',
    'pocketwatch',
    'postcard',
    'pot',
    'pouch',
    'racket',
    'radio',
    'rag',
    'ribbon',
    'ring',
    'rope',
    'ruler',
    'sack',
    'sandal',
    'saucer',
    'scarf',
    'scoop',
    'scraper',
    'screw',
    'screwdriver',
    'scroll',
    'seal',
    'seed',
    'shade',
    'shaker',
    'sheet',
    'shelf',
    'shirt',
    'shoe',
    'shovel',
    'sign',
    'sketchbook',
    'slate',
    'soap',
    'sock',
    'spade',
    'spanner',
    'spoon',
    'spring',
    'stamp',
    'star',
    'sticker',
    'stool',
    'strap',
    'string',
    'suitcase',
    'sweater',
    'switch',
    'tablecloth',
    'tag',
    'teacup',
    'teapot',
    'thermos',
    'thread',
    'ticket',
    'tin',
    'toaster',
    'token',
    'toolbox',
    'torch',
    'towel',
    'tray',
    'trowel',
    'tub',
    'umbrella',
    'vase',
    'wallet',
    'watch',
    'whistle',
    'wire',
    'wrench',
    'yarn',
    'zipper',
]

DIRECTION_VECTORS: dict[str, tuple[int, int]] = {
    'north': (-1, 0),
    'south': (1, 0),
    'east': (0, 1),
    'west': (0, -1),
}
OPPOSITE_DIRECTIONS: dict[str, str] = {
    'north': 'south',
    'south': 'north',
    'east': 'west',
    'west': 'east',
}
DIRECTION_ORDER: tuple[str, ...] = ('north', 'east', 'south', 'west')
KEY_PICKUP_REWARD = 0.0
SUCCESS_REWARD = 1.0


@dataclass(slots=True)
class TreasureMazeConfig:
    """Configuration for :class:`TreasureMazeEnv`."""

    size: int
    num_keys: int
    ordered_keys: bool
    seed: int | None
    min_distractors_per_room: int = 2
    max_steps: int | None = None

    def __post_init__(self) -> None:
        if self.size < 1:
            raise ValueError('size must be at least 1.')
        if self.size > 26:
            raise ValueError('size must be at most 26 (columns are labeled A–Z).')
        if self.num_keys < 1:
            raise ValueError('num_keys must be at least 1.')
        if self.num_keys > self.num_rooms:
            raise ValueError('num_keys cannot exceed size * size.')
        if self.min_distractors_per_room < 2:
            raise ValueError('min_distractors_per_room must be at least 2.')
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError('max_steps must be positive when provided.')

    @property
    def num_rooms(self) -> int:
        return self.size * self.size


@dataclass(slots=True)
class Room:
    """A simple room record."""

    room_id: int
    name: str
    row: int
    col: int
    exits: dict[str, int] = field(default_factory=dict)
    items: list[str] = field(default_factory=list)


def _neighbor_coords(row: int, col: int, rows: int, cols: int) -> list[tuple[int, int]]:
    neighbors: list[tuple[int, int]] = []
    for delta_row, delta_col in DIRECTION_VECTORS.values():
        next_row = row + delta_row
        next_col = col + delta_col
        if 0 <= next_row < rows and 0 <= next_col < cols:
            neighbors.append((next_row, next_col))
    return neighbors


def _direction_between(source: tuple[int, int], target: tuple[int, int]) -> str:
    row_delta = target[0] - source[0]
    col_delta = target[1] - source[1]
    for direction, (delta_row, delta_col) in DIRECTION_VECTORS.items():
        if (row_delta, col_delta) == (delta_row, delta_col):
            return direction
    raise ValueError(f'Coordinates {source} and {target} are not orthogonal neighbors.')


def _item_sort_key(item: str) -> tuple[int, str]:
    return (1 if item.startswith('key') else 0, item)


def _format_list(items: Iterable[str]) -> str:
    parts = list(items)
    if not parts:
        return 'none'
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f'{parts[0]} and {parts[1]}'
    return f'{", ".join(parts[:-1])}, and {parts[-1]}'


def _build_distances(rooms: dict[int, Room], start_room_id: int) -> dict[int, int]:
    distances = {start_room_id: 0}
    queue: deque[int] = deque([start_room_id])
    while queue:
        room_id = queue.popleft()
        room = rooms[room_id]
        for neighbor_id in room.exits.values():
            if neighbor_id in distances:
                continue
            distances[neighbor_id] = distances[room_id] + 1
            queue.append(neighbor_id)
    return distances


def _sample_distractors(rng: random.Random, minimum: int) -> list[str]:
    count = minimum + rng.randint(0, 2)
    return sorted(rng.sample(DISTRACTOR_WORDS, count))


def _choose_key_rooms(
    rooms: dict[int, Room],
    start_room_id: int,
    num_keys: int,
    rng: random.Random,
) -> list[int]:
    distances = _build_distances(rooms, start_room_id)
    candidates = [room_id for room_id in rooms if room_id != start_room_id]
    if len(candidates) < num_keys:
        candidates.append(start_room_id)

    shuffled = candidates[:]
    rng.shuffle(shuffled)
    shuffled.sort(key=lambda room_id: distances[room_id], reverse=True)

    far_pool_size = max(num_keys, len(shuffled) // 2)
    far_pool = shuffled[:far_pool_size]
    selected = [far_pool[(index * len(far_pool)) // num_keys] for index in range(num_keys)]

    unique_selected: list[int] = []
    for room_id in selected:
        if room_id not in unique_selected:
            unique_selected.append(room_id)

    if len(unique_selected) < num_keys:
        for room_id in far_pool:
            if room_id not in unique_selected:
                unique_selected.append(room_id)
            if len(unique_selected) == num_keys:
                break

    if len(unique_selected) < num_keys:
        for room_id in shuffled:
            if room_id not in unique_selected:
                unique_selected.append(room_id)
            if len(unique_selected) == num_keys:
                break

    rng.shuffle(unique_selected)
    return unique_selected


def _build_world(
    config: TreasureMazeConfig,
    seed_value: int,
) -> tuple[dict[int, Room], int, int, int]:
    rng = random.Random(seed_value)
    rows = cols = config.size
    start_coord = (rng.randrange(rows), rng.randrange(cols))

    all_coords = [(r, c) for r in range(rows) for c in range(cols)]
    ordered_coords = sorted(all_coords)
    coord_to_room_id = {coord: index for index, coord in enumerate(ordered_coords, start=1)}
    start_room_id = coord_to_room_id[start_coord]

    rooms: dict[int, Room] = {}
    for coord, room_id in coord_to_room_id.items():
        room_name = f'{coord[0] + 1}{chr(ord("A") + coord[1])}'
        rooms[room_id] = Room(
            room_id=room_id,
            name=room_name,
            row=coord[0],
            col=coord[1],
        )

    for coord in ordered_coords:
        for neighbor in _neighbor_coords(coord[0], coord[1], rows, cols):
            room_id = coord_to_room_id[coord]
            neighbor_id = coord_to_room_id[neighbor]
            direction = _direction_between(coord, neighbor)
            rooms[room_id].exits[direction] = neighbor_id

    for room in rooms.values():
        room.items.extend(_sample_distractors(rng, config.min_distractors_per_room))

    key_room_ids = _choose_key_rooms(rooms, start_room_id, config.num_keys, rng)
    for key_index, room_id in enumerate(key_room_ids, start=1):
        rooms[room_id].items.append(f'key{key_index}')

    for room in rooms.values():
        room.items.sort(key=_item_sort_key)

    return rooms, start_room_id, rows, cols


class TreasureMazeEnv:
    """A lightweight text-only maze environment with many ordered or unordered keys."""

    def __init__(self, config: TreasureMazeConfig) -> None:
        self.config = config
        self.seed_value = config.seed if config.seed is not None else random.randrange(1_000_000_000)
        self._observation_seed = self.seed_value ^ 0x5EED5EED

        template_rooms, start_room_id, grid_rows, grid_cols = _build_world(
            config=self.config,
            seed_value=self.seed_value,
        )

        self._template_rooms = template_rooms
        self.start_room_id = start_room_id
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols

        self.rooms: dict[int, Room] = {}
        self.current_room_id = self.start_room_id
        self.inventory: list[str] = []
        self.visited_room_ids: set[int] = set()
        self.visited_path: list[int] = []
        self.collected_keys: list[str] = []
        self.steps = 0
        self.success = False
        self.done = False
        self.last_action: str | None = None
        self.last_message: str | None = None
        self.last_observation = ''
        self._observation_rng = random.Random(self._observation_seed)

        self.reset()

    def reset(self) -> tuple[str, dict]:
        """Reset the environment and return the initial observation and info."""

        self.rooms = deepcopy(self._template_rooms)
        # Reset the text RNG so a fixed seed reproduces the same narration sequence.
        self._observation_rng = random.Random(self._observation_seed)
        self.current_room_id = self.start_room_id
        self.inventory = []
        self.visited_room_ids = {self.start_room_id}
        self.visited_path = [self.start_room_id]
        self.collected_keys = []
        self.steps = 0
        self.success = False
        self.done = False
        self.last_action = 'reset'
        self.last_message = 'You begin a new run in the maze and take a careful first look around.'
        self.last_observation = self._build_observation(self.last_message)
        return self.last_observation, self._build_info()

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        """Apply a tiny text command and return observation, reward, done, and info."""

        normalized = ' '.join(action.strip().lower().split())
        if self.done:
            message = 'The episode is already finished. Use reset() to start a new run.'
            observation = self._build_observation(message)
            self.last_message = message
            self.last_observation = observation
            return observation, 0.0, True, self._build_info()

        self.steps += 1
        self.last_action = normalized or '<empty>'
        reward = 0.0

        if not normalized:
            message = 'You entered an empty command, so nothing in the maze changed.'
        elif normalized == 'look':
            message = 'You pause and inspect the room again without moving anywhere.'
        elif normalized == 'help':
            message = (
                'Available commands: look, help, map, '
                'go north, go south, go east, go west, take <item>, and drop <item>.'
            )
        elif normalized == 'map':
            message = 'You review the explored ASCII map shown below.'
        elif normalized.startswith('go '):
            direction = normalized[3:]
            message = self._handle_move(direction)
        elif normalized.startswith('take '):
            item = normalized[5:].strip()
            message, reward = self._handle_take(item)
        elif normalized.startswith('drop '):
            item = normalized[5:].strip()
            message = self._handle_drop(item)
        else:
            message = f"'{action.strip()}' is not a recognized command. Use help to see the small command set."

        if not self.done and self.config.max_steps is not None and self.steps >= self.config.max_steps:
            self.done = True
            self.success = False
            message = f'{message} You have reached the step limit, so the run ends here.'

        self.last_message = message
        observation = self._build_observation(message)

        self.last_observation = observation
        return observation, reward, self.done, self._build_info()

    def render_ascii(self, reveal_all: bool = False) -> str:
        """Render the maze using a compact ASCII map."""

        room_by_coord = {(room.row, room.col): room for room in self.rooms.values()}
        cell_width = max(1, len(f'K{self.config.num_keys}'))

        def room_visible(room_id: int) -> bool:
            return reveal_all or room_id in self.visited_room_ids or room_id == self.current_room_id

        mode = 'full' if reveal_all else 'explored'
        legend = 'Legend: @ current, . visited, # hidden, S start, K<n> remaining key order, o room'
        lines = [f'Maze map ({mode})', legend]

        for row in range(self.grid_rows):
            room_line: list[str] = []
            for col in range(self.grid_cols):
                room = room_by_coord[(row, col)]
                is_visible = room_visible(room.room_id)
                marker = self._room_marker(room.room_id, reveal_all=reveal_all, visible=is_visible)
                room_line.append(marker.ljust(cell_width))

                if col == self.grid_cols - 1:
                    continue

                east_neighbor = room_by_coord[(row, col + 1)]
                show_east = 'east' in room.exits and (
                    reveal_all or (is_visible and room_visible(east_neighbor.room_id))
                )
                room_line.append('-' if show_east else ' ')

            lines.append(''.join(room_line).rstrip())

            if row == self.grid_rows - 1:
                continue

            connector_line: list[str] = []
            for col in range(self.grid_cols):
                room = room_by_coord[(row, col)]
                is_visible = room_visible(room.room_id)
                south_neighbor = room_by_coord[(row + 1, col)]
                show_south = 'south' in room.exits and (
                    reveal_all or (is_visible and room_visible(south_neighbor.room_id))
                )
                connector_line.append(('|' + ' ' * (cell_width - 1)) if show_south else (' ' * cell_width))

                if col == self.grid_cols - 1:
                    continue

                connector_line.append(' ')

            lines.append(''.join(connector_line).rstrip())

        return '\n'.join(lines).rstrip()

    def close(self) -> None:
        """Close the environment. No external resources are used."""

    def valid_actions(self) -> list[str]:
        """Return a simple list of likely valid commands in the current state."""

        room = self.rooms[self.current_room_id]
        actions = ['look', 'help', 'map']
        actions.extend(f'go {direction}' for direction in DIRECTION_ORDER if direction in room.exits)
        next_required_key = self._next_required_key()
        for item in sorted(room.items, key=_item_sort_key):
            if item.startswith('key') and self.config.ordered_keys and item != next_required_key:
                continue
            actions.append(f'take {item}')
        actions.extend(item for item in self._drop_actions())
        return actions

    def _drop_actions(self) -> list[str]:
        droppable = [item for item in self.inventory if not item.startswith('key')]
        return [f'drop {item}' for item in droppable]

    def _inventory_text(self) -> str:
        if not self.inventory:
            return 'nothing'
        return _format_list(self.inventory)

    def _visible_items_text(self) -> str:
        room = self.rooms[self.current_room_id]
        if not room.items:
            return 'nothing'
        return _format_list(sorted(room.items, key=_item_sort_key))

    def _next_required_key(self) -> str | None:
        if not self.config.ordered_keys:
            return None
        for key_index in range(1, self.config.num_keys + 1):
            key_name = f'key{key_index}'
            if key_name not in self.collected_keys:
                return key_name
        return None

    def _remaining_keys(self) -> list[str]:
        return [
            f'key{key_index}'
            for key_index in range(1, self.config.num_keys + 1)
            if f'key{key_index}' not in self.collected_keys
        ]

    def _visited_path_text(self) -> str:
        """Build a string showing the complete visit path with key annotations.

        Example: ``"6A -> 3D (has key1)"``
        """
        remaining_keys = set(self._remaining_keys())
        parts: list[str] = []
        for room_id in self.visited_path:
            room = self.rooms[room_id]
            room_keys = sorted(item for item in room.items if item in remaining_keys)
            if room_keys:
                annotation = ', '.join(room_keys)
                parts.append(f'{room.name} (has {annotation})')
            else:
                parts.append(f'{room.name}')
        return ' -> '.join(parts)

    def room_groups(self) -> dict[str, list[str]]:
        """Classify all rooms into three groups.

        Returns a dict with three keys:

        - ``visited_with_keys``: visited rooms that still contain at least one
          uncollected key.
        - ``visited_no_keys``: visited rooms with no uncollected keys.
        - ``unvisited``: rooms the player has not yet entered.

        Each entry is the room name (e.g. ``"1A"``).
        """

        remaining_keys = set(self._remaining_keys())
        visited_with_keys: list[str] = []
        visited_no_keys: list[str] = []
        unvisited: list[str] = []

        for room in self.rooms.values():
            if room.room_id not in self.visited_room_ids:
                unvisited.append(room.name)
                continue

            room_keys = [item for item in room.items if item in remaining_keys]
            if room_keys:
                visited_with_keys.append(room.name)
            else:
                visited_no_keys.append(room.name)

        return {
            'visited_with_keys': visited_with_keys,
            'visited_no_keys': visited_no_keys,
            'unvisited': unvisited,
        }

    def _build_info(self) -> dict:
        return {
            'steps': self.steps,
            'room_id': self.current_room_id,
            'inventory': list(self.inventory),
            'collected_keys': list(self.collected_keys),
            'keys_remaining': self._remaining_keys(),
            'ordered_keys': self.config.ordered_keys,
            'next_required_key': self._next_required_key(),
            'success': self.success,
            'room_groups': self.room_groups(),
            'visited_path': self._visited_path_text(),
        }

    def _room_marker(self, room_id: int, reveal_all: bool, visible: bool) -> str:
        if room_id == self.current_room_id:
            return '@'
        if not reveal_all and not visible:
            return '#'
        if reveal_all and room_id == self.start_room_id:
            return 'S'

        room = self.rooms[room_id]
        key_marker = self._remaining_key_marker(room)
        if reveal_all and key_marker is not None:
            return key_marker
        if room_id in self.visited_room_ids:
            return '.'
        return 'o' if reveal_all else '#'

    def _remaining_key_marker(self, room: Room) -> str | None:
        for item in room.items:
            if item.startswith('key') and item not in self.collected_keys:
                return f'K{item[3:]}'
        return None

    def _handle_move(self, direction: str) -> str:
        if direction not in DIRECTION_VECTORS:
            return f"'{direction}' is not a valid direction. Use north, south, east, or west."

        room = self.rooms[self.current_room_id]
        if direction not in room.exits:
            return f'You try to go {direction}, but there is no passage in that direction.'

        self.current_room_id = room.exits[direction]
        self.visited_room_ids.add(self.current_room_id)
        self.visited_path.append(self.current_room_id)
        destination = self.rooms[self.current_room_id]
        return f'You go {direction} and arrive in {destination.name}.'

    def _handle_take(self, item: str) -> tuple[str, float]:
        if not item:
            return "You must name an item after the word 'take'.", 0.0

        room = self.rooms[self.current_room_id]
        if item not in room.items:
            return f'You do not see {item} in this room.', 0.0

        if item.startswith('key'):
            next_required_key = self._next_required_key()
            if self.config.ordered_keys and item != next_required_key:
                return (
                    f'You can see {item} here, but ordered mode requires {next_required_key} first, '
                    f'so {item} stays where it is.'
                ), 0.0

            room.items.remove(item)
            self.inventory.append(item)
            self.collected_keys.append(item)
            room.items.sort(key=_item_sort_key)

            if len(self.collected_keys) == self.config.num_keys:
                self.done = True
                self.success = True
                return (
                    f'You take {item}. That was the final required key, so the run is complete.',
                    SUCCESS_REWARD,
                )

            if self.config.ordered_keys:
                return f'You take {item}. The collection order remains correct.', KEY_PICKUP_REWARD

            return f'You take {item}. Any key order is allowed in this run.', KEY_PICKUP_REWARD

        room.items.remove(item)
        self.inventory.append(item)
        room.items.sort(key=_item_sort_key)
        return f'You take {item} and add it to your inventory.', 0.0

    def _handle_drop(self, item: str) -> str:
        if not item:
            return "You must name an item after the word 'drop'."
        if item not in self.inventory:
            return f'You are not carrying {item}.'
        if item.startswith('key'):
            return 'Collected keys stay with you once taken, so keys cannot be dropped.'

        self.inventory.remove(item)
        room = self.rooms[self.current_room_id]
        room.items.append(item)
        room.items.sort(key=_item_sort_key)
        return f'You drop {item} on the floor of the current room.'

    def _build_observation(self, last_message: str | None) -> str:
        room = self.rooms[self.current_room_id]
        exits = [direction for direction in DIRECTION_ORDER if direction in room.exits]
        exit_text = _format_list(exits)
        exit_summary = ', '.join(exits) if exits else 'none'
        visible_text = _format_list(sorted(room.items, key=_item_sort_key))
        inventory_text = self._inventory_text()
        keys_remaining_text = _format_list(self._remaining_keys())
        next_required_key = self._next_required_key()
        recap_action = self.last_action or 'reset'
        recap_message = last_message or 'No recent action has been recorded.'
        available_actions_text = _format_list(self.valid_actions())
        context = ObservationTemplateContext(
            room_name=room.name,
            grid_row=room.row + 1,
            grid_col=room.col + 1,
            exit_text=exit_text,
            exit_summary=exit_summary,
            visible_text=visible_text,
            inventory_text=inventory_text,
            keys_remaining_text=keys_remaining_text,
            next_required_key_display=next_required_key or 'none',
            recap_action=recap_action,
            recap_message=recap_message,
            available_actions_text=available_actions_text,
            has_exits=bool(exits),
            ordered_keys=self.config.ordered_keys,
        )
        (
            paragraph_one,
            paragraph_two,
            paragraph_three,
            paragraph_four,
            paragraph_five,
            paragraph_six,
        ) = build_observation_paragraphs(context, self._observation_rng)

        paragraphs = [
            paragraph_one,
            paragraph_two,
            paragraph_three,
            paragraph_four,
            paragraph_five,
            paragraph_six,
        ]

        # groups = self.room_groups()
        # room_lines: list[str] = [
        #     'Room names encode grid position: the number is the row and the letter is the column (A=1, B=2, …, Z=26).',
        # ]
        # for label, key in (
        #     ('Visited rooms (with keys)', 'visited_with_keys'),
        #     ('Visited rooms (no keys)', 'visited_no_keys'),
        #     ('Unvisited rooms', 'unvisited'),
        # ):
        #     names = groups[key]
        #     room_lines.append(f'{label}: {", ".join(names) if names else "none"}.')
        # paragraphs.append('\n'.join(room_lines))
        # paragraphs.append(self.render_ascii())

        return '\n\n'.join(paragraphs)


def make_env(
    *,
    size: int,
    num_keys: int,
    ordered_keys: bool = True,
    seed: int | None = None,
    min_distractors_per_room: int = 2,
    max_steps: int | None = None,
) -> TreasureMazeEnv:
    """Build a :class:`TreasureMazeEnv` with a small keyword-based API."""

    config = TreasureMazeConfig(
        size=size,
        num_keys=num_keys,
        ordered_keys=ordered_keys,
        seed=seed,
        min_distractors_per_room=min_distractors_per_room,
        max_steps=max_steps,
    )
    return TreasureMazeEnv(config)


def make_ordered_env(
    *,
    size: int,
    num_keys: int,
    seed: int | None = None,
    min_distractors_per_room: int = 2,
    max_steps: int | None = None,
) -> TreasureMazeEnv:
    """Build an environment where keys must be collected in ascending order."""

    return make_env(
        size=size,
        num_keys=num_keys,
        ordered_keys=True,
        seed=seed,
        min_distractors_per_room=min_distractors_per_room,
        max_steps=max_steps,
    )


def make_unordered_env(
    *,
    size: int,
    num_keys: int,
    seed: int | None = None,
    min_distractors_per_room: int = 2,
    max_steps: int | None = None,
) -> TreasureMazeEnv:
    """Build an environment where keys may be collected in any order."""

    return make_env(
        size=size,
        num_keys=num_keys,
        ordered_keys=False,
        seed=seed,
        min_distractors_per_room=min_distractors_per_room,
        max_steps=max_steps,
    )


__all__ = [
    'DISTRACTOR_WORDS',
    'Room',
    'TreasureMazeConfig',
    'TreasureMazeEnv',
    'make_env',
    'make_ordered_env',
    'make_unordered_env',
]
