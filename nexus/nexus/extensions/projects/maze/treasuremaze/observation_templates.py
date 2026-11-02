"""Observation paragraph templates for TreasureMaze."""

from __future__ import annotations

import random

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ObservationTemplateContext:
    room_name: str
    grid_row: int
    grid_col: int
    exit_text: str
    exit_summary: str
    visible_text: str
    inventory_text: str
    keys_remaining_text: str
    next_required_key_display: str
    recap_action: str
    recap_message: str
    available_actions_text: str
    has_exits: bool
    ordered_keys: bool


ParagraphBuilder = Callable[[ObservationTemplateContext], str]


def paragraph_one_builder(ctx):
    return (
        f'The maze closes around you in {ctx.room_name}, room names encode grid position: the number is the row and the letter is the column (A=1, B=2, …, Z=26. '
        f'This is a square little chamber with stale air,'
        f'scuffed stone, and just enough echo to make every step sound important. A plaque fixed '
        f'somewhere in view insists on the name {ctx.room_name}, and that detail matters: you are '
        f'standing at grid row {ctx.grid_row} and grid column {ctx.grid_col}. The room may be '
        f'simple, but it does not feel empty; it feels like the kind of place where the next '
        'useful thing is waiting for whoever keeps moving.'
    )


def paragraph_two_with_exists_builder(ctx):
    return (
        f'Passages break the walls to the {ctx.exit_text}, each one a clean invitation to press '
        f'deeper into the maze. No hidden staircase, no secret ladder, no surprise trapdoor is '
        f'helping you right now; these are the only directions that will answer if you push '
        f'forward. If you want the stripped-down route list, the open ways are simply these: '
        f'{ctx.exit_summary}. Exits: {ctx.exit_summary}.'
    )


def paragraph_two_no_exit_builder(_ctx):
    return (
        'Every wall looks stubbornly sealed for the moment, leaving you boxed inside a room that '
        'offers no visible way onward. If the maze means to open later, it is not showing its '
        'hand yet. Exits: none.'
    )


def paragraph_three_builder(ctx):
    return (
        f'A slow sweep of the room turns up {ctx.visible_text}, scattered across the floor or '
        f'marooned in corners like leftovers from older, less successful searches. Some of it is '
        f'junk, some of it may matter, and the maze will not bother to tell you which until you '
        f'start picking things up. Visible items: {ctx.visible_text}.'
    )


def paragraph_four_ordered_builder(ctx):
    return (
        'The job is straightforward even if the route is not: collect every key, and in this '
        f'ordered run you must do it in strict ascending sequence like the maze is daring you to '
        f'lose patience. Goal: collect all keys in order. Keys remaining: {ctx.keys_remaining_text}. '
        f'Next required key: {ctx.next_required_key_display}. Your pockets, hands, and improvised '
        f'carrying capacity currently hold {ctx.inventory_text}. Inventory: {ctx.inventory_text}.'
    )


def paragraph_four_unordered_builder(ctx):
    return (
        'The assignment is to gather every key, but this looser version lets you snatch any one '
        f'you can reach instead of obeying a ceremony. Goal: collect all keys in any order. Keys '
        f'remaining: {ctx.keys_remaining_text}. Your pockets, hands, and improvised carrying '
        f'capacity currently hold {ctx.inventory_text}. Inventory: {ctx.inventory_text}.'
    )


def paragraph_five_builder(ctx):
    return (
        f'The last beat of the story is still ringing in the room. Previous action: {ctx.recap_action}. '
        f'Result: {ctx.recap_message} Beneath the dramatic phrasing, the repeated anchors are the '
        f'parts worth trusting: room name, exits, visible items, goal, and inventory. Read those '
        'closely, then choose the next move.'
    )


def paragraph_six_builder(ctx):
    return f'If you want the clean tactical summary, your current options are these: {ctx.available_actions_text}.'


def build_observation_paragraphs(
    context: ObservationTemplateContext,
    rng: random.Random,
) -> tuple[str, str, str, str, str, str]:
    del rng

    paragraph_two_builder = paragraph_two_with_exists_builder if context.has_exits else paragraph_two_no_exit_builder
    paragraph_four_builder = (
        paragraph_four_ordered_builder if context.ordered_keys else paragraph_four_unordered_builder
    )

    paragraph_one = paragraph_one_builder(context)
    paragraph_two = paragraph_two_builder(context)
    paragraph_three = paragraph_three_builder(context)
    paragraph_four = paragraph_four_builder(context)
    paragraph_five = paragraph_five_builder(context)
    paragraph_six = paragraph_six_builder(context)
    return (
        paragraph_one,
        paragraph_two,
        paragraph_three,
        paragraph_four,
        paragraph_five,
        paragraph_six,
    )
