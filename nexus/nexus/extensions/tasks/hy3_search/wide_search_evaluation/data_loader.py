# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
Core data classes for WideSearch evaluation within Nexus.

Only the dataclasses required for evaluation are kept here.
Dataset loading (from files / HuggingFace) is handled upstream by the task layer.
"""

import logging
import re

from dataclasses import dataclass
from io import StringIO

import pandas as pd

from nexus.extensions.tasks.hy3_search.utils import norm_column


logger = logging.getLogger(__name__)


@dataclass
class WideSearchQuery:
    instance_id: str
    query: str
    evaluation: dict
    answer: pd.DataFrame
    language: str


@dataclass
class WideSearchResponse:
    instance_id: str
    response: str
    messages: list[dict] | None = None
    trial_idx: int | None = None

    def extract_dataframe(self) -> pd.DataFrame | None:
        """Extract a DataFrame from the agent's text response (markdown table or pipe table)."""
        response_df = None
        markdown_str = re.findall(r'```markdown(.*?)```', self.response, re.DOTALL)
        if not markdown_str:
            pipe_positions = [m.start() for m in re.finditer(r'\|', self.response)]
            if len(pipe_positions) >= 4:
                first_pipe = pipe_positions[0]
                last_pipe = pipe_positions[-1]
                start = self.response.rfind('\n', 0, first_pipe)
                start = 0 if start == -1 else start
                end = self.response.find('\n', last_pipe)
                end = len(self.response) if end == -1 else end
                table_candidate = self.response[start:end]
                markdown_str = re.findall(r'((?:\|.*\n?)+)', table_candidate)
        if markdown_str:
            logger.debug(f'find markdown_str {markdown_str[0][:64]} ...')
            table_text = markdown_str[0].strip()
            lines = table_text.split('\n')
            lines[0] = lines[0].replace(' ', '').lower()
            lines = [line.strip() for line in lines]
            new_lines = []
            for line in lines:
                if set(line.strip()).issubset(set('|- :')) or '|' not in line:
                    continue
                new_lines.append('|'.join([seg.strip() for seg in line.split('|')]))
            table_text = '\n'.join(new_lines)
            response_df = pd.read_csv(StringIO(table_text), sep='|')
            response_df = response_df.loc[:, ~response_df.columns.str.startswith('Unnamed')]
            response_df.columns = [norm_column(c) for c in response_df.columns]
        else:
            logger.warning(f'No markdown table found in response (first 200 chars): {self.response[:200]!r}')
        return response_df
