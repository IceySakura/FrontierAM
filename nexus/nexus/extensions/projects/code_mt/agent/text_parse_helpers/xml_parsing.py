"""XML parser for terminus2 agent response format.

Parses LLM responses in the XML-plain format:

<response>
<analysis>...</analysis>
<plan>...</plan>
<commands>
<keystrokes duration="0.1">ls -la
</keystrokes>
</commands>
<task_complete>true</task_complete>
</response>
"""

import re

from nexus.extensions.projects.code_mt.agent.text_parse_helpers.parse_types import ParseResult, ParsedCommand


class TerminusXMLParser:
    """Parser for terminus XML plain response format."""

    def __init__(self):
        self.required_sections = ['analysis', 'plan', 'commands']

    def parse_response(self, response: str) -> ParseResult:
        """Parse a terminus XML plain response and extract commands.

        Args:
            response: The full LLM response string

        Returns:
            ParseResult with commands, completion status, errors and warnings
        """
        result = self._try_parse_response(response)

        if result.error:
            for fix_name, fix_function in self._get_auto_fixes():
                corrected_response, was_fixed = fix_function(response, result.error)
                if was_fixed:
                    corrected_result = self._try_parse_response(corrected_response)
                    if corrected_result.error == '':
                        auto_warning = f'AUTO-CORRECTED: {fix_name} - please fix this in future responses'
                        corrected_result.warning = self._combine_warnings(auto_warning, corrected_result.warning)
                        return corrected_result

        return result

    def _try_parse_response(self, response: str) -> ParseResult:
        warnings: list[str] = []

        self._check_extra_text(response, warnings)

        response_content, missing_close = self._extract_response_content(response)
        if not response_content:
            return ParseResult([], False, 'No <response> tag found', self._fmt_warnings(warnings), '', '')
        if missing_close:
            return ParseResult([], False, 'Missing </response> closing tag', self._fmt_warnings(warnings), '', '')

        is_complete = self._check_task_complete(response_content)

        sections = self._extract_sections(response_content, warnings)

        analysis = sections.get('analysis', '')
        plan = sections.get('plan', '')

        commands_content = sections.get('commands', '')
        if not commands_content:
            if 'commands' in sections:
                if not is_complete:
                    warnings.append(
                        'Commands section is empty; not taking any action. '
                        'If you want to wait a specific amount of time please use '
                        "`sleep`, but if you're waiting for a command to finish then "
                        'continue to wait.'
                    )
                return ParseResult([], is_complete, '', self._fmt_warnings(warnings), analysis, plan)
            if is_complete:
                return ParseResult([], True, '', self._fmt_warnings(warnings), analysis, plan)
            return ParseResult([], False, 'Missing <commands> section', self._fmt_warnings(warnings), analysis, plan)

        commands, parse_error = self._parse_xml_commands(commands_content, warnings)
        if parse_error:
            if is_complete:
                warnings.append(parse_error)
                return ParseResult([], True, '', self._fmt_warnings(warnings), analysis, plan)
            return ParseResult([], False, parse_error, self._fmt_warnings(warnings), analysis, plan)

        return ParseResult(commands, is_complete, '', self._fmt_warnings(warnings), analysis, plan)

    # ---- Auto-fixes ----

    def _get_auto_fixes(self):
        return [
            ('Missing </response> tag was automatically inserted', self._fix_missing_response_tag),
        ]

    def _fix_missing_response_tag(self, response: str, error: str) -> tuple[str, bool]:
        if 'Missing </response> closing tag' not in error:
            return response, False
        corrected = response.rstrip() + '\n</response>'
        return corrected, True

    def _combine_warnings(self, auto_warning: str, existing_warning: str) -> str:
        if existing_warning:
            return f'- {auto_warning}\n{existing_warning}'
        return f'- {auto_warning}'

    @staticmethod
    def _fmt_warnings(warnings: list[str]) -> str:
        return '- ' + '\n- '.join(warnings) if warnings else ''

    # ---- Extraction ----

    def _check_extra_text(self, response: str, warnings: list[str]) -> None:
        start_pos = response.find('<response>')
        end_pos = response.find('</response>')

        if start_pos == -1:
            return

        before_text = response[:start_pos].strip()
        if before_text:
            warnings.append('Extra text detected before <response> tag')

        if end_pos != -1:
            after_text = response[end_pos + len('</response>') :].strip()
            if after_text:
                warnings.append('Extra text detected after </response> tag')
                total_response_count = response.count('<response>')
                if total_response_count > 1:
                    warnings.append(
                        f'IMPORTANT: Only issue one <response> block at a time. '
                        f'You issued {total_response_count} and only the first was executed.'
                    )

    def _extract_response_content(self, response: str) -> tuple[str, bool]:
        """Extract content from <response> tags.

        Returns:
            Tuple of (content, missing_close_tag).
            missing_close_tag is True when <response> exists but </response> is absent.
        """
        start_pos = response.find('<response>')
        if start_pos == -1:
            return '', False

        end_pos = response.find('</response>', start_pos)
        if end_pos == -1:
            content = response[start_pos + len('<response>') :].strip()
            return content, True

        return response[start_pos + len('<response>') : end_pos].strip(), False

    def _extract_sections(self, content: str, warnings: list[str]) -> dict:
        sections: dict[str, str] = {}
        found_sections: set[str] = set()

        section_patterns = {
            'analysis': (r'<analysis>(.*?)</analysis>', r'<analysis\s*/>', r'<analysis></analysis>'),
            'plan': (r'<plan>(.*?)</plan>', r'<plan\s*/>', r'<plan></plan>'),
            'commands': (r'<commands>(.*?)</commands>', r'<commands\s*/>', r'<commands></commands>'),
            'task_complete': (
                r'<task_complete>(.*?)</task_complete>',
                r'<task_complete\s*/>',
                r'<task_complete></task_complete>',
            ),
        }

        for section_name, patterns in section_patterns.items():
            full_pattern, self_closing_pattern, empty_pattern = patterns

            match = re.search(full_pattern, content, re.DOTALL)
            if match:
                sections[section_name] = match.group(1).strip()
                found_sections.add(section_name)
                continue

            if re.search(self_closing_pattern, content):
                sections[section_name] = ''
                found_sections.add(section_name)
                continue

            if re.search(empty_pattern, content):
                sections[section_name] = ''
                found_sections.add(section_name)
                continue

        required = set(self.required_sections)
        missing = required - found_sections
        for section in missing:
            if section != 'task_complete':
                warnings.append(f'Missing <{section}> section')

        top_level_tags = self._find_top_level_tags(content)
        expected_tags = set(self.required_sections + ['task_complete'])
        unexpected = set(top_level_tags) - expected_tags
        for tag in unexpected:
            warnings.append(f'Unknown tag found: <{tag}>, expected analysis/plan/commands/task_complete')

        for section_name in self.required_sections + ['task_complete']:
            tag_count = len(re.findall(f'<{section_name}(?:\\s|>|/>)', content))
            if tag_count > 1:
                if section_name == 'commands':
                    warnings.append(
                        f'IMPORTANT: Only issue one <commands> block at a time. '
                        f'You issued {tag_count} and only the first was executed.'
                    )
                else:
                    warnings.append(f'Multiple <{section_name}> sections found')

        self._check_section_order(content, warnings)

        return sections

    def _parse_xml_commands(self, xml_content: str, warnings: list[str]) -> tuple[list[ParsedCommand], str]:
        commands: list[ParsedCommand] = []
        keystrokes_pattern = re.compile(r'<keystrokes([^>]*)>(.*?)</keystrokes>', re.DOTALL)

        matches = keystrokes_pattern.findall(xml_content)
        for i, (attributes_str, keystrokes_content) in enumerate(matches):
            self._check_attribute_issues(attributes_str, i + 1, warnings)

            duration = 1.0
            duration_match = re.search(r'duration\s*=\s*["\']([^"\']*)["\']', attributes_str)
            if duration_match:
                try:
                    duration = float(duration_match.group(1))
                except ValueError:
                    warnings.append(
                        f"Command {i + 1}: Invalid duration value '{duration_match.group(1)}', using default 1.0"
                    )
            else:
                warnings.append(f'Command {i + 1}: Missing duration attribute, using default 1.0')

            if i < len(matches) - 1 and not keystrokes_content.endswith('\n'):
                warnings.append(
                    f'Command {i + 1} should end with newline when followed '
                    'by another command. Otherwise the two commands will be '
                    'concatenated together on the same line.'
                )

            commands.append(ParsedCommand(keystrokes=keystrokes_content, duration=duration))

        entities = {'&lt;': '<', '&gt;': '>', '&amp;': '&', '&quot;': '"', '&apos;': "'"}
        for entity, char in entities.items():
            if entity in xml_content:
                warnings.append(
                    f'Warning: {entity} is read verbatim and not converted to {char}. '
                    f'NEVER USE {entity}, unless you want these exact characters to '
                    f'appear directly in the output.'
                )

        if '\\r\\n' in xml_content:
            warnings.append('Warning: \\r\\n line endings are not necessary - use \\n instead for simpler output')

        return commands, ''

    def _find_top_level_tags(self, content: str) -> list[str]:
        top_level_tags: list[str] = []
        depth = 0
        i = 0

        while i < len(content):
            if content[i] == '<':
                tag_end = content.find('>', i)
                if tag_end == -1:
                    break

                tag_content = content[i + 1 : tag_end]

                if tag_content.startswith('!') or tag_content.startswith('?'):
                    i = tag_end + 1
                    continue

                if tag_content.startswith('/'):
                    depth -= 1
                    i = tag_end + 1
                    continue

                is_self_closing = tag_content.endswith('/')

                tag_name = tag_content.split()[0] if ' ' in tag_content else tag_content
                if tag_name.endswith('/'):
                    tag_name = tag_name[:-1]

                if depth == 0:
                    top_level_tags.append(tag_name)

                if not is_self_closing:
                    depth += 1

                i = tag_end + 1
            else:
                i += 1

        return top_level_tags

    def _check_section_order(self, content: str, warnings: list[str]) -> None:
        positions: dict[str, int] = {}
        for section in ['analysis', 'plan', 'commands']:
            match = re.search(f'<{section}(?:\\s|>|/>)', content)
            if match:
                positions[section] = match.start()

        if len(positions) < 2:
            return

        expected_order = ['analysis', 'plan', 'commands']
        present_sections = [(section, positions[section]) for section in expected_order if section in positions]
        actual_order = [section for section, _pos in sorted(present_sections, key=lambda x: x[1])]
        expected_present = [s for s in expected_order if s in positions]

        if actual_order != expected_present:
            actual_str = ' → '.join(actual_order)
            expected_str = ' → '.join(expected_present)
            warnings.append(f'Sections appear in wrong order. Found: {actual_str}, expected: {expected_str}')

    def _check_attribute_issues(self, attributes_str: str, command_num: int, warnings: list[str]) -> None:
        unquoted_pattern = re.compile(r'(\w+)\s*=\s*([^"\'\s>]+)')
        unquoted_matches = unquoted_pattern.findall(attributes_str)
        for attr_name, attr_value in unquoted_matches:
            warnings.append(
                f'Command {command_num}: Attribute \'{attr_name}\' value should be quoted: {attr_name}="{attr_value}"'
            )

        single_quote_pattern = re.compile(r"(\w+)\s*=\s*'([^']*)'")
        single_quote_matches = single_quote_pattern.findall(attributes_str)
        for attr_name, attr_value in single_quote_matches:
            warnings.append(
                f'Command {command_num}: Use double quotes for attribute \'{attr_name}\': {attr_name}="{attr_value}"'
            )

        known_attributes = {'duration'}
        all_attributes = re.findall(r'(\w+)\s*=', attributes_str)
        for attr_name in all_attributes:
            if attr_name not in known_attributes:
                warnings.append(
                    f"Command {command_num}: Unknown attribute '{attr_name}' - "
                    f'known attributes are: {", ".join(sorted(known_attributes))}'
                )

    def _check_task_complete(self, response_content: str) -> bool:
        true_match = re.search(r'<task_complete>\s*true\s*</task_complete>', response_content, re.IGNORECASE)
        return bool(true_match)

    def salvage_truncated_response(self, truncated_response: str) -> tuple[str | None, bool]:
        """Try to salvage a valid response from truncated output.

        When OutputLengthExceededError occurs, the response may contain a
        complete <response>...</response> block followed by extra content
        that caused the truncation. This method tries to extract the first
        valid complete response.

        Args:
            truncated_response: The truncated response from the LLM

        Returns:
            Tuple of (salvaged_response, has_multiple_blocks)
            - salvaged_response: Clean response up to </response> if salvageable, None otherwise
            - has_multiple_blocks: True if multiple response/commands blocks were detected
        """
        commands_end = truncated_response.find('</commands>')
        if commands_end == -1:
            return None, False

        response_end = truncated_response.find('</response>', commands_end)
        if response_end == -1:
            return None, False

        clean_response = truncated_response[: response_end + len('</response>')]

        try:
            parse_result = self.parse_response(clean_response)

            has_multiple_blocks = False
            if parse_result.warning:
                warning_lower = parse_result.warning.lower()
                has_multiple_blocks = 'only issue one' in warning_lower and 'block at a time' in warning_lower

            if not parse_result.error and not has_multiple_blocks:
                return clean_response, False
            return None, has_multiple_blocks
        except Exception:
            return None, False
