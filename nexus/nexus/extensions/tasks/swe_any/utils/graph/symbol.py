"""Parse SCIP symbol strings into structured descriptors.

A global SCIP symbol looks like::

    scip-python python zoo-demo 0.1.0 `zoo.animals`/Dog#speak().

which decomposes into a *scheme*, a *package* (manager, name, version) and a
sequence of *descriptors*. Each descriptor is ``name`` + a suffix that encodes
kind (SCIP symbol grammar §9.1):

    ``name/``   namespace
    ``name#``   type (class)
    ``name.``   term / field
    ``name().`` method / function   (optional disambiguator: ``name(+1).``)
    ``(name)``  parameter
    ``[name]``  type parameter
    ``name:``   meta
    ``name!``   macro

scip-python (v0.6.x) leaves ``SymbolInformation.kind`` unset, so the descriptor
suffix is the *only* reliable way to tell a class from a method from a field.
This module implements just enough of the grammar to recover that.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Descriptor suffix kinds we care about.
NAMESPACE = 'namespace'
TYPE = 'type'
TERM = 'term'
METHOD = 'method'
PARAMETER = 'parameter'
TYPE_PARAMETER = 'type_parameter'
META = 'meta'
MACRO = 'macro'

_SIMPLE_NAME_CHARS = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_$+-')


@dataclass(frozen=True)
class Descriptor:
    name: str
    suffix: str


@dataclass
class ParsedSymbol:
    """A structured view of a SCIP symbol string."""

    raw: str
    is_local: bool = False
    scheme: str = ''
    manager: str = ''
    package_name: str = ''
    package_version: str = ''
    descriptors: list[Descriptor] = field(default_factory=list)

    @property
    def last(self) -> Descriptor | None:
        return self.descriptors[-1] if self.descriptors else None

    @property
    def namespace(self) -> str:
        """Dotted module path built from leading namespace descriptors."""
        return '.'.join(d.name for d in self.descriptors if d.suffix == NAMESPACE)

    @property
    def display_name(self) -> str:
        return self.last.name if self.last else self.raw

    @property
    def qualified_name(self) -> str:
        """Human-readable owner path, e.g. ``Cat.Toy.__init__``."""
        parts = [d.name for d in self.descriptors if d.suffix in (TYPE, METHOD, TERM)]
        return '.'.join(parts)

    def owner_symbol(self) -> str | None:
        """Reconstruct the SCIP symbol string of the immediate owner.

        The owner is this symbol with its final descriptor dropped. Returns
        ``None`` when the owner would be a bare namespace (i.e. top-level).
        """
        if self.is_local or len(self.descriptors) < 2:
            return None
        owner_descriptors = self.descriptors[:-1]
        # A pure-namespace owner is a module, not an entity; skip it.
        if all(d.suffix == NAMESPACE for d in owner_descriptors):
            return None
        prefix = f'{self.scheme} {self.manager} {self.package_name} {self.package_version} '
        return prefix + _render_descriptors(owner_descriptors)


def _escape_name(name: str) -> str:
    if name and all(c in _SIMPLE_NAME_CHARS for c in name):
        return name
    return '`' + name.replace('`', '``') + '`'


def _render_descriptors(descriptors: list[Descriptor]) -> str:
    out: list[str] = []
    for d in descriptors:
        n = _escape_name(d.name)
        if d.suffix == NAMESPACE:
            out.append(n + '/')
        elif d.suffix == TYPE:
            out.append(n + '#')
        elif d.suffix == TERM:
            out.append(n + '.')
        elif d.suffix == METHOD:
            out.append(n + '().')
        elif d.suffix == META:
            out.append(n + ':')
        elif d.suffix == MACRO:
            out.append(n + '!')
        elif d.suffix == PARAMETER:
            out.append('(' + n + ')')
        elif d.suffix == TYPE_PARAMETER:
            out.append('[' + n + ']')
    return ''.join(out)


def _read_name(s: str, i: int) -> tuple[str, int]:
    """Read a (possibly backtick-escaped) descriptor name starting at ``i``."""
    if i < len(s) and s[i] == '`':
        j = i + 1
        buf: list[str] = []
        while j < len(s):
            if s[j] == '`':
                if j + 1 < len(s) and s[j + 1] == '`':  # escaped backtick
                    buf.append('`')
                    j += 2
                    continue
                j += 1
                break
            buf.append(s[j])
            j += 1
        return ''.join(buf), j
    j = i
    while j < len(s) and s[j] in _SIMPLE_NAME_CHARS:
        j += 1
    return s[i:j], j


def _parse_descriptors(s: str) -> list[Descriptor]:
    out: list[Descriptor] = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c == '(':  # parameter: (name)
            name, j = _read_name(s, i + 1)
            if j < n and s[j] == ')':
                j += 1
            out.append(Descriptor(name, PARAMETER))
            i = j
            continue
        if c == '[':  # type parameter: [name]
            name, j = _read_name(s, i + 1)
            if j < n and s[j] == ']':
                j += 1
            out.append(Descriptor(name, TYPE_PARAMETER))
            i = j
            continue

        name, j = _read_name(s, i)
        if j >= n:
            break
        suffix_char = s[j]
        if suffix_char == '(':  # method: name(<disambiguator>).
            k = j + 1
            while k < n and s[k] != ')':
                k += 1
            k += 1  # skip ')'
            if k < n and s[k] == '.':
                k += 1
            out.append(Descriptor(name, METHOD))
            i = k
        elif suffix_char == '#':
            out.append(Descriptor(name, TYPE))
            i = j + 1
        elif suffix_char == '/':
            out.append(Descriptor(name, NAMESPACE))
            i = j + 1
        elif suffix_char == '.':
            out.append(Descriptor(name, TERM))
            i = j + 1
        elif suffix_char == ':':
            out.append(Descriptor(name, META))
            i = j + 1
        elif suffix_char == '!':
            out.append(Descriptor(name, MACRO))
            i = j + 1
        else:  # unknown, stop to avoid infinite loop
            break
    return out


def parse_symbol(symbol: str) -> ParsedSymbol:
    """Parse a SCIP symbol string. Never raises; unknowns become empty parses."""
    if symbol.startswith('local '):
        return ParsedSymbol(raw=symbol, is_local=True)
    # scheme + package(manager, name, version) are the first four space-separated
    # tokens in scip-python output; the remainder is the descriptor string.
    parts = symbol.split(' ', 4)
    if len(parts) < 5:
        return ParsedSymbol(raw=symbol)
    scheme, manager, pkg_name, pkg_version, desc = parts
    return ParsedSymbol(
        raw=symbol,
        scheme=scheme,
        manager=manager,
        package_name=pkg_name,
        package_version=pkg_version,
        descriptors=_parse_descriptors(desc),
    )
