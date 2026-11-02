"""Normalized data types for deterministic evaluation.

This module provides type-aware normalization classes that convert raw values
into standardized forms for comparison. Each NormalizedType handles:
- Value normalization (e.g., "$100.00" -> 100.0, "15min" -> 900 seconds)
- Alternative values (multiple correct answers)
- Type-specific equality comparison

Supported data types (matching WebArena-Verified):
- string: Case-insensitive, whitespace-normalized string comparison
- number/integer/float: Numeric comparison with tolerance
- currency: Currency value with symbol stripping
- boolean: Boolean parsing (yes/no, true/false, 1/0)
- date: Date parsing with multiple format support
- url: URL normalization with trailing slash handling
- duration: Time duration parsing (HH:MM:SS, Xmin, Xhr Ymin, etc.)
- distance: Distance measurement with unit conversion
- location-name: Location name with abbreviation expansion
- full_address: Street address normalization
- coordinates: Lat/lon coordinates with tolerance

Reference: WebArena-Verified evaluation system
https://github.com/ServiceNow/webarena-verified
"""

from __future__ import annotations

import logging
import re

from abc import ABC, abstractmethod
from decimal import Decimal, InvalidOperation
from typing import Any


logger = logging.getLogger(__name__)


class NormalizedType(ABC):
    """Base class for all normalized types.

    Each NormalizedType handles:
    - Normalization of raw values to a canonical form
    - Support for alternative values (list of valid answers)
    - Type-specific equality comparison
    """

    def __init__(self, value: Any, **kwargs: Any):
        """Initialize with a value or list of alternatives.

        Args:
            value: Single value or list of alternative values
        """
        if isinstance(value, list):
            # Multiple alternatives
            self.alternatives = [self._normalize(v, **kwargs) for v in value]
            self.normalized = self.alternatives[0] if self.alternatives else None
        else:
            # Single value
            self.normalized = self._normalize(value, **kwargs)
            self.alternatives = [self.normalized]

    @abstractmethod
    def _normalize(self, value: Any, **kwargs: Any) -> Any:
        """Normalize a single value to canonical form."""
        pass

    def __eq__(self, other: Any) -> bool:
        """Check equality, supporting alternatives on both sides."""
        if isinstance(other, NormalizedType):
            # Check if any alternative from self matches any from other
            for self_alt in self.alternatives:
                for other_alt in other.alternatives:
                    if self._values_equal(self_alt, other_alt):
                        return True
            return False
        # Compare against raw value
        try:
            # Try to normalize the other value using the same type
            other_normalized = self._normalize(other)
            for alt in self.alternatives:
                if self._values_equal(alt, other_normalized):
                    return True
        except Exception:
            pass
        # Fall back to direct comparison
        return any(self._values_equal(alt, other) for alt in self.alternatives)

    def _values_equal(self, a: Any, b: Any) -> bool:
        """Compare two normalized values for equality."""
        return a == b

    def __repr__(self) -> str:
        if len(self.alternatives) > 1:
            return f'{self.__class__.__name__}({self.alternatives})'
        return f'{self.__class__.__name__}({self.normalized!r})'

    def __str__(self) -> str:
        return str(self.normalized)


class NormalizedString(NormalizedType):
    """Normalized string with case-insensitive comparison and whitespace normalization."""

    def _normalize(self, value: Any, **kwargs: Any) -> str:
        if value is None:
            return ''
        s = str(value).strip()
        # Normalize whitespace
        s = re.sub(r'\s+', ' ', s)
        # Lowercase for comparison
        return s.lower()

    def _values_equal(self, a: Any, b: Any) -> bool:
        return str(a).lower().strip() == str(b).lower().strip()


class NormalizedNumber(NormalizedType):
    """Normalized number with tolerance for floating point comparison."""

    TOLERANCE = 1e-6

    def _normalize(self, value: Any, **kwargs: Any) -> float | None:
        if value is None:
            return None

        if isinstance(value, (int, float)):
            return float(value)

        # Try to parse string as number
        s = str(value).strip()

        # Remove common formatting
        s = s.replace(',', '')  # Remove thousand separators
        s = s.replace(' ', '')  # Remove spaces

        # Handle percentage
        if s.endswith('%'):
            try:
                return float(s[:-1]) / 100.0
            except ValueError:
                pass

        # Try direct conversion
        try:
            return float(s)
        except ValueError:
            pass

        # Try extracting number from string
        match = re.search(r'[-+]?\d*\.?\d+', s)
        if match:
            try:
                return float(match.group())
            except ValueError:
                pass

        logger.warning(f"Could not normalize '{value}' as number")
        return None

    def _values_equal(self, a: Any, b: Any) -> bool:
        if a is None or b is None:
            return a is None and b is None
        try:
            return abs(float(a) - float(b)) < self.TOLERANCE
        except (ValueError, TypeError):
            return False


class NormalizedCurrency(NormalizedType):
    """Normalized currency with symbol stripping and decimal precision."""

    # Common currency symbols
    CURRENCY_SYMBOLS = ['$', '€', '£', '¥', '₹', 'USD', 'EUR', 'GBP', 'JPY', 'CNY']

    def _normalize(self, value: Any, **kwargs: Any) -> Decimal | None:
        if value is None:
            return None

        if isinstance(value, (int, float, Decimal)):
            return Decimal(str(value)).quantize(Decimal('0.01'))

        s = str(value).strip()

        # Remove currency symbols
        for symbol in self.CURRENCY_SYMBOLS:
            s = s.replace(symbol, '')

        # Remove thousand separators and spaces
        s = s.replace(',', '').replace(' ', '').strip()

        try:
            return Decimal(s).quantize(Decimal('0.01'))
        except InvalidOperation:
            # Try extracting number
            match = re.search(r'[-+]?\d+\.?\d*', s)
            if match:
                try:
                    return Decimal(match.group()).quantize(Decimal('0.01'))
                except InvalidOperation:
                    pass

        logger.warning(f"Could not normalize '{value}' as currency")
        return None

    def _values_equal(self, a: Any, b: Any) -> bool:
        if a is None or b is None:
            return a is None and b is None
        try:
            da = Decimal(str(a)).quantize(Decimal('0.01'))
            db = Decimal(str(b)).quantize(Decimal('0.01'))
            return da == db
        except (InvalidOperation, ValueError, TypeError):
            return False


class NormalizedBoolean(NormalizedType):
    """Normalized boolean with string parsing."""

    TRUE_VALUES = {'true', 'yes', '1', 'on', 'y', 't'}
    FALSE_VALUES = {'false', 'no', '0', 'off', 'n', 'f'}

    def _normalize(self, value: Any, **kwargs: Any) -> bool | None:
        if value is None:
            return None

        if isinstance(value, bool):
            return value

        s = str(value).lower().strip()

        if s in self.TRUE_VALUES:
            return True
        if s in self.FALSE_VALUES:
            return False

        logger.warning(f"Could not normalize '{value}' as boolean")
        return None


class NormalizedDate(NormalizedType):
    """Normalized date with multiple format support."""

    def _normalize(self, value: Any, **kwargs: Any) -> str | None:
        if value is None:
            return None

        from datetime import datetime

        s = str(value).strip()

        # Try common date formats
        formats = [
            '%Y-%m-%d',
            '%m/%d/%Y',
            '%d/%m/%Y',
            '%Y/%m/%d',
            '%B %d, %Y',
            '%b %d, %Y',
            '%d %B %Y',
            '%d %b %Y',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%d %H:%M:%S',
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(s, fmt)
                # Return ISO format for comparison
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                continue

        # If already in ISO-like format, clean it
        match = re.search(r'(\d{4})-(\d{1,2})-(\d{1,2})', s)
        if match:
            y, m, d = match.groups()
            return f'{y}-{int(m):02d}-{int(d):02d}'

        logger.warning(f"Could not normalize '{value}' as date")
        return s.lower()  # Fall back to lowercased string


class NormalizedUrl(NormalizedType):
    """Normalized URL with trailing slash handling."""

    def _normalize(self, value: Any, **kwargs: Any) -> str | None:
        if value is None:
            return None

        from urllib.parse import urlparse, urlunparse

        s = str(value).strip()

        try:
            parsed = urlparse(s)
            # Normalize: lowercase host, remove trailing slash from path
            return urlunparse((
                parsed.scheme.lower(),
                parsed.netloc.lower(),
                parsed.path.rstrip('/') or '/',
                parsed.params,
                parsed.query,
                '',  # Remove fragment
            ))
        except Exception:
            return s.lower()


class NormalizedDuration(NormalizedType):
    """Normalized duration with multiple format support.

    Converts various time duration formats to seconds for comparison:
    - HH:MM:SS format: "00:15:00" -> 900
    - HH:MM format: "01:30" -> 5400
    - Xmin format: "15min" -> 900
    - Xhr Ymin format: "1hr 35min" -> 5700
    - X hours Y minutes format: "1 hour 30 minutes" -> 5400
    - X hours format: "2 hours" -> 7200
    - X minutes format: "30 minutes" -> 1800
    """

    # Tolerance in seconds for duration comparison (2 minutes)
    TOLERANCE_SECONDS = 120

    def _normalize(self, value: Any, **kwargs: Any) -> float | None:
        if value is None:
            return None

        s = str(value).strip().lower()

        # Try HH:MM:SS format
        match = re.match(r'^(\d+):(\d+):(\d+)$', s)
        if match:
            h, m, sec = int(match.group(1)), int(match.group(2)), int(match.group(3))
            return float(h * 3600 + m * 60 + sec)

        # Try HH:MM format
        match = re.match(r'^(\d+):(\d+)$', s)
        if match:
            h, m = int(match.group(1)), int(match.group(2))
            return float(h * 3600 + m * 60)

        # Try Xhr Ymin format (e.g., "1hr 35min")
        match = re.match(r'^(\d+)\s*h(?:r|our)?s?\s*(\d+)\s*min(?:ute)?s?$', s)
        if match:
            h, m = int(match.group(1)), int(match.group(2))
            return float(h * 3600 + m * 60)

        # Try X hours Y minutes format
        match = re.match(r'^(\d+)\s*hours?\s*(?:and\s*)?(\d+)\s*(?:min(?:ute)?s?)$', s)
        if match:
            h, m = int(match.group(1)), int(match.group(2))
            return float(h * 3600 + m * 60)

        # Try Xmin format (e.g., "15min", "15 min", "15 minutes")
        match = re.match(r'^(\d+)\s*(?:min(?:ute)?s?)$', s)
        if match:
            return float(int(match.group(1)) * 60)

        # Try X hours format
        match = re.match(r'^(\d+)\s*h(?:r|our)?s?$', s)
        if match:
            return float(int(match.group(1)) * 3600)

        # Try X seconds format
        match = re.match(r'^(\d+)\s*(?:sec(?:ond)?s?)$', s)
        if match:
            return float(int(match.group(1)))

        # Try plain number (assume minutes)
        match = re.match(r'^(\d+(?:\.\d+)?)$', s)
        if match:
            return float(match.group(1)) * 60

        logger.warning(f"Could not normalize '{value}' as duration")
        return None

    def _values_equal(self, a: Any, b: Any) -> bool:
        if a is None or b is None:
            return a is None and b is None
        try:
            diff = abs(float(a) - float(b))
            return diff <= self.TOLERANCE_SECONDS
        except (ValueError, TypeError):
            return False


class NormalizedDistance(NormalizedType):
    """Normalized distance with unit conversion.

    Converts various distance formats to meters for comparison:
    - "1.4km" -> 1400
    - "500m" -> 500
    - "1.5 miles" -> 2414.02
    - "3 feet" -> 0.9144
    """

    # Tolerance in meters (50 meters)
    TOLERANCE_METERS = 50

    # Conversion factors to meters
    UNIT_CONVERSIONS = {
        'km': 1000,
        'kilometer': 1000,
        'kilometers': 1000,
        'm': 1,
        'meter': 1,
        'meters': 1,
        'mi': 1609.34,
        'mile': 1609.34,
        'miles': 1609.34,
        'ft': 0.3048,
        'feet': 0.3048,
        'foot': 0.3048,
        'yd': 0.9144,
        'yard': 0.9144,
        'yards': 0.9144,
    }

    def _normalize(self, value: Any, **kwargs: Any) -> float | None:
        if value is None:
            return None

        s = str(value).strip().lower()

        # Try to extract number and unit
        match = re.match(r'^([\d.]+)\s*([a-z]+)?$', s)
        if match:
            number = float(match.group(1))
            unit = match.group(2) or 'm'  # Default to meters

            conversion = self.UNIT_CONVERSIONS.get(unit, 1)
            return number * conversion

        # Try plain number (assume meters)
        try:
            return float(s)
        except ValueError:
            pass

        logger.warning(f"Could not normalize '{value}' as distance")
        return None

    def _values_equal(self, a: Any, b: Any) -> bool:
        if a is None or b is None:
            return a is None and b is None
        try:
            diff = abs(float(a) - float(b))
            return diff <= self.TOLERANCE_METERS
        except (ValueError, TypeError):
            return False


class NormalizedLocationName(NormalizedType):
    """Normalized location/place name with common variations handled.

    Handles:
    - Case insensitivity
    - Common abbreviations (St./Street, Ave./Avenue, etc.)
    - Extra whitespace normalization
    - Common punctuation removal
    """

    # Common abbreviation expansions
    ABBREVIATIONS = {
        'st': 'street',
        'st.': 'street',
        'ave': 'avenue',
        'ave.': 'avenue',
        'blvd': 'boulevard',
        'blvd.': 'boulevard',
        'dr': 'drive',
        'dr.': 'drive',
        'rd': 'road',
        'rd.': 'road',
        'ln': 'lane',
        'ln.': 'lane',
        'ct': 'court',
        'ct.': 'court',
        'pl': 'place',
        'pl.': 'place',
        'sq': 'square',
        'sq.': 'square',
        'mt': 'mount',
        'mt.': 'mount',
        'ctr': 'center',
        'intl': 'international',
        'univ': 'university',
        'n': 'north',
        'n.': 'north',
        's': 'south',
        's.': 'south',
        'e': 'east',
        'e.': 'east',
        'w': 'west',
        'w.': 'west',
    }

    def _normalize(self, value: Any, **kwargs: Any) -> str | None:
        if value is None:
            return None

        s = str(value).strip().lower()

        # Normalize whitespace
        s = re.sub(r'\s+', ' ', s)

        # Remove common punctuation but keep spaces
        s = re.sub(r'[,\'"()]', '', s)

        # Expand abbreviations
        words = s.split()
        expanded = []
        for word in words:
            expanded.append(self.ABBREVIATIONS.get(word, word))

        return ' '.join(expanded)

    def _values_equal(self, a: Any, b: Any) -> bool:
        if a is None or b is None:
            return a is None and b is None

        # Direct comparison after normalization
        if str(a) == str(b):
            return True

        # Check if one contains the other (for partial matches)
        a_str, b_str = str(a), str(b)
        if a_str in b_str or b_str in a_str:
            return True

        # Token-based comparison (all tokens in shorter must be in longer)
        a_tokens = set(a_str.split())
        b_tokens = set(b_str.split())

        # Check significant overlap
        if len(a_tokens) > 0 and len(b_tokens) > 0:
            smaller = a_tokens if len(a_tokens) <= len(b_tokens) else b_tokens
            larger = b_tokens if len(a_tokens) <= len(b_tokens) else a_tokens

            # If all tokens from smaller set are in larger set
            if smaller.issubset(larger):
                return True

            # Check if most tokens match (80% threshold)
            overlap = len(a_tokens & b_tokens)
            min_size = min(len(a_tokens), len(b_tokens))
            if min_size > 0 and overlap / min_size >= 0.8:
                return True

        return False


class NormalizedFullAddress(NormalizedType):
    """Normalized street address with common variations handled.

    Similar to LocationName but with additional address-specific handling:
    - State name abbreviations (PA -> Pennsylvania)
    - Number formatting
    - Suite/Apt normalization
    """

    # US State abbreviations
    STATE_ABBREVIATIONS = {
        'al': 'alabama',
        'ak': 'alaska',
        'az': 'arizona',
        'ar': 'arkansas',
        'ca': 'california',
        'co': 'colorado',
        'ct': 'connecticut',
        'de': 'delaware',
        'fl': 'florida',
        'ga': 'georgia',
        'hi': 'hawaii',
        'id': 'idaho',
        'il': 'illinois',
        'in': 'indiana',
        'ia': 'iowa',
        'ks': 'kansas',
        'ky': 'kentucky',
        'la': 'louisiana',
        'me': 'maine',
        'md': 'maryland',
        'ma': 'massachusetts',
        'mi': 'michigan',
        'mn': 'minnesota',
        'ms': 'mississippi',
        'mo': 'missouri',
        'mt': 'montana',
        'ne': 'nebraska',
        'nv': 'nevada',
        'nh': 'new hampshire',
        'nj': 'new jersey',
        'nm': 'new mexico',
        'ny': 'new york',
        'nc': 'north carolina',
        'nd': 'north dakota',
        'oh': 'ohio',
        'ok': 'oklahoma',
        'or': 'oregon',
        'pa': 'pennsylvania',
        'ri': 'rhode island',
        'sc': 'south carolina',
        'sd': 'south dakota',
        'tn': 'tennessee',
        'tx': 'texas',
        'ut': 'utah',
        'vt': 'vermont',
        'va': 'virginia',
        'wa': 'washington',
        'wv': 'west virginia',
        'wi': 'wisconsin',
        'wy': 'wyoming',
        'dc': 'district of columbia',
    }

    # Street abbreviations (inherit from LocationName)
    STREET_ABBREVIATIONS = NormalizedLocationName.ABBREVIATIONS.copy()

    def _normalize(self, value: Any, **kwargs: Any) -> str | None:
        if value is None:
            return None

        s = str(value).strip().lower()

        # Normalize whitespace
        s = re.sub(r'\s+', ' ', s)

        # Remove common punctuation
        s = re.sub(r'[,\'"#()]', '', s)

        # Normalize suite/apt
        s = re.sub(r'\b(suite|ste|apt|apartment|unit)\b\.?\s*', 'unit ', s)

        # Expand abbreviations
        words = s.split()
        expanded = []
        for word in words:
            # Check state abbreviations first
            if word in self.STATE_ABBREVIATIONS:
                expanded.append(self.STATE_ABBREVIATIONS[word])
            elif word in self.STREET_ABBREVIATIONS:
                expanded.append(self.STREET_ABBREVIATIONS[word])
            else:
                expanded.append(word)

        return ' '.join(expanded)

    def _values_equal(self, a: Any, b: Any) -> bool:
        if a is None or b is None:
            return a is None and b is None

        a_str, b_str = str(a), str(b)

        # Direct comparison
        if a_str == b_str:
            return True

        # Token-based comparison
        a_tokens = set(a_str.split())
        b_tokens = set(b_str.split())

        # Check if most tokens match (70% threshold for addresses)
        if len(a_tokens) > 0 and len(b_tokens) > 0:
            overlap = len(a_tokens & b_tokens)
            min_size = min(len(a_tokens), len(b_tokens))
            if min_size > 0 and overlap / min_size >= 0.7:
                return True

        return False


class NormalizedCoordinates(NormalizedType):
    """Normalized geographic coordinates (latitude, longitude).

    Handles various formats:
    - Decimal degrees: "40.4406, -79.9959"
    - DMS format: "40°26'26.2\"N 79°59'45.2\"W"
    - Object format: {"lat": 40.4406, "lon": -79.9959}
    """

    # Tolerance in degrees (approximately 100 meters at equator)
    TOLERANCE_DEGREES = 0.001

    def _normalize(self, value: Any, **kwargs: Any) -> tuple[float, float] | None:
        if value is None:
            return None

        # Handle dict format
        if isinstance(value, dict):
            lat = value.get('lat') or value.get('latitude')
            lon = value.get('lon') or value.get('lng') or value.get('longitude')
            if lat is not None and lon is not None:
                try:
                    return (float(lat), float(lon))
                except (ValueError, TypeError):
                    pass

        # Handle tuple/list format
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            try:
                return (float(value[0]), float(value[1]))
            except (ValueError, TypeError):
                pass

        # Handle string format
        s = str(value).strip()

        # Try decimal degrees format: "40.4406, -79.9959"
        match = re.match(r'^([-\d.]+)[,\s]+([-\d.]+)$', s)
        if match:
            try:
                return (float(match.group(1)), float(match.group(2)))
            except ValueError:
                pass

        # Try DMS format (simplified)
        # TODO: Add full DMS parsing if needed

        logger.warning(f"Could not normalize '{value}' as coordinates")
        return None

    def _values_equal(self, a: Any, b: Any) -> bool:
        if a is None or b is None:
            return a is None and b is None

        try:
            if not (isinstance(a, tuple) and isinstance(b, tuple)):
                return False

            lat_diff = abs(a[0] - b[0])
            lon_diff = abs(a[1] - b[1])

            return lat_diff <= self.TOLERANCE_DEGREES and lon_diff <= self.TOLERANCE_DEGREES
        except (TypeError, IndexError):
            return False


class NormalizedList(NormalizedType):
    """Normalized list for ordered/unordered comparison."""

    def __init__(
        self, value: Any, item_type: type[NormalizedType] = NormalizedString, ordered: bool = False, **kwargs: Any
    ):
        self.item_type = item_type
        self.ordered = ordered
        super().__init__(value, **kwargs)

    def _normalize(self, value: Any, **kwargs: Any) -> tuple:
        if value is None:
            return ()

        if isinstance(value, str):
            # Try to parse as JSON array
            import json

            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                # Split by comma
                value = [v.strip() for v in value.split(',')]

        if not isinstance(value, (list, tuple)):
            value = [value]

        # Normalize each item
        return tuple(self.item_type(item).normalized for item in value)

    def _values_equal(self, a: Any, b: Any) -> bool:
        if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)):
            return False

        if len(a) != len(b):
            return False

        if self.ordered:
            return all(self.item_type(x) == self.item_type(y) for x, y in zip(a, b, strict=False))
        # Unordered: check if all items in a are in b
        b_remaining = list(b)
        for item_a in a:
            found = False
            for i, item_b in enumerate(b_remaining):
                if self.item_type(item_a) == self.item_type(item_b):
                    b_remaining.pop(i)
                    found = True
                    break
            if not found:
                return False
        return True


# Type registry for schema-driven normalization
TYPE_REGISTRY: dict[str, type[NormalizedType]] = {
    # Basic types
    'string': NormalizedString,
    'number': NormalizedNumber,
    'integer': NormalizedNumber,
    'float': NormalizedNumber,
    'currency': NormalizedCurrency,
    'boolean': NormalizedBoolean,
    'date': NormalizedDate,
    'url': NormalizedUrl,
    # WebArena-Verified specific types
    'duration': NormalizedDuration,
    'distance': NormalizedDistance,
    'location-name': NormalizedLocationName,
    'location_name': NormalizedLocationName,  # Alternative naming
    'full_address': NormalizedFullAddress,
    'full-address': NormalizedFullAddress,
    'address': NormalizedFullAddress,
    'coordinates': NormalizedCoordinates,
    'geo': NormalizedCoordinates,
}


def get_normalized_type(type_name: str) -> type[NormalizedType]:
    """Get NormalizedType class by name."""
    return TYPE_REGISTRY.get(type_name, NormalizedString)
