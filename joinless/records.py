# SPDX-License-Identifier: MIT
"""Records and their identity.

Two sources describe the same entities and share no key, so identity has to be
derived from content. Deriving it from name and coordinates alone collapses two
same-named rows that carry no coordinates into one identifier — which loses data
in a second way immediately after the resolver went to trouble to keep it
(PRD FR-3, FR-4).

Identity is therefore **two** hashes over a canonical serialisation, each
answering a different question:

``content_id``
    Over the source namespace and the record's content. Two byte-identical rows
    from one source share it, and that is the point — it is what makes an exact
    duplicate visible instead of silent.

``record_id``
    Over the same serialisation plus the record's ordinal within its source.
    Every input row has its own, including exact duplicates.

**Exact duplicates are defined rather than left to the hash.** A hash is
collision-resistant, not collision-free, and two byte-identical rows from one
source cannot be separated by their content at all — there is nothing in them
that differs. The ordinal is what separates them, so it participates in
``record_id`` always and not only when a duplicate is detected. A scheme that
added it conditionally would give the same row different identifiers depending
on whether some other row happened to be present.

The source name is a namespace rather than a field, so two records that are
byte-identical apart from their origin cannot collide across sources.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field

_ENCODING = "utf-8"


@dataclass(frozen=True, slots=True)
class Record:
    """One row from one source.

    ``ordinal`` is the row's position within its source, and it is required
    rather than defaulted: identity depends on it, so a caller that omitted it
    would be given an identifier that silently disagreed with the one the same
    row got when read as part of its source.
    """

    source: str
    ordinal: int
    name: str
    latitude: float | None = None
    longitude: float | None = None
    fields: Mapping[str, str] = field(default_factory=dict)


def _canonical(record: Record) -> str:
    """Serialise a record's content to one stable string.

    ``sort_keys`` and a fixed separator matter: a dict's iteration order is its
    insertion order, so two records with equal content built in different field
    orders would otherwise serialise differently and be given different
    identifiers.
    """
    return json.dumps(
        {
            "name": record.name,
            "latitude": record.latitude,
            "longitude": record.longitude,
            "fields": dict(sorted(record.fields.items())),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _digest(*parts: str) -> str:
    """Hash parts under a namespace, with a separator that cannot appear in them.

    A null byte joins the parts because it cannot occur in a Python ``str`` that
    came from text input, so no combination of field values can be arranged to
    produce another combination's digest.
    """
    return hashlib.sha256("\0".join(parts).encode(_ENCODING)).hexdigest()


def content_id(record: Record) -> str:
    """Identify a record by its source and content, ignoring its position."""
    return _digest(record.source, _canonical(record))


def record_id(record: Record) -> str:
    """Identify one input row uniquely, including among exact duplicates."""
    return _digest(record.source, _canonical(record), str(record.ordinal))
