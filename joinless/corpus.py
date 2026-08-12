# SPDX-License-Identifier: MIT
"""The synthetic evaluation corpus (ADR-0004, ADR-0011, RFC-0005).

Eight perturbation families, one code path, one seed determining the whole result.

**Why one code path.** ``LabelledPair`` is the only representation of a pair. There is
no second, generation-only struct that a serialiser later re-derives fields from — the
in-memory attributes a caller reads and the record a caller would write out come from
the same object, so they cannot drift apart the way two independently maintained shapes
would (RFC-0005, issue #40).

**Why the split lives outside the pair.** RFC-0005's file schema has five fields, and a
split column is not one of them — that is a decision this module does not get to make
unilaterally. A pair's role is tracked in a mapping alongside the pairs rather than as a
field on ``LabelledPair``, so nothing about the schema changes by generating a corpus
instead of reading one.

**Why the label rule is asymmetric in a specific way.** Six families perturb a name
while keeping it the same entity — formatting, word order and abbreviation change
surface form; character noise and transliteration change it more aggressively — and are
label ``1``. Two families change the *entity*, and both are label ``0``, for two
different reasons a matcher can fail in two different directions:

* **near-miss negative** pairs a name with a lexically similar, different entity
  (shares tokens or characters). This is the failure a **classical**, string-based arm
  is expected to be tempted by.
* **semantic alias** pairs a name with a *different* entity whose name means nearly the
  same thing, sharing little or no surface form (``Riverbend Bakery`` /
  ``Stonefield Bread House``). This is the failure an **embedding-style** arm is
  expected to be tempted by — calling two different businesses the same because their
  names are semantically close. Without this family that failure mode goes unmeasured
  entirely, because every other family that varies surface form keeps the same entity.

A benchmark containing only cases a matcher already handles well measures the fixture
author, not the matcher (issue #37) — semantic alias is the family that keeps that from
being true of this one.

**Why the two negative families are three times the size of a positive one.** Six
families carry label ``1`` and two carry label ``0``; drawing every family at the same
size would make the corpus 75% positive, and a matcher that answers "yes" to everything
would score close to that on F1 — exactly the gaming issue #37 exists to price out.
Positive families draw ``_POSITIVE_FAMILY_SIZE`` pairs each; negative families draw
``_NEGATIVE_FAMILY_SIZE`` pairs each, chosen as exactly three times the positive size so
``6 * _POSITIVE_FAMILY_SIZE == 2 * _NEGATIVE_FAMILY_SIZE`` — the corpus is exactly 50%
label ``1`` regardless of seed. Against that balance, a matcher that always answers
"yes" gets precision 0.5, recall 1.0, F1 ≈ 0.667: a real, visible penalty, and one
``test_corpus.py`` asserts directly.

**Why every family draws from the seed's ``random.Random``, not just some of them.** A
corpus is claimed to be "a pure function of seed" (ADR-0011 rule 3 depends on this: it
is what makes pooling several seeds show real variation instead of the same rows N
times). Every base name below is drawn via ``rng.sample`` or ``rng.choice`` from an
invented pool larger than what any one seed uses, including the three families
(exact, abbreviation, transliteration) whose *transformation* is fixed — the seed still
decides *which* invented names go into the corpus, so two seeds never produce the same
480 rows in the same order, or usually even the same set of names.

**Why base names never repeat across families.** Two families disagreeing about the
same base name — one calling a lexically close variant of it a positive alias, another
calling it a near-miss negative — makes both families jointly unsatisfiable for any
similarity-based matcher: whichever ordering the matcher's threshold respects, one of
the two families is wrong. Every family therefore draws its base names from its own
disjoint slice of the shared name-component pools, so no left-hand name a family emits
can appear in another family at all.

**Why calibration is sized the way it is.** ADR-0011 rule 2 selects a threshold from
calibration alone; a calibration split with only one or two negatives per family makes
that selection degenerate (accepting everything already scores near-perfectly on so
little data). With ``_POSITIVE_FAMILY_SIZE = 30`` and ``_NEGATIVE_FAMILY_SIZE = 90``,
``_split_into_roles``'s ratios put roughly 7 pairs per positive family and 22 per
negative family into calibration — 86 pairs total, comfortably double digits for both
classes, which is what makes a sweep over the calibration set mean something.
"""

from __future__ import annotations

import random
import string
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, cast

FAMILIES: tuple[str, ...] = (
    "exact",
    "formatting",
    "word order",
    "abbreviation",
    "character noise",
    "semantic alias",
    "transliteration",
    "near-miss negative",
)

# Both negative families change the *entity*, not just the surface form — see the
# module docstring for why there have to be two of them and what each one measures.
_NEGATIVE_FAMILIES: frozenset[str] = frozenset({"semantic alias", "near-miss negative"})

# Chosen so 6 positive families * _POSITIVE_FAMILY_SIZE == 2 negative families *
# _NEGATIVE_FAMILY_SIZE, which is what makes the corpus exactly 50% label 1 regardless
# of seed (module docstring). _NEGATIVE_FAMILY_SIZE must stay 3x _POSITIVE_FAMILY_SIZE
# for that identity to hold if either constant changes.
_POSITIVE_FAMILY_SIZE = 30
_NEGATIVE_FAMILY_SIZE = 90

# The canonical seed set a run record draws corpora from (ADR-0011 rule 3). Five is
# enough for a per-family report to show a range rather than a single draw, while
# keeping a full generation fast enough to run on every change.
SEEDS: tuple[int, ...] = (1, 2, 3, 4, 5)

Role = Literal["development", "calibration", "sealed test"]

ROLES: tuple[Role, ...] = ("development", "calibration", "sealed test")


@dataclass(frozen=True, slots=True)
class LabelledPair:
    """One row of the RFC-0005 labelled-pairs schema.

    Only ``label`` is validated against a closed domain, because RFC-0005 fixes it to
    ``{0, 1}``. ``category`` and ``pair_id`` are left open — RFC-0005 defines
    ``category`` as an optional, user-defined string for a supplied file, so this
    constructor must accept whatever a supplied file's category column says rather
    than rejecting anything outside this module's own eight family names. The
    generator in this module still only ever emits categories from ``FAMILIES``; that
    is a property of what ``generate_corpus`` builds (it is the only thing reading
    ``FAMILIES`` to begin with), not something this type enforces on every caller.
    """

    left_name: str
    right_name: str
    label: int
    category: str | None = None
    pair_id: str | None = None

    def __post_init__(self) -> None:
        if self.label not in (0, 1):
            raise ValueError(f"label must be 0 or 1, got {self.label!r}")

    def as_record(self) -> dict[str, str | int | None]:
        """The RFC-0005 row shape. Every serialiser this pair ever feeds reads it
        from here, not from re-listing the fields a second time."""
        return {
            "left_name": self.left_name,
            "right_name": self.right_name,
            "label": self.label,
            "category": self.category,
            "pair_id": self.pair_id,
        }


@dataclass(frozen=True, slots=True)
class Corpus:
    """A generated corpus: its pairs, and the role each pair's id was assigned.

    ``roles`` is keyed by ``pair_id`` rather than nested by role, because that is the
    shape "recorded per pair" (issue #38) asks for — a reader with one pair in hand
    looks its role up directly rather than searching three collections.

    Validated at construction, not only inside ``generate_corpus``: any caller that
    builds a ``Corpus`` directly — including a future loader for a supplied file — gets
    the same check that pair ids are unique, that ``roles`` covers exactly those ids
    and no others, and that every role named is one of ``ROLES``. A corpus assembled
    elsewhere cannot slip past this by skipping ``generate_corpus`` (issue #38).
    """

    seed: int
    pairs: tuple[LabelledPair, ...]
    roles: Mapping[str, Role]

    # roles holds a Mapping (a MappingProxyType in practice), which is itself
    # unhashable, so the hash a frozen dataclass would otherwise synthesise fails the
    # moment something actually calls hash() on a Corpus — several frames inside
    # dict's own hashing code, reporting "unhashable type: 'dict'" as if Corpus were
    # never involved. Declaring __hash__ = None makes "a Corpus cannot be hashed" a
    # stated fact about the type instead of a surprise at the call site.
    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        # pair_id is str | None (LabelledPair leaves it open for a supplied file
        # that has none), so sorting keys on str(...) rather than the raw value -
        # None has no ordering against str, but every value has a string form.
        pair_ids = [pair.pair_id for pair in self.pairs]
        duplicates = sorted(
            {pid for pid in pair_ids if pair_ids.count(pid) > 1}, key=str
        )
        if duplicates:
            raise ValueError(f"duplicate pair id(s): {duplicates!r}")

        if set(self.roles) != set(pair_ids):
            missing = sorted(set(pair_ids) - set(self.roles), key=str)
            extra = sorted(set(self.roles) - set(pair_ids), key=str)
            raise ValueError(
                "roles must cover exactly the corpus's pair ids "
                f"(missing {missing!r}, extra {extra!r})"
            )

        invalid_roles = sorted(
            {role for role in self.roles.values() if role not in ROLES}
        )
        if invalid_roles:
            raise ValueError(f"unknown role(s): {invalid_roles!r}")


def generate_corpus(seed: int) -> Corpus:
    """Build the full corpus for one seed: every family, then a role per pair.

    One ``random.Random(seed)`` instance is threaded through family generation and
    role assignment, in the fixed ``FAMILIES`` order, so the entire sequence of draws
    — and therefore the entire corpus — is a pure function of ``seed``.
    """
    rng = random.Random(seed)
    pairs = _build_pairs(seed, rng)
    role_by_pair_id = _split_into_roles(pairs, rng)
    return Corpus(seed=seed, pairs=pairs, roles=MappingProxyType(role_by_pair_id))


def generate_corpora(seeds: Iterable[int] = SEEDS) -> tuple[Corpus, ...]:
    """Build one corpus per seed, in the given order.

    This is the surface a run record's cross-seed report (ADR-0011 rule 3) consumes.
    It adds no state beyond "these are the seeds used" — every corpus in the result is
    independently reproducible from its own seed via ``generate_corpus``.
    """
    return tuple(generate_corpus(seed) for seed in seeds)


def _slug(family: str) -> str:
    return family.replace(" ", "-")


def _build_pairs(seed: int, rng: random.Random) -> tuple[LabelledPair, ...]:
    pairs: list[LabelledPair] = []
    for family in FAMILIES:
        label = 0 if family in _NEGATIVE_FAMILIES else 1
        for index, (left, right) in enumerate(_FAMILY_GENERATORS[family](rng)):
            pair_id = f"{seed:04d}-{_slug(family)}-{index:03d}"
            pairs.append(
                LabelledPair(
                    pair_id=pair_id,
                    left_name=left,
                    right_name=right,
                    label=label,
                    category=family,
                )
            )
    return tuple(pairs)


def _split_into_roles(
    pairs: tuple[LabelledPair, ...], rng: random.Random
) -> dict[str, Role]:
    """Stratify by family, then split each family's ids across the three roles.

    Splitting within each family rather than across the whole corpus is what keeps
    every family reportable in every role — a family-blind split could, by chance,
    place all of one family's pairs in development and leave calibration or sealed
    test with nothing to report for it (ADR-0011 rule 3).

    Ratios are not dictated by the ADR. Development gets half, since it is inspected
    freely during fixture and matcher work and benefits most from volume; the
    remainder splits as evenly as possible between calibration and sealed test, the
    two roles ADR-0011 requires to be reported at all. Splitting the remainder in two
    can itself leave one pair over; that pair goes to sealed test, not development —
    development already absorbed the corpus-level rounding via the ceiling division
    above it, so a second remainder landing there too would let development drift
    further ahead every time a family's count is odd.
    """
    by_category: dict[str, list[str]] = {family: [] for family in FAMILIES}
    for pair in pairs:
        category = pair.category
        assert category is not None  # every generated pair sets it (see LabelledPair)
        by_category[category].append(cast(str, pair.pair_id))

    roles: dict[str, list[str]] = {role: [] for role in ROLES}
    for ids in by_category.values():
        shuffled = list(ids)
        rng.shuffle(shuffled)
        n_dev = -(-len(shuffled) // 2)  # ceil division without importing math
        remaining = shuffled[n_dev:]
        n_cal = len(remaining) // 2
        roles["development"].extend(shuffled[:n_dev])
        roles["calibration"].extend(remaining[:n_cal])
        roles["sealed test"].extend(remaining[n_cal:])
    return {pair_id: cast(Role, role) for role, ids in roles.items() for pair_id in ids}


# --- Invented name components ----------------------------------------------------
#
# Every word below is an invented component: no real business, in any jurisdiction,
# carries these names (ADR-0004). Each family draws its base names from its own
# disjoint slice of the combined pool below, never from another family's slice, so
# the same left-hand name can never appear in two families (see the module
# docstring's "why base names never repeat across families").

_GEO_QUALIFIERS: tuple[str, ...] = (
    "Acacia",
    "Copper",
    "Savannah",
    "Highland",
    "Riverside",
    "Crescent",
    "Baobab",
    "Lakeside",
    "Nile",
    "Silver",
    "Cedar",
    "Ironwood",
    "Golden",
    "Amber",
    "Driftwood",
    "Thornwood",
    "Emerald",
    "Sunset",
    "Meadow",
    "Cypress",
    "Juniper",
    "Marble",
    "Frost",
    "Coral",
    "Willow",
    "Granite",
    "Sandstone",
    "Bluebell",
    "Foxglove",
    "Ebony",
    "Maple",
    "Birch",
)  # 32

_GEO_FEATURES: tuple[str, ...] = (
    "Ridge",
    "Hill",
    "Grove",
    "Bay",
    "Reef",
    "Valley",
    "Basin",
    "Cove",
    "Marsh",
    "Delta",
    "Cliff",
    "Hollow",
    "Mist",
    "Star",
    "Palm",
    "Point",
    "Glen",
    "Bend",
    "Vale",
    "Crossing",
    "Landing",
    "Terrace",
    "Heights",
    "Springs",
    "Junction",
)  # 25

_CATEGORY_NOUNS: tuple[str, ...] = (
    "Bakery",
    "Traders",
    "Carpentry",
    "Warehouse",
    "Electricals",
    "Textiles",
    "Grocery",
    "Pharmacy",
    "Cleaners",
    "Motors",
    "Hardware",
    "Logistics",
)

_NEAR_MISS_NOUNS: tuple[str, ...] = (
    "Bistro",
    "Deli",
    "Clinic",
    "Tailors",
    "Autos",
    "Studio",
    "Depot",
    "Outfitters",
    "Foundry",
    "Garage",
    "Boutique",
    "Stationers",
)

_SEMANTIC_ALIAS_MAP: dict[str, tuple[str, ...]] = {
    "Bakery": ("Bread House", "Bake Shop"),
    "Grocery": ("Supermarket", "Food Mart"),
    "Pharmacy": ("Chemist", "Drug Store"),
    "Cleaners": ("Laundry Service", "Wash House"),
    "Motors": ("Automotive", "Auto Works"),
    "Hardware": ("Tool Supply", "Building Supplies"),
}

# Semantic alias needs its LEFT entity and RIGHT entity to share no word at all
# (module docstring). Slicing one shared pool by index only rules out the two
# sides ever being the *same* prefix; two different prefixes drawn from the same
# qualifier/feature vocabulary can still share one of the two words by chance.
# These four banks are wholly separate from _GEO_QUALIFIERS/_GEO_FEATURES and
# from each other, so a left prefix and a right prefix can never share a word by
# construction, and no retry loop is needed.
_ALIAS_LEFT_QUALIFIERS: tuple[str, ...] = (
    "Sunrise",
    "Timberline",
    "Falcon",
    "Harborview",
    "Windmere",
    "Oakhaven",
    "Brambleton",
    "Silvercreek",
    "Fernwood",
    "Wolfsbane",
)
_ALIAS_LEFT_FEATURES: tuple[str, ...] = (
    "Court",
    "Corner",
    "Way",
    "Row",
    "Path",
    "Yard",
    "Green",
    "Square",
    "Trail",
    "Loop",
)
_ALIAS_RIGHT_QUALIFIERS: tuple[str, ...] = (
    "Moonrise",
    "Ashgrove",
    "Kestrel",
    "Pinehollow",
    "Stormcrest",
    "Larkspur",
    "Ravenswood",
    "Goldbrook",
    "Hazelmere",
    "Thistledown",
)
_ALIAS_RIGHT_FEATURES: tuple[str, ...] = (
    "Close",
    "Walk",
    "Passage",
    "Circle",
    "Mews",
    "Parade",
    "Esplanade",
    "Promenade",
    "Arcade",
    "Wharf",
)

_ABBREVIATION_MAP: dict[str, str] = {
    "Company": "Co",
    "Limited": "Ltd",
    "Brothers": "Bros",
    "Associates": "Assoc",
    "Enterprises": "Ent",
    "Corporation": "Corp",
}

_ACCENTED_QUALIFIERS: tuple[str, ...] = (
    "Café",
    "Château",
    "Señor",
    "Île",
    "Möwe",
    "Zürich",
    "Öresund",
    "Björk",
    "Ångström",
    "Élan",
    "Façade",
    "Røstvik",
)

_TRANSLITERATION_MAP: dict[str, str] = {
    "á": "a",
    "â": "a",
    "å": "a",
    "ç": "c",
    "é": "e",
    "è": "e",
    "ê": "e",
    "ë": "e",
    "î": "i",
    "ï": "i",
    "ñ": "n",
    "ö": "o",
    "ø": "o",
    "ü": "u",
}
# Every lower-case mapping gets an upper-case counterpart too, since a name is free
# to carry the accented letter at the start of a word ("Étoile") and this map would
# otherwise strip the letter everywhere except where it is most visible.
_TRANSLITERATION_MAP.update(
    {char.upper(): repl.upper() for char, repl in _TRANSLITERATION_MAP.items()}
)

# One pool of two-word geographic prefixes, sliced below into a disjoint span per
# family. 32 qualifiers * 25 features = 800, comfortably more than the 550 the six
# slices below claim between them.
_PREFIX_POOL: tuple[str, ...] = tuple(
    f"{qualifier} {feature}"
    for qualifier in _GEO_QUALIFIERS
    for feature in _GEO_FEATURES
)


def _pool_slice(start: int, size: int) -> tuple[str, ...]:
    return _PREFIX_POOL[start : start + size]


_EXACT_PREFIXES = _pool_slice(0, 70)
_FORMATTING_PREFIXES = _pool_slice(70, 70)
_WORD_ORDER_PREFIXES = _pool_slice(140, 70)
_ABBREVIATION_PREFIXES = _pool_slice(210, 70)
_CHARACTER_NOISE_PREFIXES = _pool_slice(280, 70)
_NEAR_MISS_PREFIXES = _pool_slice(350, 200)

_ALIAS_LEFT_POOL: tuple[str, ...] = tuple(
    f"{qualifier} {feature}"
    for qualifier in _ALIAS_LEFT_QUALIFIERS
    for feature in _ALIAS_LEFT_FEATURES
)
_ALIAS_RIGHT_POOL: tuple[str, ...] = tuple(
    f"{qualifier} {feature}"
    for qualifier in _ALIAS_RIGHT_QUALIFIERS
    for feature in _ALIAS_RIGHT_FEATURES
)

_TRANSLITERATION_POOL: tuple[str, ...] = tuple(
    f"{qualifier} {noun}"
    for qualifier in _ACCENTED_QUALIFIERS
    for noun in _CATEGORY_NOUNS
)


def _scramble_word(rng: random.Random, word: str) -> str:
    """Replace roughly half of a word's letters with different letters.

    At least one position always changes when the word has any letters at all
    (``max(1, ...)``), so the result is never equal to the input in that case;
    changing about half rather than one character is what keeps the result from
    accidentally landing on another real word in the corpus. A word with no
    alphabetic characters at all (not something any base name below contains, but not
    ruled out by the type either) is returned unchanged rather than raising — there is
    nothing in it to scramble.
    """
    chars = list(word)
    alpha_positions = [i for i, ch in enumerate(chars) if ch.isalpha()]
    if not alpha_positions:
        return word
    swap_count = max(1, len(alpha_positions) // 2)
    for i in rng.sample(alpha_positions, k=swap_count):
        current = chars[i].lower()
        replacement = rng.choice([c for c in string.ascii_lowercase if c != current])
        chars[i] = replacement.upper() if chars[i].isupper() else replacement
    return "".join(chars)


def _exact_pairs(rng: random.Random) -> list[tuple[str, str]]:
    names = [
        f"{prefix} {rng.choice(_CATEGORY_NOUNS)}"
        for prefix in rng.sample(_EXACT_PREFIXES, k=_POSITIVE_FAMILY_SIZE)
    ]
    return [(name, name) for name in names]


def _formatting_pairs(rng: random.Random) -> list[tuple[str, str]]:
    styles: tuple[Callable[[str], str], ...] = (
        str.upper,
        str.lower,
        lambda s: f"{s}.",
        lambda s: s.replace(" ", "  "),
    )
    result: list[tuple[str, str]] = []
    for prefix in rng.sample(_FORMATTING_PREFIXES, k=_POSITIVE_FAMILY_SIZE):
        name = f"{prefix} {rng.choice(_CATEGORY_NOUNS)}"
        result.append((name, rng.choice(styles)(name)))
    return result


def _word_order_pairs(rng: random.Random) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for prefix in rng.sample(_WORD_ORDER_PREFIXES, k=_POSITIVE_FAMILY_SIZE):
        name = f"{prefix} {rng.choice(_CATEGORY_NOUNS)}"
        words = name.split()
        shuffled = list(words)
        while shuffled == words:
            rng.shuffle(shuffled)
        result.append((name, " ".join(shuffled)))
    return result


def _abbreviation_pairs(rng: random.Random) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for prefix in rng.sample(_ABBREVIATION_PREFIXES, k=_POSITIVE_FAMILY_SIZE):
        suffix = rng.choice(tuple(_ABBREVIATION_MAP))
        name = f"{prefix} {suffix}"
        abbreviated = f"{prefix} {_ABBREVIATION_MAP[suffix]}"
        result.append((name, abbreviated))
    return result


def _character_noise_pairs(rng: random.Random) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for prefix in rng.sample(_CHARACTER_NOISE_PREFIXES, k=_POSITIVE_FAMILY_SIZE):
        name = f"{prefix} {rng.choice(_CATEGORY_NOUNS)}"
        scrambled = [_scramble_word(rng, word) for word in name.split()]
        result.append((name, " ".join(scrambled)))
    return result


def _semantic_alias_pairs(rng: random.Random) -> list[tuple[str, str]]:
    """Two DIFFERENT entities whose names mean nearly the same thing.

    ``left`` is drawn from a dedicated pool, ``right`` from a second pool built out
    of an entirely separate set of words — so a left prefix and a right prefix can
    never share a qualifier or feature word, by construction rather than by retrying
    an unlucky draw. Combined with a category noun and one of its curated synonyms
    on the two sides, the pair shares no token at all, the shape the family exists to
    test (module docstring). Sampling each side without replacement also keeps every
    left-hand name in the family unique, since the prefix alone already differs
    pair to pair.
    """
    left_prefixes = rng.sample(_ALIAS_LEFT_POOL, k=_NEGATIVE_FAMILY_SIZE)
    right_prefixes = rng.sample(_ALIAS_RIGHT_POOL, k=_NEGATIVE_FAMILY_SIZE)
    result: list[tuple[str, str]] = []
    for prefix_a, prefix_b in zip(left_prefixes, right_prefixes, strict=True):
        noun = rng.choice(tuple(_SEMANTIC_ALIAS_MAP))
        synonym = rng.choice(_SEMANTIC_ALIAS_MAP[noun])
        result.append((f"{prefix_a} {noun}", f"{prefix_b} {synonym}"))
    return result


def _transliteration_pairs(rng: random.Random) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for base in rng.sample(_TRANSLITERATION_POOL, k=_POSITIVE_FAMILY_SIZE):
        stripped = "".join(_TRANSLITERATION_MAP.get(ch, ch) for ch in base)
        result.append((base, stripped))
    return result


def _near_miss_negative_pairs(rng: random.Random) -> list[tuple[str, str]]:
    """A lexically similar, DIFFERENT entity: same location words, different
    business — the failure a classical, string-based arm is expected to be tempted
    by (module docstring)."""
    result: list[tuple[str, str]] = []
    for prefix in rng.sample(_NEAR_MISS_PREFIXES, k=_NEGATIVE_FAMILY_SIZE):
        base_noun = rng.choice(_CATEGORY_NOUNS)
        other_noun = rng.choice(_NEAR_MISS_NOUNS)
        result.append((f"{prefix} {base_noun}", f"{prefix} {other_noun}"))
    return result


_FAMILY_GENERATORS: dict[str, Callable[[random.Random], list[tuple[str, str]]]] = {
    "exact": _exact_pairs,
    "formatting": _formatting_pairs,
    "word order": _word_order_pairs,
    "abbreviation": _abbreviation_pairs,
    "character noise": _character_noise_pairs,
    "semantic alias": _semantic_alias_pairs,
    "transliteration": _transliteration_pairs,
    "near-miss negative": _near_miss_negative_pairs,
}
