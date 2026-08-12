# SPDX-License-Identifier: MIT
"""The synthetic corpus: deterministic generation, disjoint roles, honest families.

ADR-0011 rule 1 exists to stop a threshold selected on calibration data from being
partly fitted to the sealed test set. Every property tested here is a property that
rule depends on: reproducibility (so a seed names one corpus, not a family of them),
disjoint roles (so no pair funds both tuning and reporting), and per-family fidelity
(so "the corpus contains hard cases for both arms" is checked rather than assumed).
"""

from __future__ import annotations

import itertools
import random
import re
from types import MappingProxyType

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from joinless.corpus import (
    _NEGATIVE_FAMILIES,
    _SEMANTIC_ALIAS_MAP,
    FAMILIES,
    ROLES,
    SEEDS,
    Corpus,
    LabelledPair,
    Role,
    _scramble_word,
    generate_corpora,
    generate_corpus,
)

_SEEDS = st.integers(min_value=0, max_value=10_000)


def _role_of(corpus: Corpus, pair: LabelledPair) -> Role:
    """``LabelledPair.pair_id`` is ``str | None`` (RFC-0005 leaves it open for a
    supplied file), but every pair a generated corpus produces sets one - this
    narrows that for the tests below rather than repeating the assert in each."""
    assert pair.pair_id is not None
    return corpus.roles[pair.pair_id]


def _family_of(pair: LabelledPair) -> str:
    assert pair.category is not None
    return pair.category


def _normalise(name: str) -> str:
    """Collapse whitespace/punctuation noise so a formatting perturbation's words
    can be compared to its base without the surface change getting in the way."""
    return re.sub(r"[.\s]+", " ", name).strip().lower()


# --- Determinism -------------------------------------------------------------------


@given(seed=_SEEDS)
def test_the_same_seed_reproduces_the_same_pairs_and_roles(seed: int) -> None:
    first = generate_corpus(seed)
    second = generate_corpus(seed)

    assert first.pairs == second.pairs
    assert dict(first.roles) == dict(second.roles)


@given(seed=_SEEDS)
def test_a_corpus_records_the_seed_that_produced_it(seed: int) -> None:
    assert generate_corpus(seed).seed == seed


@given(seed_a=_SEEDS, seed_b=_SEEDS)
def test_different_seeds_change_every_familys_content(seed_a: int, seed_b: int) -> None:
    """The seed test that cannot pass by construction alone: pair_id embeds the
    seed (so ``pairs != pairs`` would hold even if the seed stopped influencing
    anything else), so this compares the drawn names, with ids excluded.

    Checked per family, not just over the corpus as a whole - a single family
    that stopped drawing from the rng would still leave the *aggregate* name set
    different, because the other seven families still vary; only a per-family
    comparison can catch one family going seed-invariant on its own."""
    assume(seed_a != seed_b)

    corpus_a = generate_corpus(seed_a)
    corpus_b = generate_corpus(seed_b)

    for family in FAMILIES:
        names_a = {
            (p.left_name, p.right_name) for p in corpus_a.pairs if p.category == family
        }
        names_b = {
            (p.left_name, p.right_name) for p in corpus_b.pairs if p.category == family
        }
        assert names_a != names_b, family


def test_generate_corpora_builds_one_corpus_per_seed_in_order() -> None:
    seeds = (11, 22, 33)
    corpora = generate_corpora(seeds)

    assert tuple(corpus.seed for corpus in corpora) == seeds
    assert corpora == tuple(generate_corpus(seed) for seed in seeds)


def test_generate_corpora_defaults_to_the_canonical_seed_set() -> None:
    assert generate_corpora() == tuple(generate_corpus(seed) for seed in SEEDS)


# --- Corpus validation at construction -----------------------------------------------


def test_corpus_rejects_a_duplicate_pair_id() -> None:
    """Issue #38 bullet 3: this check has to run wherever a Corpus is built, not
    only inside generate_corpus, so a corpus assembled elsewhere cannot slip past
    it."""
    first = LabelledPair(left_name="A", right_name="B", label=1, pair_id="dup")
    second = LabelledPair(left_name="C", right_name="D", label=1, pair_id="dup")

    with pytest.raises(ValueError, match="dup"):
        Corpus(
            seed=1,
            pairs=(first, second),
            roles=MappingProxyType({"dup": "development"}),
        )


def test_corpus_rejects_roles_that_do_not_cover_every_pair() -> None:
    pair = LabelledPair(left_name="A", right_name="B", label=1, pair_id="p1")

    with pytest.raises(ValueError, match="roles"):
        Corpus(seed=1, pairs=(pair,), roles=MappingProxyType({}))


def test_corpus_rejects_a_role_key_with_no_matching_pair() -> None:
    pair = LabelledPair(left_name="A", right_name="B", label=1, pair_id="p1")

    with pytest.raises(ValueError, match="roles"):
        Corpus(
            seed=1,
            pairs=(pair,),
            roles=MappingProxyType({"p1": "development", "ghost": "development"}),
        )


def test_corpus_rejects_a_role_outside_the_known_roles() -> None:
    pair = LabelledPair(left_name="A", right_name="B", label=1, pair_id="p1")

    bad_roles: dict[str, str] = {"p1": "unknown-role"}
    with pytest.raises(ValueError, match="unknown role"):
        Corpus(seed=1, pairs=(pair,), roles=MappingProxyType(bad_roles))  # type: ignore[arg-type]


def test_corpus_roles_is_read_only() -> None:
    corpus = generate_corpus(1)
    some_pair_id = next(iter(corpus.roles))

    with pytest.raises(TypeError):
        corpus.roles[some_pair_id] = "development"  # type: ignore[index]


def test_corpus_is_explicitly_unhashable() -> None:
    """roles holds a Mapping, which is itself unhashable, so Corpus was already
    unhashable before this fix - just unhelpfully: without a declared __hash__,
    dataclass synthesises one that tries to hash every field and fails several
    frames inside dict's own hashing code, reporting "unhashable type: 'dict'" as
    if Corpus were never involved. Declaring __hash__ = None makes the message
    name Corpus itself, which is the only part of this a test can tell apart from
    the accidental version."""
    with pytest.raises(TypeError, match="Corpus"):
        hash(generate_corpus(1))


# --- Disjoint roles ------------------------------------------------------------------


@given(seed=_SEEDS)
def test_every_pair_gets_exactly_one_role_and_none_is_left_unassigned(
    seed: int,
) -> None:
    corpus = generate_corpus(seed)

    pair_ids = [pair.pair_id for pair in corpus.pairs]

    assert len(pair_ids) == len(set(pair_ids)), "pair ids must be unique"
    assert set(corpus.roles) == set(pair_ids)
    assert all(role in ROLES for role in corpus.roles.values())


@given(seed=_SEEDS)
def test_every_family_is_reportable_in_every_role(seed: int) -> None:
    """The property a flat pair_id -> role coverage check cannot see: a split that
    puts one family entirely in development, leaving nothing to report for it in
    calibration or sealed test, still satisfies "every id has a role"."""
    corpus = generate_corpus(seed)

    combos = {(_family_of(pair), _role_of(corpus, pair)) for pair in corpus.pairs}
    assert combos == set(itertools.product(FAMILIES, ROLES))


@given(seed=_SEEDS)
def test_calibration_holds_a_double_digit_count_for_every_family(seed: int) -> None:
    """ADR-0011 rule 2 selects a threshold from calibration alone; a handful of
    pairs per family makes that selection degenerate."""
    corpus = generate_corpus(seed)

    counts: dict[str, int] = {family: 0 for family in FAMILIES}
    for pair in corpus.pairs:
        if _role_of(corpus, pair) == "calibration":
            counts[_family_of(pair)] += 1

    for family, count in counts.items():
        assert count >= 5, (family, count)


# --- Class balance -------------------------------------------------------------------


@given(seed=_SEEDS)
def test_a_constant_yes_matcher_scores_poorly_on_the_corpus(seed: int) -> None:
    """Correction 3: a trivial always-yes matcher must not be rewarded. The corpus
    is built exactly 50% label 1 (module docstring), so an always-yes matcher gets
    precision 0.5, recall 1.0, F1 ~ 0.667 - a real, visible penalty."""
    corpus = generate_corpus(seed)

    positive = sum(1 for pair in corpus.pairs if pair.label == 1)
    negative = sum(1 for pair in corpus.pairs if pair.label == 0)

    assert positive == negative, "the corpus must be exactly balanced"

    precision_of_always_yes = positive / (positive + negative)
    recall_of_always_yes = 1.0
    f1_of_always_yes = (
        2
        * precision_of_always_yes
        * recall_of_always_yes
        / (precision_of_always_yes + recall_of_always_yes)
    )
    assert f1_of_always_yes < 0.75


# --- Family coverage and labels -----------------------------------------------------


@given(seed=_SEEDS)
def test_every_named_family_is_present_and_carries_the_right_label(
    seed: int,
) -> None:
    corpus = generate_corpus(seed)

    categories = {pair.category for pair in corpus.pairs}
    assert categories == set(FAMILIES)

    for pair in corpus.pairs:
        expected_label = 0 if pair.category in _NEGATIVE_FAMILIES else 1
        assert pair.label == expected_label, pair


@given(seed=_SEEDS)
def test_no_base_name_is_shared_between_two_families(seed: int) -> None:
    """The defect this guards: reusing a base name across a positive and a
    negative family makes both jointly unsatisfiable for any similarity-based
    matcher, since one similarity ordering would have to be correct for both a
    same-entity and a different-entity label. Every family draws its left-hand
    names from its own disjoint pool slice, so no name can repeat."""
    corpus = generate_corpus(seed)

    left_names = [pair.left_name for pair in corpus.pairs]
    assert len(left_names) == len(set(left_names))


# --- Individual family behaviour -----------------------------------------------------


def _pairs_in(seed: int, category: str) -> list[LabelledPair]:
    return [p for p in generate_corpus(seed).pairs if p.category == category]


@given(seed=_SEEDS)
def test_exact_pairs_repeat_the_base_name_unchanged(seed: int) -> None:
    for pair in _pairs_in(seed=seed, category="exact"):
        assert pair.left_name == pair.right_name


@given(seed=_SEEDS)
def test_formatting_pairs_change_surface_form_but_keep_the_same_words(
    seed: int,
) -> None:
    for pair in _pairs_in(seed=seed, category="formatting"):
        assert pair.left_name != pair.right_name
        assert _normalise(pair.left_name) == _normalise(pair.right_name)


@given(seed=_SEEDS)
def test_word_order_pairs_reorder_the_same_multiset_of_words(seed: int) -> None:
    for pair in _pairs_in(seed=seed, category="word order"):
        left_words = pair.left_name.split()
        right_words = pair.right_name.split()
        assert left_words != right_words
        assert sorted(left_words) == sorted(right_words)


@given(seed=_SEEDS)
def test_abbreviation_pairs_shorten_only_the_final_word(seed: int) -> None:
    for pair in _pairs_in(seed=seed, category="abbreviation"):
        left_words = pair.left_name.split()
        right_words = pair.right_name.split()
        assert left_words[:-1] == right_words[:-1]
        assert right_words[-1] != left_words[-1]
        assert len(right_words[-1]) < len(left_words[-1])


@given(seed=_SEEDS)
def test_character_noise_pairs_share_no_token_with_their_base(seed: int) -> None:
    """The property the family exists for: if a "noised" name still shares a token
    with its base, token-overlap comparison solves it for free and the family is
    not exercising the character-aware arm at all."""
    for pair in _pairs_in(seed=seed, category="character noise"):
        left_tokens = set(pair.left_name.lower().split())
        right_tokens = set(pair.right_name.lower().split())
        assert left_tokens.isdisjoint(right_tokens)


@given(seed=_SEEDS)
def test_transliteration_pairs_turn_accented_names_into_ascii_ones(
    seed: int,
) -> None:
    for pair in _pairs_in(seed=seed, category="transliteration"):
        assert not pair.left_name.isascii()
        assert pair.right_name.isascii()
        assert pair.left_name != pair.right_name


@given(seed=_SEEDS)
def test_near_miss_negative_pairs_are_a_different_entity_that_looks_similar(
    seed: int,
) -> None:
    for pair in _pairs_in(seed=seed, category="near-miss negative"):
        assert pair.label == 0
        assert pair.left_name != pair.right_name
        left_tokens = set(pair.left_name.lower().split())
        right_tokens = set(pair.right_name.lower().split())
        assert left_tokens & right_tokens, "a near-miss should still look similar"


@given(seed=_SEEDS)
def test_semantic_alias_pairs_are_a_different_entity_sharing_no_surface_form(
    seed: int,
) -> None:
    """Correction 1: semantic alias is label 0 - two DIFFERENT entities whose
    names mean similar things and share little or no surface form. This is the
    opposite direction from near-miss negative, and asserting both directly is
    what stops the two families from silently swapping labels again."""
    for pair in _pairs_in(seed=seed, category="semantic alias"):
        assert pair.label == 0
        left_tokens = set(pair.left_name.lower().split())
        right_tokens = set(pair.right_name.lower().split())
        assert left_tokens.isdisjoint(right_tokens)


@given(seed=_SEEDS)
def test_semantic_alias_pairs_replace_the_category_word_with_a_curated_synonym(
    seed: int,
) -> None:
    """Pins the specific perturbation, not just its surface effect: the
    replacement word must be a member of the curated synonym set, not merely
    "some word that differs" - a generator regressed to emit unrelated nouns
    would still pass a same-length-tail check but fails this one."""
    for pair in _pairs_in(seed=seed, category="semantic alias"):
        left_words = pair.left_name.split()
        right_words = pair.right_name.split()
        base_noun = left_words[-1]
        # right_name is always "{qualifier} {feature} {synonym...}" - exactly two
        # prefix words followed by the (possibly multi-word) synonym.
        synonym = " ".join(right_words[2:])

        assert base_noun in _SEMANTIC_ALIAS_MAP
        assert synonym in _SEMANTIC_ALIAS_MAP[base_noun]


# --- LabelledPair's own invariants --------------------------------------------------


def test_labelled_pair_rejects_a_label_outside_zero_or_one() -> None:
    with pytest.raises(ValueError, match="label"):
        LabelledPair(left_name="A", right_name="B", label=2)


def test_labelled_pair_accepts_a_user_defined_category() -> None:
    """RFC-0005 issue #40: category is optional and user-defined for a supplied
    file. The generator narrows to FAMILIES on its own; the shared row type must
    not reject a category a supplied file is free to use."""
    pair = LabelledPair(
        left_name="A", right_name="B", label=1, category="my own family"
    )
    assert pair.category == "my own family"


def test_labelled_pair_defaults_category_and_pair_id_to_none() -> None:
    pair = LabelledPair(left_name="A", right_name="B", label=1)
    assert pair.category is None
    assert pair.pair_id is None


def test_as_record_carries_exactly_the_rfc_0005_fields() -> None:
    pair = LabelledPair(
        pair_id="p-1", left_name="A", right_name="B", label=1, category="exact"
    )

    assert pair.as_record() == {
        "left_name": "A",
        "right_name": "B",
        "label": 1,
        "category": "exact",
        "pair_id": "p-1",
    }


# --- _scramble_word's own edge case --------------------------------------------------


def test_scramble_word_returns_a_word_unchanged_when_it_has_no_letters() -> None:
    """No base name below contains a word with zero letters, but the function's
    type does not rule one out, and it used to raise (ValueError: Sample larger
    than population) rather than returning the word unchanged."""
    assert _scramble_word(random.Random(0), "24/7") == "24/7"


@given(seed=_SEEDS)
def test_scramble_word_always_changes_a_word_made_only_of_letters(seed: int) -> None:
    rng = random.Random(seed)
    assert _scramble_word(rng, "Bakery") != "Bakery"
