# SPDX-License-Identifier: MIT
"""The resolver: candidate generation, retention, merge policy, and resolve()."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest
from hypothesis import assume, example, given
from hypothesis import strategies as st

from joinless.records import Record, content_id, record_id
from joinless.resolver import (
    _REASON_NO_CANDIDATES,
    _REASON_NO_COORDINATES,
    _REASON_NO_MATCH_ABOVE_THRESHOLD,
    _REASON_NOT_SELECTED,
    DEFAULT_CELL_SIZE_DEGREES,
    ResolutionResult,
    _haversine_km,
    _populatedness,
    _winner,
    bucket_occupancy,
    candidate_pairs,
    merge,
    resolve,
)
from joinless.scoring import FuzzyScorer, OverlapScorer, ThresholdMatcher


def _record(
    source: str,
    ordinal: int,
    name: str,
    *,
    latitude: float | None = None,
    longitude: float | None = None,
    fields: Mapping[str, str] | None = None,
) -> Record:
    return Record(
        source=source,
        ordinal=ordinal,
        name=name,
        latitude=latitude,
        longitude=longitude,
        fields=fields or {},
    )


_OVERLAP_MATCHER = ThresholdMatcher(scorer=OverlapScorer(), threshold=0.5)


def test_unmatched_reason_strings_pin_their_actual_text() -> None:
    """Every other test that touches these constants imports them and compares
    ``entry.reason == _REASON_X`` - both sides are the same object, so a
    mutation that rewrites a constant's text to garbage passes every one of
    them unnoticed. These strings are caller-facing diagnostic text
    (``ResolutionResult``'s docstring is why they exist), so pin the literal
    words here, independent of the constants' own definitions."""
    assert _REASON_NO_COORDINATES == (
        "no coordinates: a record without them can never enter a candidate set (FR-3)"
    )
    assert (
        _REASON_NO_CANDIDATES
        == "no candidate record shares its nine-cell grid neighbourhood"
    )
    assert (
        _REASON_NO_MATCH_ABOVE_THRESHOLD
        == "no candidate's score met the matcher's threshold"
    )
    assert (
        _REASON_NOT_SELECTED == "not chosen as any left record's best-scoring candidate"
    )


# --- Candidate generation (issue #26) -----------------------------------------------


def test_bucket_occupancy_counts_records_sharing_a_cell() -> None:
    records = [
        _record("right", 0, "Acme Traders", latitude=0.2, longitude=0.2),
        _record("right", 1, "Riverside Bakery", latitude=0.4, longitude=0.4),
        _record("right", 2, "Highland Freight", latitude=5.0, longitude=5.0),
    ]
    occupancy = bucket_occupancy(records, cell_size=1.0)
    assert occupancy == {(0, 0): 2, (5, 5): 1}


def test_bucket_occupancy_excludes_coordinate_less_records() -> None:
    records = [
        _record("right", 0, "Acme Traders", latitude=0.0, longitude=0.0),
        _record("right", 1, "No Coordinates Co"),
    ]
    occupancy = bucket_occupancy(records, cell_size=1.0)
    assert sum(occupancy.values()) == 1


def test_bucket_occupancy_excludes_a_record_missing_only_one_coordinate() -> None:
    """``latitude`` and ``longitude`` are independent ``Optional`` fields on
    :class:`Record` - a row carrying only one of the two is exactly as
    unplaceable in a grid cell as one carrying neither, and must not be
    treated as coordinate-bearing just because one field happens to be set."""
    records = [
        _record("right", 0, "Half Coordinates Co", latitude=0.0, longitude=None),
        _record("right", 1, "Other Half Co", latitude=None, longitude=0.0),
    ]
    assert bucket_occupancy(records, cell_size=1.0) == {}


_ALL_NINE_NEIGHBOUR_OFFSETS: tuple[tuple[int, int], ...] = tuple(
    (row_delta, col_delta) for row_delta in (-1, 0, 1) for col_delta in (-1, 0, 1)
)


@pytest.mark.parametrize("row_delta,col_delta", _ALL_NINE_NEIGHBOUR_OFFSETS)
def test_candidate_pairs_finds_a_record_in_every_one_of_the_nine_neighbour_cells(
    row_delta: int, col_delta: int
) -> None:
    """FR-2's nine-cell neighbourhood, checked in each of the nine directions
    individually rather than spot-checked. A left record that loses a
    candidate simply becomes unmatched, which is still a valid traceable
    outcome under the no-record-dropped invariant - so a test that only
    exercises the centre cell and one diagonal cannot tell a correct offset
    list from one missing seven of its nine entries."""
    left = [_record("left", 0, "Acme Traders", latitude=0.5, longitude=0.5)]
    right = [
        _record(
            "right",
            0,
            "Acme Traders",
            latitude=row_delta + 0.5,
            longitude=col_delta + 0.5,
        )
    ]
    assert candidate_pairs(left, right, cell_size=1.0) == [(left[0], right[0])]


def test_candidate_pairs_excludes_a_record_two_cells_away() -> None:
    left = [_record("left", 0, "Acme Traders", latitude=0.1, longitude=0.1)]
    right = [_record("right", 0, "Acme Traders", latitude=2.5, longitude=0.1)]
    assert candidate_pairs(left, right, cell_size=1.0) == []


def test_candidate_pairs_is_empty_for_a_left_record_without_coordinates() -> None:
    left = [_record("left", 0, "Acme Traders")]
    right = [_record("right", 0, "Acme Traders", latitude=0.0, longitude=0.0)]
    assert candidate_pairs(left, right, cell_size=1.0) == []


def test_candidate_pairs_is_empty_when_no_right_record_carries_coordinates() -> None:
    left = [_record("left", 0, "Acme Traders", latitude=0.0, longitude=0.0)]
    right = [_record("right", 0, "Acme Traders")]
    assert candidate_pairs(left, right, cell_size=1.0) == []


@given(cell_count=st.integers(min_value=1, max_value=40))
def test_comparison_count_stays_linear_when_occupancy_is_bounded(
    cell_count: int,
) -> None:
    """FR-2's complexity claim, checked under the condition it depends on: one
    right record per cell (occupancy bounded at 1) laid out along a line keeps
    each left record's candidate set to at most 3 of its neighbourhood's nine
    cells (the other six never hold a populated column), so total comparisons
    stay proportional to record count rather than approaching the n^2 an
    all-pairs comparison would need. ADR-0001: property tests cannot establish
    asymptotic behaviour on their own, so this checks the bound directly at
    each drawn size instead of trusting a shape."""
    right = [
        _record("right", i, f"Listing {i}", latitude=float(i), longitude=0.0)
        for i in range(cell_count)
    ]
    left = [
        _record("left", i, f"Listing {i}", latitude=float(i), longitude=0.0)
        for i in range(cell_count)
    ]
    pairs = candidate_pairs(left, right, cell_size=1.0)
    # An upper bound alone cannot catch under-generation: a bug that
    # silently drops legitimate candidates still satisfies `<=`. Every right
    # record shares a left record's column, so a candidate set holds exactly
    # the right records in its own row plus the rows immediately above and
    # below - 3 per interior row, dropping to 2 at each end of the line
    # where a neighbouring row runs off it (or 1 when there is only one row
    # to begin with) - an exact count constrains both directions at once.
    expected = 1 if cell_count == 1 else 3 * cell_count - 2
    assert len(pairs) == expected


def test_comparison_count_degrades_toward_quadratic_in_a_dense_bucket() -> None:
    """The complexity claim's condition, shown failing on purpose (module
    docstring): every record at the same point puts all of them in one cell,
    so each of n left records is compared against all n right records."""
    n = 12
    right = [
        _record("right", i, f"Listing {i}", latitude=0.0, longitude=0.0)
        for i in range(n)
    ]
    left = [
        _record("left", i, f"Listing {i}", latitude=0.0, longitude=0.0)
        for i in range(n)
    ]
    assert len(candidate_pairs(left, right, cell_size=1.0)) == n * n


def test_haversine_km_rejects_a_record_missing_coordinates() -> None:
    with_coords = _record("left", 0, "Acme Traders", latitude=0.0, longitude=0.0)
    without_coords = _record("right", 0, "Acme Traders")
    with pytest.raises(ValueError, match="requires coordinates"):
        _haversine_km(with_coords, without_coords)


def test_haversine_km_matches_a_known_reference_distance() -> None:
    """Pins the formula's actual output, not just the ordering it produces:
    a relative-ordering test (near beats far) would still pass if the
    distance were miscomputed by a constant factor, since near would still
    come out smaller than far either way. One degree of longitude at the
    equator is a fixed, independently checkable distance."""
    origin = _record("left", 0, "Origin", latitude=0.0, longitude=0.0)
    one_degree_east = _record(
        "right", 0, "One Degree East", latitude=0.0, longitude=1.0
    )
    assert _haversine_km(origin, one_degree_east) == pytest.approx(111.195, abs=1e-3)


def test_haversine_km_matches_an_independently_computed_reference_distance() -> None:
    """The test above pins the formula's output at ``(0.0, 0.0)`` as point A,
    where swapping which field is read as latitude and which as longitude is
    a no-op - both are zero, so there is nothing to swap. Both points here
    have non-zero latitude and longitude, unequal to each other, so a field
    swap on either point changes the result. The reference value is computed
    independently with the spherical law of cosines - a different formula
    for the same great-circle distance (Wikipedia, "Great-circle distance")
    - rather than recomputed with this module's own haversine implementation,
    so a shared formula bug could not hide behind agreement between the two:
    ``R * acos(sin(lat1)*sin(lat2) + cos(lat1)*cos(lat2)*cos(lon2-lon1))``
    for round-number graticule points (10, 20) and (15, 25) gives
    776.861 km."""
    point_a = _record("left", 0, "Point A", latitude=10.0, longitude=20.0)
    point_b = _record("right", 0, "Point B", latitude=15.0, longitude=25.0)
    assert _haversine_km(point_a, point_b) == pytest.approx(776.861, abs=1e-3)


# --- Coordinate-less records are retained, not dropped (issue #27) -----------------


def test_a_coordinate_less_left_record_is_retained_as_unmatched_with_a_reason() -> None:
    left = [_record("left", 0, "No Coordinates Co")]
    right = [_record("right", 0, "Acme Traders", latitude=0.0, longitude=0.0)]
    result = resolve(left, right, _OVERLAP_MATCHER, cell_size=1.0)

    assert result.pairs == ()
    left_entry = next(u for u in result.unmatched if u.record is left[0])
    assert left_entry.reason == _REASON_NO_COORDINATES


def test_a_coordinate_less_right_record_is_retained_as_unmatched_with_a_reason() -> (
    None
):
    left = [_record("left", 0, "Acme Traders", latitude=0.0, longitude=0.0)]
    right = [_record("right", 0, "No Coordinates Co")]
    result = resolve(left, right, _OVERLAP_MATCHER, cell_size=1.0)

    assert result.pairs == ()
    right_entry = next(u for u in result.unmatched if u.record is right[0])
    assert right_entry.reason == _REASON_NO_COORDINATES


def test_a_left_record_missing_only_one_coordinate_is_treated_as_coordinate_less() -> (
    None
):
    """``latitude`` and ``longitude`` are independent ``Optional`` fields - a
    row with only one of the two set is exactly as unmatchable as one with
    neither."""
    left = [_record("left", 0, "Half Coordinates Co", latitude=0.0, longitude=None)]
    right = [_record("right", 0, "Half Coordinates Co", latitude=0.0, longitude=0.0)]
    result = resolve(left, right, _OVERLAP_MATCHER, cell_size=1.0)

    left_entry = next(u for u in result.unmatched if u.record is left[0])
    assert left_entry.reason == _REASON_NO_COORDINATES


def test_a_right_record_missing_only_one_coordinate_is_treated_as_coordinate_less() -> (
    None
):
    left = [_record("left", 0, "Half Coordinates Co", latitude=0.0, longitude=0.0)]
    right = [_record("right", 0, "Half Coordinates Co", latitude=None, longitude=0.0)]
    result = resolve(left, right, _OVERLAP_MATCHER, cell_size=1.0)

    right_entry = next(u for u in result.unmatched if u.record is right[0])
    assert right_entry.reason == _REASON_NO_COORDINATES


def test_a_left_record_with_no_nearby_candidate_is_unmatched_with_that_reason() -> None:
    left = [_record("left", 0, "Acme Traders", latitude=0.0, longitude=0.0)]
    right = [_record("right", 0, "Acme Traders", latitude=9.0, longitude=9.0)]
    result = resolve(left, right, _OVERLAP_MATCHER, cell_size=1.0)

    left_entry = next(u for u in result.unmatched if u.record is left[0])
    assert left_entry.reason == _REASON_NO_CANDIDATES


def test_a_left_record_with_a_candidate_below_threshold_is_unmatched_with_that_reason() -> (
    None
):
    left = [_record("left", 0, "Zzz Unrelated", latitude=0.0, longitude=0.0)]
    right = [_record("right", 0, "Acme Traders", latitude=0.0, longitude=0.0)]
    result = resolve(left, right, _OVERLAP_MATCHER, cell_size=1.0)

    left_entry = next(u for u in result.unmatched if u.record is left[0])
    assert left_entry.reason == _REASON_NO_MATCH_ABOVE_THRESHOLD


def test_a_right_record_never_chosen_by_any_left_record_is_unmatched_with_that_reason() -> (
    None
):
    left = [_record("left", 0, "Zzz Unrelated", latitude=0.0, longitude=0.0)]
    right = [_record("right", 0, "Acme Traders", latitude=0.0, longitude=0.0)]
    result = resolve(left, right, _OVERLAP_MATCHER, cell_size=1.0)

    right_entry = next(u for u in result.unmatched if u.record is right[0])
    assert right_entry.reason == _REASON_NOT_SELECTED


def test_every_input_record_is_traceable_in_the_result() -> None:
    left = [
        _record("left", 0, "Acme Traders", latitude=0.0, longitude=0.0),
        _record("left", 1, "No Coordinates Co"),
    ]
    right = [
        _record("right", 0, "Acme Traders", latitude=0.0, longitude=0.0),
        _record("right", 1, "Zzz Unrelated", latitude=9.0, longitude=9.0),
    ]
    result = resolve(left, right, _OVERLAP_MATCHER, cell_size=1.0)

    referenced = (
        {id(p.left) for p in result.pairs}
        | {id(p.right) for p in result.pairs}
        | {id(u.record) for u in result.unmatched}
    )
    assert referenced == {id(r) for r in left} | {id(r) for r in right}


# --- Merge policy (issue #28) --------------------------------------------------------


def test_merge_prefers_the_more_populated_record_on_disagreement() -> None:
    a = _record(
        "left",
        0,
        "Acme",
        fields={"category": "wholesale", "email": "a@example.test"},
        latitude=0.0,
        longitude=0.0,
    )
    b = _record(
        "right",
        0,
        "Acme Trading Co",
        fields={"category": "retail"},
        latitude=0.0,
        longitude=0.0,
    )
    merged = merge(a, b)
    assert merged.name == "Acme"
    assert merged.fields["category"] == "wholesale"


def test_merge_keeps_a_field_present_on_only_one_side_regardless_of_populatedness() -> (
    None
):
    a = _record(
        "left",
        0,
        "Acme",
        fields={"email": "a@example.test"},
        latitude=0.0,
        longitude=0.0,
    )
    b = _record(
        "right", 0, "Acme", fields={"category": "bakery"}, latitude=0.0, longitude=0.0
    )
    merged = merge(a, b)
    assert merged.fields["email"] == "a@example.test"
    assert merged.fields["category"] == "bakery"


def test_merge_ignores_empty_field_values_when_counting_populatedness() -> None:
    a = _record("left", 0, "", fields={"phone": ""})
    b = _record("right", 0, "Acme", fields={"phone": "+000 000 000 000"})
    merged = merge(a, b)
    assert merged.name == "Acme"
    assert merged.fields["phone"] == "+000 000 000 000"


def test_merge_takes_the_winners_coordinates_when_both_sides_have_them() -> None:
    a = _record(
        "left", 0, "Acme", fields={"x": "1", "y": "2"}, latitude=1.0, longitude=1.0
    )
    b = _record("right", 0, "Acme", latitude=9.0, longitude=9.0)
    merged = merge(a, b)
    assert (merged.latitude, merged.longitude) == (1.0, 1.0)


def test_merge_takes_coordinates_from_a_even_when_a_is_less_populated() -> None:
    a = _record("left", 0, "Acme", latitude=5.0, longitude=6.0)
    b = _record(
        "right",
        0,
        "Acme Trading Co",
        fields={"phone": "x", "email": "y", "category": "z"},
    )
    merged = merge(a, b)
    assert merged.name == "Acme Trading Co"
    assert (merged.latitude, merged.longitude) == (5.0, 6.0)


def test_merge_takes_coordinates_from_b_even_when_b_is_less_populated() -> None:
    a = _record(
        "left",
        0,
        "Acme Holdings Group",
        fields={"phone": "x", "email": "y", "category": "z"},
    )
    b = _record("right", 0, "Acme", latitude=1.0, longitude=2.0)
    merged = merge(a, b)
    assert merged.name == "Acme Holdings Group"
    assert (merged.latitude, merged.longitude) == (1.0, 2.0)


def test_merge_returns_no_coordinates_when_neither_side_has_them() -> None:
    a = _record("left", 0, "Acme")
    b = _record("right", 0, "Acme")
    merged = merge(a, b)
    assert merged.latitude is None
    assert merged.longitude is None


def test_merge_unions_provenance_from_both_sources() -> None:
    a = _record("registry", 0, "Acme")
    b = _record("listings", 0, "Acme")
    assert merge(a, b).sources == frozenset({"registry", "listings"})


def test_merge_is_order_independent() -> None:
    a = _record("left", 0, "Acme", fields={"phone": "1"}, latitude=1.0, longitude=1.0)
    b = _record(
        "right",
        0,
        "Acme Trading",
        fields={"category": "bakery"},
        latitude=2.0,
        longitude=2.0,
    )
    assert merge(a, b) == merge(b, a)


def test_populatedness_counts_a_non_empty_name() -> None:
    with_name = _record("left", 0, "Acme")
    without_name = _record("left", 1, "")
    assert _populatedness(with_name) == _populatedness(without_name) + 1


def test_populatedness_counts_carrying_coordinates() -> None:
    with_coords = _record("left", 0, "Acme", latitude=0.0, longitude=0.0)
    without_coords = _record("left", 1, "Acme")
    assert _populatedness(with_coords) == _populatedness(without_coords) + 1


def test_populatedness_ignores_empty_field_values() -> None:
    with_empty_extra = _record("left", 0, "Acme", fields={"phone": "x", "email": ""})
    without_the_empty_one = _record("left", 1, "Acme", fields={"phone": "x"})
    assert _populatedness(with_empty_extra) == _populatedness(without_the_empty_one)


def test_winner_tie_break_uses_the_lower_record_id_regardless_of_argument_order() -> (
    None
):
    a = _record("left", 0, "Acme")
    b = _record("right", 0, "Acme")
    expected = min((a, b), key=record_id)
    assert _winner(a, b) is expected
    assert _winner(b, a) is expected


# --- resolve() end to end (issue #29) ------------------------------------------------


def test_resolve_merges_two_record_sets_with_no_shared_key() -> None:
    left = [_record("left", 0, "Acme Traders", latitude=0.0, longitude=0.0)]
    right = [_record("right", 0, "Acme Traders Ltd", latitude=0.0, longitude=0.0)]
    result = resolve(left, right, _OVERLAP_MATCHER, cell_size=1.0)

    assert len(result.pairs) == 1
    assert result.pairs[0].left is left[0]
    assert result.pairs[0].right is right[0]
    assert result.unmatched == ()
    assert result.merged == (result.pairs[0].merged,)


def test_resolve_breaks_a_tied_score_by_distance() -> None:
    """RFC-0001: the distance tie-break is resolver policy, not the scorer's -
    two candidates that score identically are separated by which is closer,
    regardless of which one appears first in the input.

    Constructed so the closer candidate has the *larger* record_id: a
    tie-break that fell through to record_id without genuinely comparing
    distance would pick the farther candidate here instead, so this is the
    one arrangement that tells the two apart."""
    left_record = _record("left", 0, "Acme Traders", latitude=0.0, longitude=0.0)
    for offset in range(20):
        near = _record("right", offset, "Acme Traders", latitude=0.01, longitude=0.0)
        far = _record(
            "right", offset + 1_000, "Acme Traders", latitude=0.5, longitude=0.0
        )
        if record_id(near) > record_id(far):
            break
    else:
        pytest.fail("could not find a near/far pair with a contrary record_id order")

    result = resolve([left_record], [far, near], _OVERLAP_MATCHER, cell_size=1.0)
    assert result.pairs[0].right is near


def test_resolve_breaks_a_full_tie_by_the_lower_record_id_deterministically() -> None:
    left = [_record("left", 0, "Acme Traders", latitude=0.0, longitude=0.0)]
    right = [
        _record("right", 0, "Acme Traders", latitude=0.0, longitude=0.0),
        _record("right", 1, "Acme Traders", latitude=0.0, longitude=0.0),
    ]
    expected = min(right, key=record_id)

    result = resolve(left, right, _OVERLAP_MATCHER, cell_size=1.0)
    assert result.pairs[0].right is expected

    result_reversed = resolve(
        left, list(reversed(right)), _OVERLAP_MATCHER, cell_size=1.0
    )
    assert result_reversed.pairs[0].right is expected


def test_resolve_keeps_the_current_best_when_a_later_candidate_scores_lower() -> None:
    left = [_record("left", 0, "Acme Traders", latitude=0.0, longitude=0.0)]
    strong = _record("right", 0, "Acme Traders", latitude=0.0, longitude=0.0)
    weak = _record("right", 1, "Acme Zzz", latitude=0.0, longitude=0.0)
    result = resolve(left, [strong, weak], _OVERLAP_MATCHER, cell_size=1.0)
    assert result.pairs[0].right is strong


def test_two_left_records_may_both_match_the_same_right_record() -> None:
    """Documented design choice (module docstring): resolve() picks each left
    record's best candidate independently rather than running a global
    one-to-one assignment, so one right record can legitimately be the best
    match for more than one left record."""
    anchor = _record("right", 0, "Acme Plaza", latitude=0.0, longitude=0.0)
    left = [
        _record("left", 0, "Acme Plaza Unit A", latitude=0.0, longitude=0.0),
        _record("left", 1, "Acme Plaza Unit B", latitude=0.0, longitude=0.0),
    ]
    result = resolve(left, [anchor], _OVERLAP_MATCHER, cell_size=1.0)

    assert len(result.pairs) == 2
    assert all(pair.right is anchor for pair in result.pairs)
    assert result.unmatched == ()


def test_resolve_handles_two_empty_record_sets() -> None:
    result = resolve([], [], _OVERLAP_MATCHER, cell_size=1.0)
    assert result == ResolutionResult(pairs=(), unmatched=(), occupancy={})


def test_resolve_handles_an_empty_left_set() -> None:
    right = [_record("right", 0, "Acme Traders", latitude=0.0, longitude=0.0)]
    result = resolve([], right, _OVERLAP_MATCHER, cell_size=1.0)
    assert result.pairs == ()
    assert [u.record for u in result.unmatched] == right
    assert result.unmatched[0].reason == _REASON_NOT_SELECTED


def test_resolve_handles_an_empty_right_set() -> None:
    left = [_record("left", 0, "Acme Traders", latitude=0.0, longitude=0.0)]
    result = resolve(left, [], _OVERLAP_MATCHER, cell_size=1.0)
    assert [u.record for u in result.unmatched] == left
    assert result.unmatched[0].reason == _REASON_NO_CANDIDATES


def test_resolve_exposes_the_occupancy_it_used_for_candidate_generation() -> None:
    left = [_record("left", 0, "Acme Traders", latitude=0.0, longitude=0.0)]
    right = [
        _record("right", 0, "Acme Traders", latitude=0.0, longitude=0.0),
        _record("right", 1, "Acme Trading Co", latitude=0.4, longitude=0.4),
    ]
    result = resolve(left, right, _OVERLAP_MATCHER, cell_size=1.0)
    assert result.occupancy == bucket_occupancy(right, cell_size=1.0)


def test_resolution_result_merged_property_combines_pairs_and_unmatched() -> None:
    left = [_record("left", 0, "Acme Traders", latitude=0.0, longitude=0.0)]
    right = [
        _record("right", 0, "Acme Traders", latitude=0.0, longitude=0.0),
        _record("right", 1, "Zzz Unrelated", latitude=9.0, longitude=9.0),
    ]
    result = resolve(left, right, _OVERLAP_MATCHER, cell_size=1.0)
    assert result.merged == (result.pairs[0].merged, result.unmatched[0].record)


def test_resolve_uses_a_sensible_default_cell_size_when_none_is_given() -> None:
    left = [_record("left", 0, "Acme Traders", latitude=0.0, longitude=0.0)]
    right = [_record("right", 0, "Acme Traders", latitude=0.001, longitude=0.001)]
    result = resolve(left, right, _OVERLAP_MATCHER)
    assert len(result.pairs) == 1
    assert DEFAULT_CELL_SIZE_DEGREES == 0.01


class _PrepareAllSpy:
    """Wraps :class:`OverlapScorer` and counts calls to each preparation
    method, so :func:`resolve` can be checked against ADR-0009's contract
    directly: batched preparation (``prepare_all``) is the production call
    pattern, and the unbatched ``prepare`` is not the path resolve() takes.
    Delegating every real computation to the wrapped scorer keeps this a
    spy, not a stand-in for the scoring logic itself."""

    def __init__(self) -> None:
        self._inner = OverlapScorer()
        self.prepare_all_calls = 0
        self.prepare_calls = 0

    @property
    def name(self) -> str:
        return self._inner.name

    def prepare_all(self, names: Sequence[str | None]) -> list[frozenset[str]]:
        self.prepare_all_calls += 1
        return self._inner.prepare_all(names)

    def prepare(self, name: str | None) -> frozenset[str]:
        self.prepare_calls += 1
        return self._inner.prepare(name)

    def score(self, a: frozenset[str], b: frozenset[str]) -> float:
        return self._inner.score(a, b)


def test_resolve_uses_batched_preparation_not_the_per_record_path() -> None:
    """ADR-0009: prepare_all is the contractual production call pattern - a
    resolve() that instead called prepare() once per record inside the
    comparison loop would produce identical matches, so only a spy on the
    seam itself, not an assertion on the result, can tell the two apart."""
    spy = _PrepareAllSpy()
    left = [_record("left", 0, "Acme Traders", latitude=0.0, longitude=0.0)]
    right = [_record("right", 0, "Acme Traders", latitude=0.0, longitude=0.0)]
    matcher = ThresholdMatcher(scorer=spy, threshold=0.5)

    resolve(left, right, matcher, cell_size=1.0)

    assert spy.prepare_all_calls == 2  # once for left's names, once for right's
    assert spy.prepare_calls == 0


def test_swapping_the_scorer_changes_which_pairs_match_and_nothing_else() -> None:
    """RFC-0001's substitution invariant, pinned at the resolver: the scorer in
    use can change which pairs form, but candidate generation (the occupancy
    is identical either way) and the merge policy applied to whichever pairs
    do form are independent of it - and every record outside the one
    scorer-sensitive case ends up in the same place either way."""
    exact_left = _record("left", 0, "Acme Traders", latitude=0.0, longitude=0.0)
    exact_right = _record("right", 0, "Acme Traders", latitude=0.0, longitude=0.0)
    # Character noise: overlap sees zero shared tokens; fuzzy's character-aware
    # metric clears a high threshold easily (ADR-0008, test_scoring.py).
    noisy_left = _record("left", 1, "BRIGHTWATR", latitude=5.0, longitude=5.0)
    noisy_right = _record("right", 1, "BRIGHTWATER", latitude=5.0, longitude=5.0)
    # No scorer ever sees this one: it has no coordinates, so it can never
    # enter a candidate set (FR-3) regardless of which scorer is in use.
    # Issue #29 asks for "nothing else" to be checked too, not just the
    # pairs - this is the record that proves an unmatched entry's identity
    # and reason are as scorer-independent as the pairs are.
    untouchable_left = _record("left", 2, "No Coordinates Co")

    left = [exact_left, noisy_left, untouchable_left]
    right = [exact_right, noisy_right]
    threshold = 0.9

    overlap_result = resolve(
        left,
        right,
        ThresholdMatcher(scorer=OverlapScorer(), threshold=threshold),
        cell_size=1.0,
    )
    fuzzy_result = resolve(
        left,
        right,
        ThresholdMatcher(scorer=FuzzyScorer(), threshold=threshold),
        cell_size=1.0,
    )

    assert overlap_result.occupancy == fuzzy_result.occupancy

    exact_pair_overlap = next(p for p in overlap_result.pairs if p.left is exact_left)
    exact_pair_fuzzy = next(p for p in fuzzy_result.pairs if p.left is exact_left)
    assert exact_pair_overlap.right is exact_pair_fuzzy.right is exact_right
    assert exact_pair_overlap.merged == exact_pair_fuzzy.merged

    assert not any(p.left is noisy_left for p in overlap_result.pairs)
    assert any(p.left is noisy_left for p in fuzzy_result.pairs)

    untouchable_overlap = next(
        u for u in overlap_result.unmatched if u.record is untouchable_left
    )
    untouchable_fuzzy = next(
        u for u in fuzzy_result.unmatched if u.record is untouchable_left
    )
    assert (
        untouchable_overlap.reason == untouchable_fuzzy.reason == _REASON_NO_COORDINATES
    )


# --- Property tests for the resolver invariants (issue #30) ------------------------

_PROPERTY_NAMES = (
    "Acme Traders",
    "Riverside Bakery",
    "Highland Freight",
    "Baobab Holdings",
    "Grove Logistics",
)

_coordinate_component = st.floats(
    min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False
)

_optional_coordinates = st.one_of(
    st.tuples(_coordinate_component, _coordinate_component),
    st.tuples(_coordinate_component, _coordinate_component),
    st.just((None, None)),
)


def _records_strategy(source: str) -> st.SearchStrategy[list[Record]]:
    """Records for one source, mostly-scattered coordinates from a small
    invented vocabulary. The degenerate shapes issue #30 asks generators to
    cover - empty, no coordinates anywhere, one shared bucket, identical
    names - are pinned deterministically with ``@example`` on the tests
    below rather than left to chance."""
    row = st.tuples(st.sampled_from(_PROPERTY_NAMES), _optional_coordinates)

    def _build(
        rows: list[tuple[str, tuple[float, float] | tuple[None, None]]],
    ) -> list[Record]:
        return [
            Record(source=source, ordinal=i, name=name, latitude=lat, longitude=lon)
            for i, (name, (lat, lon)) in enumerate(rows)
        ]

    return st.lists(row, max_size=8).map(_build)


@given(left=_records_strategy("left"), right=_records_strategy("right"))
@example(left=[], right=[])
@example(
    left=[Record(source="left", ordinal=0, name="Acme Traders")],
    right=[Record(source="right", ordinal=0, name="Acme Traders")],
)
@example(
    left=[
        Record(
            source="left", ordinal=0, name="Acme Traders", latitude=0.0, longitude=0.0
        ),
        Record(
            source="left", ordinal=1, name="Acme Traders", latitude=0.0, longitude=0.0
        ),
    ],
    right=[
        Record(
            source="right", ordinal=0, name="Acme Traders", latitude=0.0, longitude=0.0
        ),
        Record(
            source="right", ordinal=1, name="Acme Traders", latitude=0.0, longitude=0.0
        ),
    ],
)
@example(
    left=[
        Record(
            source="left", ordinal=0, name="Acme Traders", latitude=0.0, longitude=0.0
        ),
        Record(
            source="left", ordinal=1, name="Acme Traders", latitude=10.0, longitude=10.0
        ),
    ],
    right=[
        Record(
            source="right", ordinal=0, name="Acme Traders", latitude=0.0, longitude=0.0
        ),
        Record(
            source="right",
            ordinal=1,
            name="Acme Traders",
            latitude=10.0,
            longitude=10.0,
        ),
    ],
)
def test_no_record_is_ever_dropped_from_the_result(
    left: list[Record], right: list[Record]
) -> None:
    """FR-3 / issue #27: every input record is traceable in the result, matched
    or not - as one side of a pair, or as an unmatched entry - across empty
    sets, records with no coordinates at all, every record sharing one
    bucket, and every record sharing one name."""
    result = resolve(left, right, _OVERLAP_MATCHER, cell_size=1.0)

    referenced = (
        {id(p.left) for p in result.pairs}
        | {id(p.right) for p in result.pairs}
        | {id(u.record) for u in result.unmatched}
    )
    assert referenced == {id(r) for r in left} | {id(r) for r in right}


_source_names = st.sampled_from(("registry", "listings", "left", "right"))
_identity_records = st.builds(
    Record,
    source=_source_names,
    ordinal=st.integers(min_value=0, max_value=1_000),
    name=st.text(max_size=40),
    latitude=st.one_of(st.none(), _coordinate_component),
    longitude=st.one_of(st.none(), _coordinate_component),
)


@given(record=_identity_records)
def test_record_id_is_stable_across_repeated_calls(record: Record) -> None:
    assert record_id(record) == record_id(record)


@given(
    source_a=_source_names,
    source_b=_source_names,
    ordinal=st.integers(min_value=0, max_value=1_000),
    name=st.text(max_size=40),
)
def test_record_id_never_collides_across_distinct_sources(
    source_a: str, source_b: str, ordinal: int, name: str
) -> None:
    assume(source_a != source_b)
    a = Record(source=source_a, ordinal=ordinal, name=name)
    b = Record(source=source_b, ordinal=ordinal, name=name)
    assert record_id(a) != record_id(b)


@given(
    source=_source_names,
    first_ordinal=st.integers(min_value=0, max_value=1_000),
    second_ordinal=st.integers(min_value=0, max_value=1_000),
    name=st.text(max_size=40),
)
def test_exact_duplicates_get_distinct_record_ids_but_share_a_content_id(
    source: str, first_ordinal: int, second_ordinal: int, name: str
) -> None:
    """CONTRIBUTING.md's identity invariant, "exact duplicates behaving as
    defined": two byte-identical rows differing only in their ordinal are
    distinct input rows (record_id disagrees) but the same content (content_id
    agrees) - which is what makes an exact duplicate visible rather than
    silently collapsed into one."""
    assume(first_ordinal != second_ordinal)
    first = Record(source=source, ordinal=first_ordinal, name=name)
    second = Record(source=source, ordinal=second_ordinal, name=name)
    assert record_id(first) != record_id(second)
    assert content_id(first) == content_id(second)
