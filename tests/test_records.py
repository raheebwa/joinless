# SPDX-License-Identifier: MIT
"""Record identity: stable, non-colliding, and defined for exact duplicates."""

from joinless.records import Record, content_id, record_id


def test_identical_input_yields_the_same_id_on_every_run() -> None:
    a = Record(source="registry", ordinal=0, name="Acme Trading Co")
    b = Record(source="registry", ordinal=0, name="Acme Trading Co")

    assert record_id(a) == record_id(b)


def test_two_same_named_rows_without_coordinates_get_distinct_ids() -> None:
    """The failure PRD FR-4 exists to prevent: identity from name and coordinates
    alone collapses these two into one, discarding a row the resolver just went to
    trouble to retain."""
    first = Record(source="registry", ordinal=0, name="Nile Valley Traders")
    second = Record(source="registry", ordinal=1, name="Nile Valley Traders")

    assert record_id(first) != record_id(second)


def test_byte_identical_rows_are_separated_by_their_ordinal() -> None:
    first = Record(source="registry", ordinal=0, name="Riverside Bakery")
    second = Record(source="registry", ordinal=1, name="Riverside Bakery")

    assert record_id(first) != record_id(second)
    # ...and are still recognisable as the same content, which is what makes an
    # exact duplicate visible rather than silent.
    assert content_id(first) == content_id(second)


def test_ids_cannot_collide_across_sources() -> None:
    left = Record(source="registry", ordinal=0, name="Falcon Freight Services")
    right = Record(source="listings", ordinal=0, name="Falcon Freight Services")

    assert record_id(left) != record_id(right)
    assert content_id(left) != content_id(right)


def test_field_insertion_order_does_not_change_identity() -> None:
    """A dict iterates in insertion order, so two records with equal content built
    in different orders would serialise differently without a canonical form."""
    one = Record(source="registry", ordinal=0, name="Acme", fields={"a": "1", "b": "2"})
    two = Record(source="registry", ordinal=0, name="Acme", fields={"b": "2", "a": "1"})

    assert record_id(one) == record_id(two)


def test_field_values_cannot_be_rearranged_into_another_records_id() -> None:
    """Concatenating parts without a separator lets ("ab", "c") and ("a", "bc")
    hash alike. The separator is a null byte, which text input cannot contain."""
    one = Record(source="registry", ordinal=0, name="ab")
    two = Record(source="registrya", ordinal=0, name="b")

    assert record_id(one) != record_id(two)
