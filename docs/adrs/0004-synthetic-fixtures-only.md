# ADR-0004 — Synthetic fixtures only; no real-world corpus

**Status:** Accepted · **Date:** 2026-08-12

## Context

Entity resolution needs data to run and labelled data to evaluate. The obvious source is
a real corpus — public registers, open-data extracts, scraped listings. Real data would
make the accuracy numbers more externally valid.

It would also mean this repository contains records about real people and real
businesses, under source licences that vary and often forbid redistribution, with privacy
obligations attached, in a repository that is public and permanently archived by third
parties.

## Decision

**All fixtures are synthetic and invented.** No real-world corpus enters this repository
in any form — not as data files, not as fixtures, not as examples in documentation, not
as test constants.

Fixture design deliberately includes the hard cases real data exhibits: abbreviations,
business-suffix noise, word-order variation, transliteration variants, coordinate-less
records, and near-miss negatives that *should not* match.

No fixture uses a real business name, a real person's name, a real address, a real phone
number, or coordinates that resolve to a real premises. Phone numbers use the
`+000 000 000 000` form rather than a locale-plausible number — many countries reserve no
fake range, so a plausible-looking number can belong to someone.

## Consequences

- The repository can be public, permissively licensed, and redistributed with no data
  licence or privacy exposure. Nobody has to audit what is in it.
- Accuracy results are **directional, not universal**. They characterise matcher
  behaviour on a controlled distribution with deliberately constructed hard cases. The
  README must say so rather than implying the numbers transfer unchanged to any corpus.
- Fixture design becomes a real engineering task with real influence on the result. A set
  built only of easy cases would flatter the classical matcher; one built only of
  transliterations would flatter the neural arm. The composition is documented and
  justified, and is itself reviewable.
- Latency, memory and model-size results are unaffected by this decision — they depend on
  string length distribution and comparison count, not on whether the strings are real.

## Alternatives rejected

**Use a public open-data extract.** Licence terms vary by source and frequently forbid
redistribution. Verifying that for every record is more work than building good fixtures,
and gets it wrong permanently if missed once.

**Ship a downloader instead of data.** Moves the licence problem to the user, makes the
benchmark non-reproducible when the upstream source changes or disappears, and breaks the
no-network requirement.
