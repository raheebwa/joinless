#!/usr/bin/env bash
# Tests for the staged-content checks the pre-commit hook applies.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || exit 1
LIB=.githooks/checks.sh
pass=0; fail=0
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

# expect <ok|bad> <check-fn> <label> ; file content on stdin, path in $3
try() {
  local want="$1" fn="$2" label="$3" path="$4"
  mkdir -p "$tmp/$(dirname "$path")"; cat > "$tmp/$path"
  if ( cd "$tmp" && bash "$OLDPWD/$LIB" "$fn" "$path" ) >/dev/null 2>&1; then got=ok; else got=bad; fi
  if [ "$got" = "$want" ]; then pass=$((pass+1)); else
    fail=$((fail+1)); printf '  FAIL  %s: wanted %s got %s\n' "$label" "$want" "$got"; fi
}

try ok  check_spdx "py with spdx" "a.py" <<<'# SPDX-License-Identifier: MIT
x = 1'
try bad check_spdx "py without spdx" "b.py" <<<'x = 1'
try bad check_spdx "spdx below line 5" "c.py" <<<'1
2
3
4
5
# SPDX-License-Identifier: MIT'

try ok  check_license_classifier "manifest without classifier" "pyproject.toml" <<<'license = "MIT"'
try bad check_license_classifier "manifest with classifier" "pyproject.toml" <<<'classifiers = ["License :: OSI Approved :: MIT License"]'

try ok  check_workflow ".github/workflows/ok.yml" ".github/workflows/ok.yml" <<<'permissions:
  contents: read
jobs:
  a:
    steps:
      - uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683'
try bad check_workflow "unpinned action" ".github/workflows/bad.yml" <<<'permissions:
  contents: read
jobs:
  a:
    steps:
      - uses: actions/checkout@v4'
try bad check_workflow "no permissions block" ".github/workflows/np.yml" <<<'jobs:
  a:
    steps:
      - uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683'

try ok  check_changelog "iso date" "CHANGELOG.md" <<<'## [0.1.0] - 2026-08-12'
try bad check_changelog "non-iso date" "CHANGELOG.md" <<<'## [0.1.0] - 12/08/2026'
try ok  check_changelog "unreleased only" "CHANGELOG.md" <<<'## [Unreleased]'

try bad check_no_artifacts "onnx artifact" "model.onnx" <<<'binary'
try ok  check_no_artifacts "ordinary file" "notes.md" <<<'text'

try ok  check_pii "placeholder number" "d.md" <<<'Call +000 000 000 000 for support.'
# Assembled at runtime: writing a locale-plausible number as a literal would
# put one in a tracked file, which is the very thing check_pii exists to stop.
cc="+"; plausible="${cc}256 772 123 456"
try bad check_pii "locale-plausible number" "e.md" <<<"Call $plausible for support."

printf '\n%d passed, %d failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ]
