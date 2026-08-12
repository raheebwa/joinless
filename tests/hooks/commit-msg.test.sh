#!/usr/bin/env bash
# Test for the commit-msg hook. No dependencies beyond bash and git.
# Run: bash tests/hooks/commit-msg.test.sh
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || exit 1

HOOK=.githooks/commit-msg
pass=0
fail=0
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# expect <accept|reject> <label> <message...>
expect() {
  local want="$1" label="$2"
  shift 2
  printf '%s\n' "$@" > "$tmp/msg"
  if bash "$HOOK" "$tmp/msg" >/dev/null 2>&1; then got=accept; else got=reject; fi
  if [ "$got" = "$want" ]; then
    pass=$((pass + 1))
  else
    fail=$((fail + 1))
    printf '  FAIL  expected %s, got %s: %s\n' "$want" "$got" "$label"
  fi
}

SIGN="Signed-off-by: A Name <a@example.com>"

expect accept "conventional with scope and sign-off" \
  "feat(packaging): add pyproject.toml manifest" "" "$SIGN"
expect accept "conventional without scope" \
  "docs: record the installable surface" "" "$SIGN"
expect reject "unknown type" \
  "wibble(cli): do a thing" "" "$SIGN"
expect reject "uppercase type" \
  "Feat(cli): add entry point" "" "$SIGN"
expect reject "uppercase scope" \
  "feat(CLI): add entry point" "" "$SIGN"
expect reject "missing sign-off" \
  "feat(cli): add entry point"
expect reject "subject 51 chars" \
  "feat(packaging): xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" "" "$SIGN"
expect accept "subject exactly 50 chars" \
  "feat(packaging): xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" "" "$SIGN"
expect reject "trailer other than sign-off" \
  "feat(cli): add entry point" "" "$SIGN" "Reviewed-by: Someone <s@example.com>"
expect accept "merge commit bypasses the format rule" \
  "Merge pull request #12 from raheebwa/feat/thing"
expect reject "no colon separator" \
  "feat cli add entry point" "" "$SIGN"
expect reject "empty subject after colon" \
  "feat(cli):" "" "$SIGN"

printf '\n%d passed, %d failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ]
