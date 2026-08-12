#!/usr/bin/env bash
# Checks applied to staged content by the pre-commit hook.
# Usage: checks.sh <check-name|all> <path>...
# Each check is a pure function of the paths it is given, so it can be tested
# against fixtures without a repository.
set -uo pipefail

note() { printf '  %s\n' "$1" >&2; }

check_spdx() {
  local rc=0 f
  for f in "$@"; do
    case "$f" in *.py) ;; *) continue ;; esac
    [ -f "$f" ] || continue
    head -n 5 "$f" | grep -qxF '# SPDX-License-Identifier: MIT' \
      || { note "$f: no '# SPDX-License-Identifier: MIT' in the first five lines"; rc=1; }
  done
  return $rc
}

check_license_classifier() {
  local rc=0 f
  for f in "$@"; do
    case "$f" in *pyproject.toml) ;; *) continue ;; esac
    [ -f "$f" ] || continue
    if grep -q 'License ::' "$f"; then
      note "$f: a 'License ::' classifier contradicts the PEP 639 licence expression"; rc=1
    fi
  done
  return $rc
}

check_workflow() {
  local rc=0 f line
  for f in "$@"; do
    case "$f" in .github/workflows/*.yml|.github/workflows/*.yaml) ;; *) continue ;; esac
    [ -f "$f" ] || continue
    grep -qE '^[[:space:]]*permissions:' "$f" \
      || { note "$f: no 'permissions:' block; a workflow must declare its scope"; rc=1; }
    while IFS= read -r line; do
      printf '%s' "$line" | grep -qE 'uses:[[:space:]]*[^@[:space:]]+@[0-9a-f]{40}([[:space:]]|$)' \
        || { note "$f: action not pinned to a 40-character SHA -> ${line#*:}"; rc=1; }
    done < <(grep -E '^[[:space:]]*-?[[:space:]]*uses:' "$f")
  done
  return $rc
}

check_changelog() {
  local rc=0 f line
  for f in "$@"; do
    case "$f" in *CHANGELOG.md) ;; *) continue ;; esac
    [ -f "$f" ] || continue
    while IFS= read -r line; do
      printf '%s' "$line" | grep -qE '^## \[[0-9]+\.[0-9]+\.[0-9]+\] - [0-9]{4}-[0-9]{2}-[0-9]{2}$' \
        || { note "$f: version heading must be '## [x.y.z] - YYYY-MM-DD' -> $line"; rc=1; }
    done < <(grep -E '^## \[[0-9]' "$f")
  done
  return $rc
}

check_no_artifacts() {
  local rc=0 f
  for f in "$@"; do
    case "$f" in
      *.onnx|*.safetensors|*.gguf|models/*|*/models/*)
        note "$f: model artefacts are fetched at setup, never committed"; rc=1 ;;
    esac
  done
  return $rc
}

check_pii() {
  local rc=0 f hits
  for f in "$@"; do
    case "$f" in *.onnx|*.safetensors|*.gguf|*.png|*.jpg|*.pdf) continue ;; esac
    [ -f "$f" ] || continue
    hits="$(grep -oE '\+[0-9][0-9 ()-]{7,}[0-9]' "$f" 2>/dev/null | grep -v '^+000' || true)"
    if [ -n "$hits" ]; then
      note "$f: use the +000 000 000 000 placeholder; a plausible number can route to a real person -> $(printf '%s' "$hits" | head -n1)"
      rc=1
    fi
  done
  return $rc
}

ALL='check_spdx check_license_classifier check_workflow check_changelog check_no_artifacts check_pii'

cmd="${1:-}"; shift || true
case "$cmd" in
  all) rc=0; for c in $ALL; do "$c" "$@" || rc=1; done; exit $rc ;;
  check_*) case " $ALL " in *" $cmd "*) "$cmd" "$@"; exit $? ;; esac
           printf 'unknown check: %s\n' "$cmd" >&2; exit 2 ;;
  *) printf 'usage: checks.sh <all|%s> <path>...\n' "${ALL// /|}" >&2; exit 2 ;;
esac
