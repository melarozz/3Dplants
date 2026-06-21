#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT/venv}"

SKIP_TESTS=0
SKIP_GENERATION=0
BUILD_WHEEL=0
EXTRAS="dev"

usage() {
  cat <<'EOF'
3Dplants build script

Options:
  --skip-tests
  --skip-generation
  --wheel
  --venv PATH
  --python PATH
EOF
}

log() { printf "==> %s\n" "$*"; }
die() { printf "error: %s\n" "$*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-tests) SKIP_TESTS=1 ;;
    --skip-generation) SKIP_GENERATION=1 ;;
    --wheel) BUILD_WHEEL=1 ;;
    --venv)
      shift; [[ $# -gt 0 ]] || die "--venv requires path"
      VENV_DIR="$1"
      ;;
    --python)
      shift; [[ $# -gt 0 ]] || die "--python requires path"
      PYTHON="$1"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
  shift
done

if [[ "$SKIP_GENERATION" -eq 0 ]]; then
  EXTRAS="dev,generation"
fi

command -v "$PYTHON" >/dev/null 2>&1 || die "Python not found: $PYTHON"

"$PYTHON" -c 'import sys; exit(0 if sys.version_info >= (3,10) else 1)' \
  || die "Python >= 3.10 required"

if [[ ! -d "$VENV_DIR" ]]; then
  log "Creating venv: $VENV_DIR"
  "$PYTHON" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

log "Upgrading pip"
python -m pip install -U pip setuptools wheel

log "Installing project [$EXTRAS]"
python -m pip install -e ".[$EXTRAS]"

if [[ "$BUILD_WHEEL" -eq 1 ]]; then
  log "Building wheel"
  python -m pip install build
  rm -rf dist build *.egg-info src/*.egg-info
  python -m build --outdir dist
  ls -la dist/
fi

log "Verifying CLI entry points"

for cmd in plant3d plant-cpd plant-light plant-segment; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    die "Missing CLI: $cmd"
  fi

  # Safe smoke test:
  if ! "$cmd" --help >/dev/null 2>&1; then
    log "  WARN: $cmd does not support --help cleanly"
  else
    log "  $cmd OK"
  fi
done

if [[ "$SKIP_TESTS" -eq 0 ]]; then
  log "Running tests"
  python -m pytest tests/ -q --tb=short --no-cov
else
  log "Skipping tests"
fi

cat <<EOF

Build complete.

Activate:
  source "$VENV_DIR/bin/activate"

EOF