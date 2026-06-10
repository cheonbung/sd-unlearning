#!/usr/bin/env bash
set -euo pipefail

REQUIRED_MAJOR=20
NODESOURCE_MAJOR=20

log() {
  printf '[setup_node] %s\n' "$*"
}

have_node_v20_or_newer() {
  if ! command -v node >/dev/null 2>&1; then
    return 1
  fi

  local version major
  version="$(node -v 2>/dev/null || true)"
  major="${version#v}"
  major="${major%%.*}"

  [[ "$major" =~ ^[0-9]+$ ]] && (( major >= REQUIRED_MAJOR ))
}

have_npm() {
  command -v npm >/dev/null 2>&1
}

if have_node_v20_or_newer && have_npm; then
  log "Node.js $(node -v) and npm $(npm -v) are already installed."
  exit 0
fi

if [[ "$(uname -s)" != "Linux" ]]; then
  log "This installer is intended for WSL/Ubuntu Linux."
  exit 1
fi

if ! command -v apt-get >/dev/null 2>&1; then
  log "apt-get was not found. Please run this on Ubuntu or another apt-based WSL distro."
  exit 1
fi

if [[ "${EUID}" -eq 0 ]]; then
  SUDO=()
else
  if ! command -v sudo >/dev/null 2>&1; then
    log "sudo is required when not running as root."
    exit 1
  fi
  SUDO=(sudo)
fi

log "Installing prerequisites..."
"${SUDO[@]}" apt-get update
"${SUDO[@]}" apt-get install -y ca-certificates curl gnupg

log "Configuring NodeSource Node.js ${NODESOURCE_MAJOR}.x repository..."
curl -fsSL "https://deb.nodesource.com/setup_${NODESOURCE_MAJOR}.x" | "${SUDO[@]}" -E bash -

log "Installing Node.js..."
"${SUDO[@]}" apt-get install -y nodejs

if ! have_node_v20_or_newer; then
  log "Node.js installation failed or version is below v${REQUIRED_MAJOR}: $(node -v 2>/dev/null || echo 'not found')"
  exit 1
fi

if ! have_npm; then
  log "npm was not installed with Node.js."
  exit 1
fi

log "Done: Node.js $(node -v), npm $(npm -v)"
