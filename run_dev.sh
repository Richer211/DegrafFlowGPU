#!/bin/bash
set -e

DOCKER_BIN="$(command -v docker || true)"
if [ -z "$DOCKER_BIN" ] && [ -x /Applications/Docker.app/Contents/Resources/bin/docker ]; then
  DOCKER_BIN="/Applications/Docker.app/Contents/Resources/bin/docker"
fi

if [ -z "$DOCKER_BIN" ]; then
  echo "Error: docker command not found."
  echo "Please install Docker Desktop CLI or add docker to PATH."
  exit 1
fi

"$DOCKER_BIN" run --platform linux/amd64 -it --rm -v "$PWD:/app" -w /app degraf bash
