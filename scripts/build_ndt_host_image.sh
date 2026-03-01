#!/usr/bin/env bash
set -euo pipefail

# Build the Docker image used by DockerHost nodes.
#
# If you see:
#   Docker image not found locally: ndt/host:focal-nettools
# run this script once, then re-run the experiment.

IMG=${IMG:-ndt/host:focal-nettools}

echo "Building $IMG ..."
docker build -t "$IMG" -f Dockerfile.ndt_host .
echo "OK"
