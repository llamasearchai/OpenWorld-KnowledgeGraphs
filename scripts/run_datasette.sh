#!/usr/bin/env bash
set -euo pipefail
DB_PATH="${1:-db/owkg.db}"
datasette "$DB_PATH" -p 8010 -h 127.0.0.1
