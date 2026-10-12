#!/usr/bin/env bash
set -euo pipefail
REV="${1:?Pass exact published revision}"
LANGUAGE="${2:-en}"
[[ "$REV" =~ ^[0-9a-f]{40}$ && "$LANGUAGE" =~ ^(en|pl)$ ]] || { echo 'Invalid revision/language' >&2; exit 1; }
ROOT="$HOME/ai-v100"
FOLDER="$ROOT/tools/wiki-import/$REV"
mkdir -p "$FOLDER"
for MODULE in wikimedia_exports knowledge_rag; do
  curl -fL --retry 3 "https://raw.githubusercontent.com/Mantiee/rlm/$REV/rlm/v100/$MODULE.py" -o "$FOLDER/$MODULE.py.new"
  "$ROOT/venvs/v100-continual/bin/python" -m py_compile "$FOLDER/$MODULE.py.new"
  mv "$FOLDER/$MODULE.py.new" "$FOLDER/$MODULE.py"
done
exec "$ROOT/venvs/v100-continual/bin/python" -u "$FOLDER/wikimedia_exports.py" --root "$ROOT" --language "$LANGUAGE"
