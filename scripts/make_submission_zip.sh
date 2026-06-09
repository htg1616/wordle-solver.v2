#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p dist
rm -f dist/team00_code.zip dist/team00_problem.json
cp team00_problem.json dist/team00_problem.json
python - <<'PY'
import zipfile
from pathlib import Path
files = [Path('team00.py'), Path('requirements.txt')]
out = Path('dist/team00_code.zip')
with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for path in files:
        zf.write(path, arcname=path.name)
print(f'wrote {out} ({out.stat().st_size} bytes)')
PY
python - <<'PY'
import json
from pathlib import Path
p = Path('dist/team00_problem.json')
data = json.loads(p.read_text(encoding='utf-8'))
assert isinstance(data.get('secret_word'), str)
assert isinstance(data.get('candidate_words'), list)
assert data['secret_word'] in set(data['candidate_words'])
assert all(isinstance(w, str) and len(w) == 5 and w.isalpha() and w.islower() for w in data['candidate_words'])
print(f'validated {p}')
PY
