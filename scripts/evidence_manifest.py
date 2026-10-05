#!/usr/bin/env python3
"""Write an exact-path SHA256 evidence inventory, without copying large logs."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('output', type=Path)
parser.add_argument('paths', nargs='+', type=Path)
a = parser.parse_args()
files = []
for p in a.paths:
    files.extend(sorted(x for x in p.rglob('*') if x.is_file()) if p.is_dir() else [p])
a.output.write_text(json.dumps([{'path': str(p.resolve()), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                                'bytes': p.stat().st_size} for p in files], indent=2)+'\n')
