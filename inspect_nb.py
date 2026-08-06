import json, sys, io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

nb = json.loads(Path('notebooks/deconv_sanity_check.ipynb').read_text(encoding='utf-8'))
for i, c in enumerate(nb['cells']):
    src = ''.join(c['source'])
    lines = src.strip().split('\n')
    cell_type = c['cell_type']
    print(f'--- Cell {i} ({cell_type}) ---')
    for line in lines[:6]:
        print(f'  {repr(line)}')
    if len(lines) > 6:
        print(f'  ... ({len(lines)} lines total)')
    print()
