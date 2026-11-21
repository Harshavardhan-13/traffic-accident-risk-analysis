#!/usr/bin/env python3
"""Execute and save all five notebooks with this environment's Python kernel."""
from pathlib import Path
import argparse
import os
import sys
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
ROOT = Path(__file__).resolve().parents[1]
os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
preview = ROOT / '.notebook_previews'
preview.mkdir(exist_ok=True)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--from-notebook', type=int, default=1, choices=range(1, 6))
args = parser.parse_args()
for path in sorted((ROOT / 'notebooks').glob('*.ipynb')):
    if int(path.name[:2]) < args.from_notebook:
        continue
    print(f'Executing {path.name}', flush=True)
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(notebook, timeout=900, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}})
    client.execute()
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    html, _ = HTMLExporter().from_notebook_node(notebook)
    (preview / (path.stem + '.html')).write_text(html)
    print(f'Saved executed notebook and HTML preview: {path.name}', flush=True)
