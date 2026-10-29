"""Embedding retrieval in a separate Python process, released before JEV starts."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = 'google/embeddinggemma-2'
DEFAULT_PYTHON = ROOT/'.venv-dense/bin/python'
DEFAULT_CACHE = ROOT/'data/downloads/dense'


def retrieve_dense(documents, queries, limit, *, python=DEFAULT_PYTHON,
                   cache=DEFAULT_CACHE, device='auto', batch_size=8,
                   revision='main', requested_revision='main'):
    """Encode/search with Google's search prefixes and full 768d cosine similarity.

    No labels reach the subprocess. Its exit releases all embedding model memory.
    Corpus and query embeddings have independent content/revision/version caches.
    """
    python = Path(python).absolute()  # Preserve the venv executable symlink.
    if not python.is_file():
        raise FileNotFoundError(
            'Dense Python runtime missing. Pass --embedding-python /path/to/python '
            'for the separate environment described in docs/environment.md.')
    cache = Path(cache).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    payload = {'model': DEFAULT_MODEL, 'revision': revision, 'requested_revision': requested_revision,
        'documents': [{key: document.get(key, '') for key in ('_id', 'title', 'text')}
                      for document in documents],
        'queries': [{'query_id': q['query_id'], 'query': q['query']} for q in queries],
        'limit': limit, 'device': device, 'batch_size': batch_size, 'cache': str(cache)}
    with tempfile.TemporaryDirectory(prefix='request-', dir=cache) as temporary:
        folder = Path(temporary)
        (folder/'request.json').write_text(json.dumps(payload))
        # Avoid inheriting project import paths into the independently versioned runtime.
        environment = dict(os.environ)
        environment.pop('PYTHONPATH', None)
        environment['HF_HOME'] = str(cache/'huggingface-home')
        result = subprocess.run([str(python), str(Path(__file__).with_name('dense_worker.py')),
            '--request', str(folder/'request.json'), '--response', str(folder/'response.json')],
            env=environment, check=False)
        if result.returncode:
            raise RuntimeError(f'Dense embedding worker exited with status {result.returncode}; see its error above')
        return json.loads((folder/'response.json').read_text())
