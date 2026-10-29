"""Isolated EmbeddingGemma 2 encoder/search worker; invoked by dense_retrieval.py.

Use .venv-dense, not the JEV runtime. Inputs contain no relevance judgments.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

QUERY_PREFIX = 'task: search result | query: '


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def normalized(values, count):
    import numpy as np
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape != (count, 768) or not np.isfinite(values).all():
        raise ValueError('Expected finite 768-dimensional embeddings')
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if (norms <= 0).any():
        raise ValueError('Encoder returned zero-length embeddings')
    return values / norms


def run(request):
    import numpy as np
    import torch
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
    if 'embedding_gemma2' not in CONFIG_MAPPING_NAMES:
        raise RuntimeError(
            'This runtime cannot load EmbeddingGemma 2. Select --embedding-python '
            'with the environment versions recorded in docs/environment.md.')
    from sentence_transformers import SentenceTransformer
    from huggingface_hub import HfApi

    cache = Path(request['cache']).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    cache_model = cache / 'huggingface'
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    revision = request['revision']
    if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        revision = HfApi().model_info(request['model'], revision=revision).sha
    if not revision:
        raise ValueError('Hugging Face did not return an immutable model revision')
    device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu')
                          if request['device'] == 'auto' else request['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA embedding device requested but CUDA is unavailable')
    dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32
    spec = {'model': request['model'], 'revision': revision, 'dimensions': 768,
            'query_prefix': QUERY_PREFIX, 'document_format': 'title: {title or none} | text: {abstract}',
            'text_only': True, 'normalize': True, 'dtype': str(dtype),
            'device': str(device), 'batch_size': request['batch_size'],
            'transformers': importlib.metadata.version('transformers'),
            'sentence_transformers': importlib.metadata.version('sentence-transformers'),
            'torch': torch.__version__, 'max_seq_length': 8192}
    doc_texts = [f"title: {d.get('title') or 'none'} | text: {d['text']}" for d in request['documents']]
    query_texts = [QUERY_PREFIX + q['query'] for q in request['queries']]
    doc_key = fingerprint({'spec': spec, 'ids': [d['_id'] for d in request['documents']], 'texts': doc_texts})
    query_key = fingerprint({'spec': spec, 'ids': [q['query_id'] for q in request['queries']], 'texts': query_texts})
    model = None
    model_load_seconds = 0.0

    def encode_cached(key, texts):
        nonlocal model, model_load_seconds
        folder = cache / key
        folder.mkdir(exist_ok=True)
        path, meta = folder/'embeddings.npy', folder/'metadata.json'
        if path.exists() and meta.exists():
            saved = json.loads(meta.read_text())
            if saved['key'] != key or saved['sha256'] != file_hash(path):
                raise ValueError(f'Embedding cache checksum mismatch: {folder}')
            vectors = np.load(path, allow_pickle=False)
            normalized(vectors, len(texts))  # Validate, but preserve exact cached floats.
            return vectors, True
        if model is None:
            load_started = time.perf_counter()
            model = SentenceTransformer(request['model'], revision=revision, device=str(device),
                cache_folder=str(cache_model), config_kwargs={'vision_config': None, 'audio_config': None},
                model_kwargs={'torch_dtype': dtype}, trust_remote_code=False)
            model.max_seq_length = 8192
            model_load_seconds += time.perf_counter() - load_started
        vectors = normalized(model.encode(texts, batch_size=request['batch_size'],
            prompt='', normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=True), len(texts))
        temporary = folder/'embeddings.npy.tmp'
        with temporary.open('wb') as output:
            np.save(output, vectors, allow_pickle=False)
        temporary.replace(path)
        atomic_json(meta, {'key': key, 'sha256': file_hash(path), 'spec': spec, 'count': len(texts)})
        return vectors, False

    start = time.perf_counter()
    documents, doc_hit = encode_cached(doc_key, doc_texts)
    index_seconds = time.perf_counter()-start
    start = time.perf_counter()
    previous_load_seconds = model_load_seconds
    queries, query_hit = encode_cached(query_key, query_texts)
    query_phase_load_seconds = model_load_seconds - previous_load_seconds
    query_embedding_seconds = time.perf_counter()-start-query_phase_load_seconds
    index_seconds += query_phase_load_seconds
    candidates = []
    doc_ids = [d['_id'] for d in request['documents']]
    for query, embedding in zip(request['queries'], queries):
        start = time.perf_counter()
        scores = documents @ embedding
        order = sorted(range(len(doc_ids)), key=lambda i: (-float(scores[i]), doc_ids[i]))[:request['limit']]
        candidates.append({'query_id': query['query_id'], 'query': query['query'],
            'candidates': [{'doc_id': doc_ids[i], 'dense_score': float(scores[i])} for i in order],
            'retrieval_ms': (time.perf_counter()-start)*1000 + query_embedding_seconds*1000/len(queries)})
    return {'candidates': candidates, 'index_seconds': index_seconds,
        'identity': {'type': 'dense', **spec, 'requested_revision': request['requested_revision'],
                     'document_embeddings_key': doc_key, 'query_embeddings_key': query_key},
        'cache': {'documents_hit': doc_hit, 'queries_hit': query_hit},
        'model_load_seconds': model_load_seconds,
        'timing_note': 'Query embedding time is amortized over the batch. Cache hits measure cached-vector loading and search, not fresh encoding.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--response', type=Path, required=True)
    args = parser.parse_args()
    result = run(json.loads(args.request.read_text()))
    atomic_json(args.response, result)


if __name__ == '__main__':
    main()
