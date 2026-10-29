"""Download official BEIR archives and index relevance judgments by query.

Run with ``python -m jev_guardrails.beir_data --datasets all``. No optional
packages are needed. Missing qrels remain unjudged, never synthetic negatives.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import shutil
import stat
import tempfile
import time
from urllib.request import Request, urlopen
import zipfile

CATALOG = json.loads(Path(__file__).with_name('beir_catalog.json').read_text())
RANGE_CHUNK_BYTES = 16 * 1024 * 1024
PARALLEL_MIN_BYTES = 64 * 1024 * 1024
DEFAULT_ROOT = Path(__file__).resolve().parents[1] / 'data/downloads/beir'


def write_json(path, value):
    """Atomically write a JSON status/manifest file."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def digest(path, algorithm='sha256'):
    """Hash a file without loading it into memory."""
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _download_ranges(url, part):
    """Fetch large archives in four ranges at a time, reusing a contiguous prefix.

    Return False when the server does not advertise range support. Completed
    chunks survive interruption; only complete verified-size chunks are merged.
    The caller still verifies the checksum of the entire resulting archive.
    """
    try:
        with urlopen(Request(url, method='HEAD'), timeout=30) as response:
            total = int(response.headers.get('Content-Length', 0))
            supported = response.headers.get('Accept-Ranges') == 'bytes'
    except (OSError, ValueError):
        return False
    offset = part.stat().st_size if part.exists() else 0
    if total and offset == total:
        return True
    if not supported or total < PARALLEL_MIN_BYTES:
        return False
    if offset > total:
        raise ValueError('Partial archive is larger than the source')
    cache = part.with_suffix('.ranges')
    cache.mkdir(exist_ok=True)

    def fetch(bounds):
        start, end = bounds
        target = cache / f'{start}-{end}'
        size = end - start + 1
        if target.exists() and target.stat().st_size == size:
            return target
        for attempt in range(4):
            try:
                request = Request(url, headers={
                    'User-Agent': 'jev-guardrails-beir/1', 'Range': f'bytes={start}-{end}'})
                with urlopen(request, timeout=120) as response:
                    expected = f'bytes {start}-{end}/{total}'
                    if response.status != 206 or response.headers.get('Content-Range') != expected:
                        raise ValueError('Server did not honor requested archive range')
                    temporary = target.with_suffix('.part')
                    with temporary.open('wb') as output:
                        remaining = size
                        while remaining:
                            block = response.read(min(1024 * 1024, remaining))
                            if not block:
                                raise OSError('Incomplete archive range')
                            output.write(block)
                            remaining -= len(block)
                    temporary.replace(target)
                return target
            except (OSError, ValueError):
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)

    bounds = [(start, min(start + RANGE_CHUNK_BYTES, total) - 1)
              for start in range(offset, total, RANGE_CHUNK_BYTES)]
    # Consume in order; file names retain completed chunks across retries.
    with ThreadPoolExecutor(max_workers=4) as pool:
        with part.open('ab') as output:
            for chunk in pool.map(fetch, bounds):
                # A partial append remains a valid contiguous prefix after interruption.
                with chunk.open('rb') as source:
                    shutil.copyfileobj(source, output, 1024 * 1024)
                output.flush()
    return True


def download_archive(name, root):
    """Resume interrupted HTTP downloads and verify the publisher's MD5."""
    spec = CATALOG[name]
    cache = root / '.archives'
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f'{name}.zip'
    part = target.with_suffix('.zip.part')
    if not target.exists() and _download_ranges(spec['url'], part):
        if digest(part, 'md5') != spec['md5']:
            part.unlink()
            shutil.rmtree(part.with_suffix('.ranges'), ignore_errors=True)
            raise ValueError(f'Publisher MD5 mismatch: {name}')
        part.replace(target)
        shutil.rmtree(part.with_suffix('.ranges'), ignore_errors=True)
    if not target.exists():
        for attempt in range(4):
            try:
                offset = part.stat().st_size if part.exists() else 0
                headers = {'User-Agent': 'jev-guardrails-beir/1'}
                if offset:
                    headers['Range'] = f'bytes={offset}-'
                with urlopen(Request(spec['url'], headers=headers), timeout=120) as response:
                    append = offset and response.status == 206
                    if append and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                        raise ValueError('Unexpected HTTP range response')
                    expected = response.headers.get('Content-Length')
                    received = 0
                    with part.open('ab' if append else 'wb') as output:
                        for block in iter(lambda: response.read(4 * 1024 * 1024), b''):
                            output.write(block)
                            received += len(block)
                    if expected is not None and received != int(expected):
                        raise OSError('Incomplete HTTP response; partial download retained')
                if digest(part, 'md5') != spec['md5']:
                    part.unlink()
                    raise ValueError(f'Publisher MD5 mismatch: {name}')
                part.replace(target)
                break
            except (OSError, ValueError):
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
    if digest(target, 'md5') != spec['md5']:
        raise ValueError(f'Cached archive checksum mismatch: {target}')
    return target


def extract_archive(archive, destination):
    """Extract only after rejecting traversal paths and symbolic links."""
    base = destination.resolve()
    with zipfile.ZipFile(archive) as zipped:
        for item in zipped.infolist():
            target = (base / item.filename).resolve()
            if not target.is_relative_to(base) or stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError(f'Unsafe archive member: {item.filename}')
        zipped.extractall(base)


def _insert_jsonl(connection, path, kind):
    """Index document byte offsets or query texts in bounded batches."""
    count, batch = 0, []
    sql = ('INSERT INTO documents VALUES (?, ?)' if kind == 'documents'
           else 'INSERT INTO queries VALUES (?, ?)')
    with path.open('rb') as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            row = json.loads(line)
            key = row['_id']
            if not isinstance(key, str) or not isinstance(row['text'], str):
                raise ValueError(f'Invalid ID/text in {path}')
            batch.append((key, offset if kind == 'documents' else row['text']))
            count += 1
            if len(batch) == 10000:
                connection.executemany(sql, batch)
                batch.clear()
        connection.executemany(sql, batch)
    return count


def prepare_collection(folder):
    """Audit qrels references; create SQLite and per-query JSONL mappings.

    Leaves original corpus.jsonl, queries.jsonl and qrels/*.tsv untouched.
    Document text stays in the corpus; byte offsets allow fast ID-based access.
    Only queries belonging to a split's qrels appear in its mapped output.
    """
    folder = Path(folder)
    qrel_files = sorted((folder / 'qrels').glob('*.tsv'))
    if not qrel_files:
        raise ValueError(f'No qrels found: {folder}')
    index = folder / 'index.sqlite'
    if index.exists():
        raise FileExistsError(index)
    with sqlite3.connect(index) as db:
        db.executescript('''
            PRAGMA journal_mode=OFF;
            CREATE TABLE documents (id TEXT PRIMARY KEY, offset INTEGER NOT NULL);
            CREATE TABLE queries (id TEXT PRIMARY KEY, text TEXT NOT NULL);
            CREATE TABLE judgments (
                split TEXT, query_id TEXT, doc_id TEXT, score INTEGER NOT NULL,
                PRIMARY KEY(split, query_id, doc_id));
        ''')
        documents = _insert_jsonl(db, folder / 'corpus.jsonl', 'documents')
        queries = _insert_jsonl(db, folder / 'queries.jsonl', 'queries')
        splits = {}
        for path in qrel_files:
            count, batch = 0, []
            with path.open(encoding='utf-8', newline='') as stream:
                reader = csv.DictReader(stream, delimiter='\t')
                if reader.fieldnames != ['query-id', 'corpus-id', 'score']:
                    raise ValueError(f'Unexpected qrels header: {path}')
                for row in reader:
                    score = float(row['score'])
                    if not math.isfinite(score) or not score.is_integer():
                        raise ValueError(f'Invalid relevance grade in {path}: {score}')
                    batch.append((path.stem, row['query-id'], row['corpus-id'], int(score)))
                    count += 1
                    if len(batch) == 10000:
                        db.executemany('INSERT INTO judgments VALUES (?, ?, ?, ?)', batch)
                        batch.clear()
                db.executemany('INSERT INTO judgments VALUES (?, ?, ?, ?)', batch)
            missing = db.execute('''SELECT j.query_id, j.doc_id FROM judgments j
                LEFT JOIN queries q ON q.id=j.query_id
                LEFT JOIN documents d ON d.id=j.doc_id
                WHERE j.split=? AND q.id IS NULL LIMIT 1''',
                (path.stem,)).fetchone()
            if missing:
                raise ValueError(f'Dangling query/document reference in {path}: {missing}')
            missing_documents = db.execute('''SELECT COUNT(*) FROM judgments j
                LEFT JOIN documents d ON d.id=j.doc_id
                WHERE j.split=? AND d.id IS NULL''', (path.stem,)).fetchone()[0]
            mapped = folder / 'mapped' / f'{path.stem}.jsonl'
            mapped.parent.mkdir(exist_ok=True)
            query_count, current, judgments, text = 0, None, [], None
            with mapped.open('w', encoding='utf-8') as output:
                for qid, query, docid, score, available in db.execute('''
                    SELECT j.query_id, q.text, j.doc_id, j.score, d.id IS NOT NULL FROM judgments j
                    JOIN queries q ON q.id=j.query_id
                    LEFT JOIN documents d ON d.id=j.doc_id WHERE j.split=?
                    ORDER BY j.query_id, j.doc_id''', (path.stem,)):
                    if current is not None and qid != current:
                        output.write(json.dumps({'query_id': current, 'query': text,
                                                 'judgments': judgments}) + '\n')
                        query_count += 1
                        judgments = []
                    current, text = qid, query
                    judgment = {'doc_id': docid, 'score': score}
                    if not available:
                        judgment['document_available'] = False
                    judgments.append(judgment)
                if current is not None:
                    output.write(json.dumps({'query_id': current, 'query': text,
                                             'judgments': judgments}) + '\n')
                    query_count += 1
            splits[path.stem] = {'queries': query_count, 'judgments': count,
                                'missing_document_judgments': missing_documents,
                                'grades': [r[0] for r in db.execute(
                                    'SELECT DISTINCT score FROM judgments WHERE split=? ORDER BY score',
                                    (path.stem,))]}
    return {'documents': documents, 'queries': queries, 'splits': splits}


def download_one(name, root):
    """Download and atomically publish one collection (12 for CQADupstack)."""
    root = Path(root)
    destination = root / name
    if (destination / 'manifest.json').exists():
        manifest = json.loads((destination / 'manifest.json').read_text())
        if manifest.get('status') != 'complete' or manifest.get('source') != CATALOG[name]:
            raise ValueError(f'Existing manifest does not match catalog: {destination}')
        for relative, collection in manifest['collections'].items():
            folder = destination / relative
            required = [folder / item for item in ('corpus.jsonl', 'queries.jsonl', 'index.sqlite')]
            required += [folder / 'mapped' / f'{split}.jsonl' for split in collection['splits']]
            required += [folder / 'qrels' / f'{split}.tsv' for split in collection['splits']]
            if any(not path.is_file() for path in required):
                raise ValueError(f'Completed dataset has missing files: {destination}')
        return manifest
    if destination.exists():
        raise FileExistsError(f'Incomplete/unrecognized destination: {destination}')
    if not CATALOG[name]['url']:
        return {'status': 'access_required', 'instructions': CATALOG[name]['access_instructions']}
    print(f'{name}: downloading/verifying archive', flush=True)
    archive = download_archive(name, root)
    with tempfile.TemporaryDirectory(prefix=f'.{name}-', dir=root) as temporary:
        unpacked = Path(temporary)
        extract_archive(archive, unpacked)
        publish = unpacked / name if (unpacked / name).is_dir() else unpacked
        collections = sorted(p.parent for p in publish.rglob('corpus.jsonl'))
        if not collections:
            raise ValueError(f'No BEIR collections in {archive}')
        print(f'{name}: indexing {len(collections)} collection(s) and mapping labels', flush=True)
        summary = {}
        for folder in collections:
            summary[folder.relative_to(publish).as_posix()] = prepare_collection(folder)
        # Keep the archive directory hierarchy, including CQADupstack subforums.
        manifest = {'status': 'complete', 'dataset': name, 'source': CATALOG[name],
                    'archive_sha256': digest(archive),
                    'completed_at': datetime.now(timezone.utc).isoformat(),
                    'collections': summary,
                    'missing_label_policy': 'unjudged; no synthetic negative labels'}
        write_json(publish / 'manifest.json', manifest)
        publish.rename(destination)
    print(f'{name}: complete', flush=True)
    return manifest


class BeirCollection:
    """Read prepared query mappings and resolve document IDs without full loading."""
    def __init__(self, folder):
        self.folder = Path(folder).resolve()

    def iter_queries(self, split='test'):
        """Yield query text and original document relevance grades for this split."""
        with (self.folder / 'mapped' / f'{split}.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                yield json.loads(line)

    def get_query(self, query_id, split='test'):
        """Fetch one labeled query and its judgments directly from the index."""
        with closing(sqlite3.connect(
                f'{(self.folder / "index.sqlite").as_uri()}?mode=ro', uri=True)) as db:
            query = db.execute('SELECT text FROM queries WHERE id=?', (query_id,)).fetchone()
            rows = db.execute("""SELECT j.doc_id, j.score, d.id IS NOT NULL
                FROM judgments j LEFT JOIN documents d ON d.id=j.doc_id
                WHERE j.query_id=? AND j.split=? ORDER BY j.doc_id""",
                (query_id, split)).fetchall()
        if query is None or not rows:
            raise KeyError((split, query_id))
        judgments = []
        for docid, score, available in rows:
            judgment = {'doc_id': docid, 'score': score}
            if not available:
                judgment['document_available'] = False
            judgments.append(judgment)
        return {'query_id': query_id, 'query': query[0], 'judgments': judgments}

    def get_document(self, doc_id):
        """Return the original document record, using its indexed byte offset."""
        with closing(sqlite3.connect(
                f'{(self.folder / "index.sqlite").as_uri()}?mode=ro', uri=True)) as db:
            row = db.execute('SELECT offset FROM documents WHERE id=?', (doc_id,)).fetchone()
        if row is None:
            raise KeyError(doc_id)
        with (self.folder / 'corpus.jsonl').open('rb') as stream:
            stream.seek(row[0])
            return json.loads(stream.readline())


def main(argv=None):
    """Download selected datasets; report restricted/failed entries explicitly."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', default=['all'], choices=['all', *CATALOG])
    parser.add_argument('--output', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args(argv)
    if args.list:
        for name, spec in CATALOG.items():
            print(name, spec['url'] or f"ACCESS REQUIRED: {spec['access_instructions']}")
        return 0
    if args.workers < 1:
        parser.error('--workers must be positive')
    names = list(CATALOG) if 'all' in args.datasets else list(dict.fromkeys(args.datasets))
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    status_path = args.output / 'download_status.json'
    statuses = json.loads(status_path.read_text()) if status_path.exists() else {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, name, args.output): name for name in names}
        for future in as_completed(futures):
            name = futures[future]
            try:
                statuses[name] = future.result()
            except Exception as error:
                statuses[name] = {'status': 'failed', 'error': str(error)}
                print(f'{name}: FAILED: {error}', flush=True)
            write_json(status_path, statuses)
    counts = {state: sum(statuses[n]['status'] == state for n in names)
              for state in ('complete', 'access_required', 'failed')}
    print(json.dumps(counts), flush=True)
    return 1 if counts['failed'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
