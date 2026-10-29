"""Shared Open-Jev process helpers, preserving the existing suite GPU settings.

The helper implementations match run_openjev_dataset_experiments.py. That runner
is intentionally left unchanged; this module supports the new retrieval runner.
"""
import errno
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]

def project_path(value):
    """Resolve value, a path string, relative to this project unless absolute."""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path

def server_command(config, model):
    """Build argv from config server settings and one model entry; no shell eval."""
    server = {**config['server'], **model.get('server', {})}
    command = [sys.executable, '-m', 'jev.server', '--checkpoint', str(project_path(model['checkpoint'])),
               '--device', server['device'], '--max-length', str(server['max_length']),
               '--batch-size', str(server['batch_size']), '--host', server['host'], '--port', str(server['port'])]
    return command + server.get('extra_args', [])

def server_environment(config, model):
    """Build isolated subprocess environment with configured server overrides.

    Args:
        config: Suite config; server.env maps environment names to string values
            (or None to remove a variable). These values are saved in config;
            keep credentials in the parent environment, not this mapping.
        model: One model entry. Its server.env overrides shared server.env.
            The consumer-GPU placement/cache variables are cleared from inherited
            state first, so exported 9B settings cannot leak into another model.

    Returns:
        A new environment dictionary for the server only; os.environ is unchanged.
    """
    environment = dict(os.environ)
    for name in ('JEV_DEVICE_MAP', 'JEV_PREFILL_CHUNK', 'JEV_RAGGED_SUFFIX'):
        environment.pop(name, None)
    overrides = {**config['server'].get('env', {}), **model.get('server', {}).get('env', {})}
    for name, value in overrides.items():
        if value is None:
            environment.pop(name, None)
        elif isinstance(value, str):
            environment[name] = value
        else:
            raise ValueError(f'Server environment {name} must be a string or null')
    # Accelerate initializes a broad parent's hook with place_submodules=True.
    # That can materialize the CPU embedding on GPU before its child offload hook
    # is attached. Map only GPU-only subtrees, avoiding the overlapping parent.
    value = environment.get('JEV_DEVICE_MAP', '')
    if value.lstrip().startswith('{'):
        placement = json.loads(value)
        parent = 'model.language_model'
        if parent in placement and placement.get(parent + '.embed_tokens') == 'cpu':
            device = placement.pop(parent)
            for child in ('layers', 'norm', 'rotary_emb'):
                placement.setdefault(parent + '.' + child, device)
            environment['JEV_DEVICE_MAP'] = json.dumps(placement, separators=(',', ':'))
    return environment

def health(config, model):
    """Fetch health for config/model server address; caller handles startup errors."""
    server = {**config['server'], **model.get('server', {})}
    with urlopen(f"http://{server['host']}:{server['port']}/health", timeout=3) as response:
        result = json.load(response)
    if result.get('status') != 'ready':
        raise ValueError('Server not ready')
    if model.get('expected_model') and result.get('model') != model['expected_model']:
        raise ValueError(f"Wrong checkpoint identity: {result.get('model')}")
    return result

def wait_for_server_port(host, port, timeout=30):
    """Wait until the HTTP server can bind its address, ignoring stale TIME_WAIT.

    Args:
        host: Bind address of the next server.
        port: TCP port number.
        timeout: Maximum seconds to retry EADDRINUSE after shutdown. No existing
            listener is terminated. SO_REUSEADDR mirrors Python HTTPServer;
            listen() ensures an active server cannot silently share the address.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind((host, int(port)))
                probe.listen(1)
            return
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise
            if time.monotonic() >= deadline:
                raise RuntimeError(f'Port {host}:{port} still occupied after {timeout}s; '
                                   'stop the existing listener or choose another port') from exc
            time.sleep(.25)

def server_log_error(path):
    """Summarize path, a server log, so startup failures include their cause."""
    lines=Path(path).read_text(errors='replace').splitlines()
    errors=[line.strip() for line in lines if 'Error:' in line or 'Exception:' in line]
    return (errors[-1] if errors else 'no Python exception found') + f' (log: {path})'

def stop_process(process, timeout=30):
    """Stop the process group owned by process; timeout is TERM grace seconds."""
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        pass
    # Also remove any surviving children after their process-group leader exits.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()
