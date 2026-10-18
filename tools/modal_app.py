"""Modal orchestration for the experiments.

    modal run modal_app.py::prep
    modal run --detach modal_app.py::train --name bc38f_s0 --script train/boostcut/run.py \
        --ngpu 8 --args "--seed 0"
    modal run modal_app.py::py --script analysis/test_boostcut.py --args "..."

Data lives on the `noise-forward` volume under /data/data, runs under /data/runs/<name>, logs under /data/logs.
Every job also leaves a provenance record under /data/provenance/<name>/ (see _provenance_start).
"""
from datetime import datetime, timezone
import hashlib
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
APP_NAME = 'noise-forward'
app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version='3.12')
    .apt_install('build-essential')
    .pip_install('torch==2.10.0', index_url='https://download.pytorch.org/whl/cu128')
    .pip_install('numpy', 'tokenizers==0.23.1', 'datasets==4.8.4')
    .env({'PYTHONUNBUFFERED': '1'})
    .add_local_dir(HERE / 'noise', '/root/noise', ignore=['data', 'logs', '**/__pycache__', '**/*.pt'])
)
vol = modal.Volume.from_name('noise-forward', create_if_missing=True)
DATA = '/data/data'
PROVENANCE, CODE, DATA_DIRS = '/data/provenance', '/root/noise', ('/data/data', '/data/tinystories')
MODAL_ENV = ('MODAL_TASK_ID', 'MODAL_REGION', 'MODAL_CLOUD_PROVIDER', 'MODAL_IMAGE_ID', 'MODAL_ENVIRONMENT')  # no keys


def _sha256(path, block=1 << 20):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(block), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _provenance_start(name, cmd):
    """What this job ran, for the ledgers: a snapshot (code.tar.gz) and sha256 manifest of the exact code in the
    container, the data files' sha256, the command, the GPUs and the software versions. Never fails the job."""
    try:
        folder = Path(PROVENANCE) / name
        folder.mkdir(parents=True, exist_ok=True)
        files = sorted(p for p in Path(CODE).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
        manifest = {str(p.relative_to(CODE)): _sha256(p) for p in files}
        with tarfile.open(folder / 'code.tar.gz', 'w:gz') as tar:
            for p in files:
                tar.add(p, arcname=f'noise/{p.relative_to(CODE)}')
        gpus = []
        if shutil.which('nvidia-smi'):
            query = ['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader']
            gpus = subprocess.run(query, capture_output=True, text=True).stdout.strip().splitlines()
        probe = 'import sys, torch; print(torch.__version__, torch.version.cuda, sys.version.split()[0])'
        versions = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True).stdout.split()
        torch_version, cuda_version, python_version = (versions + ['?'] * 3)[:3]
        record = dict(
            name=name, command=shlex.join(cmd), start_utc=_now(),
            code_sha256=hashlib.sha256(''.join(f'{h}  {f}\n' for f, h in manifest.items()).encode()).hexdigest(),
            code_files=manifest,
            data_files={str(p): _sha256(p) for d in DATA_DIRS if Path(d).exists()
                        for p in sorted(Path(d).glob('*.pt'))},
            gpus=gpus, torch=torch_version, cuda=cuda_version, python=python_version,
            hostname=socket.gethostname(),
            modal={k: os.environ[k] for k in MODAL_ENV if k in os.environ},
        )
        (folder / 'job.json').write_text(json.dumps(record, indent=2) + '\n')
        return folder, record
    except Exception as e:  # provenance must never cost a run
        print(f'[modal_app] provenance failed: {e}', flush=True)
        return None


def _provenance_end(started, code):
    if started is None:
        return
    try:
        folder, record = started
        record.update(end_utc=_now(), exit_code=code)
        (folder / 'job.json').write_text(json.dumps(record, indent=2) + '\n')
    except Exception as e:
        print(f'[modal_app] provenance failed: {e}', flush=True)


def _committer(stop, every=45):
    while not stop.wait(every):
        try:
            vol.commit()
        except Exception as e:  # a failed background commit must not kill the run
            print(f'[modal_app] commit failed: {e}', flush=True)


def _stream(cmd, log_path, cwd='/root/noise', env=None):
    """Run cmd, tee its output into log_path on the volume, committing the volume periodically."""
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    provenance = _provenance_start(Path(log_path).stem, cmd)
    stop = threading.Event()
    thread = threading.Thread(target=_committer, args=(stop,), daemon=True)
    thread.start()
    started = time.time()
    with open(log_path, 'a') as log:
        log.write(f'$ {shlex.join(cmd)}\n')
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                env={**os.environ, **(env or {})})
        for line in proc.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        code = proc.wait()
        log.write(f'[modal_app] exit {code} after {time.time() - started:.0f}s\n')
    stop.set()
    _provenance_end(provenance, code)
    vol.commit()
    return code


@app.function(image=image, volumes={'/data': vol}, timeout=4 * 3600, cpu=8, memory=32768)
def prepare_data():
    os.makedirs(DATA, exist_ok=True)
    if all(os.path.exists(f'{DATA}/{n}') for n in ('train.pt', 'val.pt')):
        code = _stream(['python', 'utils/prepare_data.py', '--output', DATA, '--verify'], '/data/logs/prep.log')
        if code == 0:
            return 'already prepared'
    code = _stream(['python', 'utils/prepare_data.py', '--output', DATA], '/data/logs/prep.log')
    if code:
        raise RuntimeError(f'prepare_data exited {code}')
    return 'prepared'


@app.function(image=image, volumes={'/data': vol}, gpu='H100', timeout=24 * 3600, memory=65536)
def run_train(name, script, args, ngpu):
    out = f'/data/runs/{name}'
    cmd = ['torchrun', '--standalone', f'--nproc_per_node={ngpu}', script, '--data-dir', DATA, '--output', out]
    cmd += shlex.split(args)
    code = _stream(cmd, f'/data/logs/{name}.log')
    if code:
        raise RuntimeError(f'{name} exited {code}')
    return Path(f'{out}/result.json').read_text() if Path(f'{out}/result.json').exists() else 'no result'


@app.function(image=image, volumes={'/data': vol}, gpu='H100', timeout=12 * 3600, memory=65536)
def run_py(name, script, args):
    cmd = ['python', script, '--data-dir', DATA] + shlex.split(args)
    code = _stream(cmd, f'/data/logs/{name}.log')
    if code:
        raise RuntimeError(f'{name} exited {code}')
    return 'ok'


@app.function(image=image, volumes={'/data': vol}, timeout=6 * 3600, cpu=8, memory=32768)
def run_cpu(name, script, args):
    cmd = ['python', script, '--data-dir', DATA] + shlex.split(args)
    code = _stream(cmd, f'/data/logs/{name}.log')
    if code:
        raise RuntimeError(f'{name} exited {code}')
    return 'ok'


@app.local_entrypoint()
def prep():
    print(prepare_data.remote())


@app.local_entrypoint()
def train(name: str, script: str = 'train/boostcut/run.py', args: str = '', ngpu: int = 1, gpu: str = 'H100'):
    fn = run_train.with_options(gpu=f'{gpu}:{ngpu}' if ngpu > 1 else gpu)
    print(fn.remote(name, script, args, ngpu))


@app.local_entrypoint()
def py(name: str, script: str, args: str = '', gpu: str = 'H100'):
    print(run_py.with_options(gpu=gpu).remote(name, script, args))


@app.local_entrypoint()
def cpu(name: str, script: str, args: str = ''):
    print(run_cpu.remote(name, script, args))
