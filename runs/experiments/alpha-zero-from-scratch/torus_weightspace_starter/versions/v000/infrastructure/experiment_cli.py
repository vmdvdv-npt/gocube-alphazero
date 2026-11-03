"""Astra's explicit zero/positions/games/Arena interface. No API calls or training."""
from __future__ import annotations
import argparse
from contextlib import redirect_stdout
import fcntl
import io
import json
from pathlib import Path
import runpy
import shlex
import sys
import subprocess

import torch
from model import TorusNet
from weights import (load_arrays, store_arrays, tensor_schema, validate, state_digest,
                     save_checkpoint, restore_checkpoint, sha256, write_json)


def run_writer(path):
    model = TorusNet().eval().requires_grad_(False)
    if path is None:
        return model, ''
    # Guard accidental forward calls through the supplied model and every layer.
    # This is a trusted Python interface, not a security sandbox.
    def no_forward(*args):
        raise RuntimeError('Forward calls are forbidden during zero-pass')
    handles = [module.register_forward_pre_hook(no_forward) for module in model.modules()]
    log = io.StringIO()
    try:
        with redirect_stdout(log), torch.inference_mode():
            writer = runpy.run_path(str(path))['program']
            arrays = load_arrays(model)
            writer(arrays)
            store_arrays(model, arrays)
    finally:
        for handle in handles:
            handle.remove()
    validate(model)
    return model, log.getvalue()


def zero_pass(output, writer=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    frozen = None
    if writer:
        frozen = output/'writer.py'
        frozen.write_bytes(Path(writer).read_bytes())
    sources = output/'infrastructure'
    sources.mkdir()
    for source in Path(__file__).parent.glob('*.py'):
        (sources/source.name).write_bytes(source.read_bytes())
    model, log = run_writer(frozen)
    repeat, _ = run_writer(frozen)
    if state_digest(model) != state_digest(repeat):
        raise ValueError('Writer is not reproducible across two invocations')
    genesis = save_checkpoint(TorusNet(), output/'genesis.pt')
    meta = save_checkpoint(model, output/'checkpoint.pt', design='programmed' if writer else 'zeros',
                           writer_sha256=sha256(frozen) if frozen else None)
    restored, _ = restore_checkpoint(output/'checkpoint.pt')
    if state_digest(restored) != state_digest(model):
        raise ValueError('Roundtrip mismatch')
    (output/'writer.stdout.txt').write_text(log)
    report = {'status': 'PASS', 'mode': 'zero-pass', 'forward_calls': 0,
              'reproducible': True, 'checkpoint': str((output/'checkpoint.pt').resolve()),
              'metadata': meta, 'genesis_sha256': genesis['artifact_sha256']}
    write_json(output/'zero-pass.json', report)
    return report


def init_session(path, budget):
    path.mkdir(parents=True, exist_ok=False)
    write_json(path/'session.json', {'schema': 'weightspace-session-v1', 'budget': budget})
    (path/'rounds').mkdir()
    return {'session': str(path.resolve()), 'budget': budget}


def diagnostic(args):
    # Serialize reservation and execution so concurrent requests cannot overspend.
    session = args.session.resolve()
    with (session/'session.json').open() as mutex:
        fcntl.flock(mutex, fcntl.LOCK_EX)
        config = json.load(mutex)
        if config['budget'] not in (0,10,25,50):
            raise ValueError('Invalid session budget')
        rounds = session/'rounds'
        used = len(list(rounds.iterdir()))
        if used >= config['budget']:
            raise ValueError('Diagnostic budget exhausted; zero budget exposes no diagnostics')
        model, meta = restore_checkpoint(args.checkpoint)
        zero = json.loads((args.checkpoint.parent/'zero-pass.json').read_text())
        if zero['status'] != 'PASS' or zero['metadata']['artifact_sha256'] != meta['artifact_sha256']:
            raise ValueError('Successful zero-pass is required before diagnostics')
        request = vars(args).copy()
        request = {key: str(value) if isinstance(value, Path) else value for key,value in request.items()}
        folder = rounds/f'{used+1:03d}'
        folder.mkdir(exist_ok=False)
        write_json(folder/'request.json', dict(request, checkpoint_sha256=meta['artifact_sha256'],
            repository_revision=subprocess.check_output(['git','rev-parse','HEAD'], cwd=Path(__file__).resolve().parents[4], text=True).strip(),
            command=shlex.join([sys.executable, *sys.argv]), cwd=str(Path.cwd()),
            source_sha256={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}))
        try:
            import diagnostics as d
            if args.mode == 'positions':
                positions_input = json.loads(args.positions.read_text())
                write_json(folder/'positions-input.json', positions_input)
                report = d.positions(model, positions_input, komi=args.komi,
                                     top_k=args.top_k, activations=args.activations)
            else:
                openings = json.loads(args.openings.read_text()) if args.openings else [[]]
                write_json(folder/'openings-input.json', openings)
                if args.mode == 'arena':
                    reference, lock = d.load_reference(args.reference_lock)
                    write_json(folder/'reference-lock.json', lock)
                    settings = lock['search']
                    report = d.paired_games(model, komi=lock['komi'], openings=openings,
                        opponent='reference', reference=reference, simulations=settings['simulations'],
                        cpuct=settings['cpuct'], fpu=settings['fpu'], tree_reuse=settings['tree_reuse'],
                        watchdog=settings['watchdog'], seed=args.seed)
                else:
                    report = d.paired_games(model, komi=args.komi, openings=openings,
                                            opponent=args.opponent, seed=args.seed, watchdog=args.watchdog)
            report.update(mode=args.mode, round=used+1, budget=config['budget'],
                          checkpoint_sha256=meta['artifact_sha256'])
            write_json(folder/'report.json', report)
        except Exception as error:
            write_json(folder/'error.json', {'status': 'ERROR', 'error': str(error), 'round_consumed': True})
            raise
        return {'report': str(folder/'report.json'), 'round': used+1, 'budget': config['budget'],
                'status': report.get('status','PASS')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--threads', type=int, default=1)
    commands = p.add_subparsers(dest='mode', required=True)
    commands.add_parser('schema')
    init = commands.add_parser('init')
    init.add_argument('--session', type=Path, required=True)
    init.add_argument('--budget', type=int, choices=(0,10,25,50), required=True)
    zero = commands.add_parser('zero-pass')
    zero.add_argument('--output', type=Path, required=True)
    zero.add_argument('--writer', type=Path)
    verify = commands.add_parser('verify')
    verify.add_argument('--checkpoint', type=Path, required=True)
    for mode in ('positions','games','arena'):
        cmd = commands.add_parser(mode)
        cmd.add_argument('--session', type=Path, required=True)
        cmd.add_argument('--checkpoint', type=Path, required=True)
        if mode != 'arena':
            cmd.add_argument('--komi', type=float, required=True)
        if mode == 'positions':
            cmd.add_argument('--positions', type=Path, required=True)
            cmd.add_argument('--top-k', type=int, default=5)
            cmd.add_argument('--activations', action='store_true')
        else:
            cmd.add_argument('--openings', type=Path)
            cmd.add_argument('--seed', type=int, default=20261103)
            if mode == 'arena':
                cmd.add_argument('--reference-lock', type=Path, required=True)
            else:
                cmd.add_argument('--opponent', choices=('random','pass'), default='random')
                cmd.add_argument('--watchdog', type=int, default=1000)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    if args.mode == 'schema':
        report = {'tensors': tensor_schema(), 'parameter_count': 396009}
    elif args.mode == 'init':
        report = init_session(args.session, args.budget)
    elif args.mode == 'zero-pass':
        report = zero_pass(args.output, args.writer)
        write_json(args.output/'reproduce.json', {'command': shlex.join([sys.executable, *sys.argv]),
                                                 'cwd': str(Path.cwd()),
                                                 'source_sha256': {p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}})
    elif args.mode == 'verify':
        model, meta = restore_checkpoint(args.checkpoint)
        report = dict(validate(model), state_sha256=meta['state_sha256'])
    else:
        report = diagnostic(args)
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'status': 'ERROR', 'error_type': type(error).__name__, 'error': str(error)}), file=sys.stderr)
        sys.exit(1)
