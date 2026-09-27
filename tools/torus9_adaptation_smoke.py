"""Bounded real CUDA self-play -> targets -> update -> reload -> arena acceptance."""
from pathlib import Path
import argparse
import json
import torch
from gocube_golden.torus9_adaptation import AdaptationTrainer, selfplay, game_targets, validate_game
from gocube_golden.orchestrator_v2.adaptation import BOOTSTRAP, arena_statistics
from gocube_golden.process_supervision import atomic_write_json
from tools.arena_engine import ArenaExecutionConfig, run_arena
from tools.arena_profiles import get_profile


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    trainer = AdaptationTrainer(BOOTSTRAP, device='cuda')
    initial = out / 'initial.pt'
    trainer.save(initial, replay_ids=[], config_hash='smoke-only')
    result = selfplay(trainer.model, checkpoint=initial, run_id='adaptation-smoke',
        ids=[f'smoke-{i}' for i in range(4)], seed=2026092791, workers=2,
        simulations=8, progress=lambda d, n: print('selfplay', d, n, flush=True))
    games = [game_targets(g) for g in result.records]
    for g in games:
        validate_game(g)
    atomic_write_json(out / 'selfplay.json', {'telemetry': result.telemetry,
        'games': [g.to_dict() for g in result.records]})
    print('targets', sum(len(g['score']) for g in games), flush=True)
    loss = trainer.step(games)
    cp = out / 'trained.pt'
    trainer.save(cp, replay_ids=['smoke'], config_hash='smoke-only')
    resumed = AdaptationTrainer(BOOTSTRAP, device='cuda')
    resumed.restore(cp, replay_ids=['smoke'], config_hash='smoke-only')
    assert all(torch.equal(p, q) for p, q in zip(trainer.model.parameters(), resumed.model.parameters()))
    run_arena(profile=get_profile("torus9|komi=1.5|simulations=8|watchdog=1000|5ch"),
        candidate_path=cp, reference_path=BOOTSTRAP, output_dir=out / 'arena',
        master_seed=2026092792,
        config=ArenaExecutionConfig(games=4, workers=2, games_per_worker=2,
            inference_batch_rows=4, strict_production=False, early_gate_enabled=False),
        progress_callback=lambda d, n: print('arena', d, n, flush=True))
    stats = arena_statistics(out / 'arena')
    report = {'status': 'PASS', 'scope': 'execution smoke, not strength evaluation',
        'games': 4, 'positions': sum(len(g['score']) for g in games),
        'train': loss['losses'], 'arena': stats}
    atomic_write_json(out / 'acceptance.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
