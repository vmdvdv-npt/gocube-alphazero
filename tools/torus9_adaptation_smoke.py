"""Bounded real CUDA self-play -> targets -> update -> reload acceptance."""
from pathlib import Path
import argparse
import json
import torch
from gocube_golden.torus9_adaptation import AdaptationTrainer, selfplay, game_targets, validate_game
from gocube_golden.orchestrator_v2.adaptation import BOOTSTRAP
from gocube_golden.process_supervision import atomic_write_json


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
    report = {'status': 'PASS', 'scope': 'selfplay/training/reload only; arena requires Orchestrator V2',
        'games': 4, 'positions': sum(len(g['score']) for g in games),
        'train': loss['losses']}
    atomic_write_json(out / 'acceptance.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
