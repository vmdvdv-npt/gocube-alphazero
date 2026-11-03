"""Explicit zero/demo fixture creation; use experiment_cli zero-pass for Astra."""
import argparse
import json
from pathlib import Path
from model import TorusNet, write_demo_weights
from weights import save_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--design', choices=('zeros', 'demo'), default='zeros')
    p.add_argument('--channels', type=int, choices=(5,), default=5)
    p.add_argument('--width', type=int, choices=(120,), default=120)
    p.add_argument('--blocks', type=int, choices=(8,), default=8)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    model = TorusNet()
    if args.design == 'demo':
        write_demo_weights(model)
    print(json.dumps(save_checkpoint(model, args.output, design=args.design)))


if __name__ == '__main__':
    main()
