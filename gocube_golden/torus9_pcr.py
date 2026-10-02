"""Run-owned Torus9 playout caps; selection is independent of worker scheduling."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import random

from .provenance import derive_seed

PCR_NAMESPACE = 'torus9-playout-cap-randomization-v1'


@dataclass(frozen=True)
class PlayoutCapRandomization:
    cheap_simulations: int
    full_simulations: int
    full_probability: float

    def __post_init__(self):
        for name in ('cheap_simulations', 'full_simulations'):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f'pcr.{name} must be a positive integer')
        if self.full_simulations <= self.cheap_simulations:
            raise ValueError('pcr.full_simulations must exceed cheap_simulations')
        p = self.full_probability
        if type(p) not in (int, float) or not math.isfinite(p) or not 0 < p < 1:
            raise ValueError('pcr.full_probability must be finite and between 0 and 1')

    @classmethod
    def from_dict(cls, value):
        keys = {'cheap_simulations', 'full_simulations', 'full_probability'}
        if not isinstance(value, Mapping) or set(value) != keys:
            raise ValueError('pcr requires cheap_simulations, full_simulations, full_probability only')
        return cls(**value)

    @property
    def nominal_mean_simulations(self):
        return self.cheap_simulations * (1 - self.full_probability) + self.full_simulations * self.full_probability

    def mode(self, game_seed: int, ply: int) -> str:
        rng = random.Random(derive_seed(game_seed, ply, PCR_NAMESPACE))
        return 'full' if rng.random() < self.full_probability else 'cheap'


def resolve_search_mode(settings):
    mode = settings.get('search_mode', 'fixed')
    if mode not in ('fixed', 'pcr'):
        raise ValueError('self_play.search_mode must be fixed or pcr')
    if mode == 'fixed':
        if settings.get('pcr') is not None:
            raise ValueError('self_play.pcr requires search_mode=pcr')
        return None
    return PlayoutCapRandomization.from_dict(settings.get('pcr'))


def search_description(settings):
    pcr = resolve_search_mode(settings)
    if pcr is None:
        return f"self-play MCTS={settings.get('mcts_simulations', settings.get('simulations', '-'))} sims"
    return '\n'.join([
        'Self-play MCTS: PCR',
        f'Cheap: {pcr.cheap_simulations} sims / {100 * (1-pcr.full_probability):g}%',
        f'Full: {pcr.full_simulations} sims / {100 * pcr.full_probability:g}%',
        f'Nominal mean: {pcr.nominal_mean_simulations:g} sims',
    ])


def position_telemetry(records, pcr):
    raw = full = cheap = training = 0
    for game in records:
        for position in game.positions:
            raw += 1
            full += position.search_mode == 'full'
            cheap += position.search_mode == 'cheap'
            training += position.training_eligible
    return {
        'pcr_full_positions': full, 'pcr_cheap_positions': cheap,
        'pcr_full_fraction': full / raw if raw else 0.,
        'pcr_cheap_fraction': cheap / raw if raw else 0.,
        'pcr_full_simulations': pcr.full_simulations,
        'pcr_cheap_simulations': pcr.cheap_simulations,
        'nominal_mean_simulations': pcr.nominal_mean_simulations,
        'training_positions': training,
        'raw_positions': raw,
    }
