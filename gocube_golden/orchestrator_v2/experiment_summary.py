"""Publish an operator-reviewed cross-job conclusion through the durable outbox.

This boundary reads completed evidence; it never selects a training parent or
starts an engine. The operator remains responsible for the scientific conclusion.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Mapping

from ..notifications import OperatorEvent, create_telegram_dispatcher
from ..notifications.store import _atomic_create_json
from .operator_guide import guide_metadata


def _read_ref(ref: object) -> tuple[Path, bytes]:
    if not isinstance(ref, Mapping) or set(ref) != {'path', 'sha256'}:
        raise ValueError('evidence reference requires path and sha256')
    path = Path(ref['path'])
    if not path.is_absolute():
        raise ValueError('evidence path must be absolute')
    raw = path.read_bytes()
    if ref['sha256'] != 'sha256:' + hashlib.sha256(raw).hexdigest():
        raise ValueError(f'evidence SHA mismatch: {path}')
    return path, raw


def _text(value: object, label: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f'{label} requires nonempty text of at most {limit} characters')
    return value


def publish_experiment_summary(config: Mapping, *, runs_root: Path,
                               check_only: bool = False) -> dict:
    guide = guide_metadata()
    if set(config) != {'schema', 'summary_id', 'report'} or config['schema'] != 'gocube-experiment-summary-v1':
        raise ValueError('unsupported experiment summary configuration')
    summary_id = config['summary_id']
    if not isinstance(summary_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', summary_id):
        raise ValueError('invalid summary_id')
    report_path, raw = _read_ref(config['report'])
    report = json.loads(raw)
    required = {'schema', 'status', 'combinations', 'confirmation_results', 'decision', 'rationale', 'evidence'}
    if not isinstance(report, dict) or set(report) != required or report['schema'] != 'gocube-experiment-conclusion-v1':
        raise ValueError('unsupported experiment conclusion report')
    if report['status'] != 'COMPLETED':
        raise ValueError('summary requires a completed report')
    combinations = report['combinations']
    if not isinstance(combinations, list) or len(combinations) != 4:
        raise ValueError('summary requires four combinations')
    names = set()
    seeds = None
    for row in combinations:
        if not isinstance(row, dict) or set(row) != {'id', 'learning_rate', 'updates_per_generation', 'seed_results'}:
            raise ValueError('invalid combination')
        name = _text(row['id'], 'combination id', 40)
        if name in names:
            raise ValueError('duplicate combination id')
        names.add(name)
        lr = row['learning_rate']
        if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not 0 < lr < 1:
            raise ValueError('invalid learning rate')
        if type(row['updates_per_generation']) is not int or row['updates_per_generation'] <= 0:
            raise ValueError('invalid updates budget')
        results = row['seed_results']
        if not isinstance(results, dict) or len(results) != 2:
            raise ValueError('each combination requires two learner seeds')
        if any(not re.fullmatch(r'[0-9]+', seed) for seed in results):
            raise ValueError('invalid learner seed')
        if seeds is not None and seeds != set(results):
            raise ValueError('learner seeds must match across combinations')
        seeds = set(results)
        for value in results.values():
            _text(value, 'seed result', 160)
    if report['decision'] not in names | {'INCONCLUSIVE'}:
        raise ValueError('decision must identify a combination or INCONCLUSIVE')
    _text(report['rationale'], 'rationale', 600)
    confirmation = report['confirmation_results']
    if not isinstance(confirmation, dict) or set(confirmation) != seeds:
        raise ValueError('confirmation requires both learner seeds')
    for value in confirmation.values():
        _text(value, 'confirmation result', 240)
    evidence = report['evidence']
    if not isinstance(evidence, list) or not evidence:
        raise ValueError('completed report requires saved evidence')
    for ref in evidence:
        _read_ref(ref)
    # Validate the structured payload before creating any owner storage.
    event = OperatorEvent.create('EXPERIMENT_COMPLETED', topology='torus9',
        owner_type='experiment', owner_id=summary_id, action_id='final-summary',
        payload={**{k: report[k] for k in ('combinations', 'confirmation_results', 'decision', 'rationale')},
                 'cross_job_summary': True, 'report_ref': str(report_path),
                 'report_sha256': config['report']['sha256']},
        evidence_refs=[dict(config['report']), *evidence],
        producer_version='gocube-experiment-summary-v1',
        identity={'summary_id': summary_id, 'report_sha256': config['report']['sha256']})
    owner = Path(runs_root).resolve() / 'torus9' / 'orchestration' / 'summaries' / summary_id
    seal = owner / 'report.json'
    sealed = {'parameters': dict(config), 'report': report}
    if seal.exists() and json.loads(seal.read_text()) != sealed:
        raise ValueError('summary_id already has a different immutable report')
    result = {'summary_id': summary_id, 'event_id': event.event_id, 'owner_root': str(owner),
              'report_sha256': config['report']['sha256'], 'guide_sha256': guide['sha256']}
    if check_only:
        return {**result, 'state': 'VALIDATED'}
    owner.mkdir(parents=True, exist_ok=True)
    try:
        _atomic_create_json(seal, sealed)
    except FileExistsError:
        if json.loads(seal.read_text()) != sealed:
            raise ValueError('summary_id already has a different immutable report')
    dispatcher = create_telegram_dispatcher(owner)
    try:
        dispatcher.publish(event)
        dispatcher.flush(7.0)
        delivery = dispatcher.store.read_delivery(event.event_id)
        return {**result, 'state': delivery.status if delivery else 'PENDING',
                'last_error_code': delivery.last_error_code if delivery else None}
    finally:
        dispatcher.close(timeout=1.0)
