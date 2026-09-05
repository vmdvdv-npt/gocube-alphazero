from alphazero.NNetArchitecture import ResNet, FullyConnected
from alphazero.pytorch_classification.utils import Bar, AverageMeter
from alphazero.Game import GameState
from alphazero.utils import dotdict
from threading import Event
from abc import ABC, abstractmethod
from typing import Tuple, Optional

import torch.optim as optim
import numpy as np
import warnings
import torch
import pickle
import time
import os


class BaseWrapper(ABC):
    """Base neural-network wrapper."""

    def __init__(self, game_cls: GameState, args):
        self.game_cls = game_cls
        self.args = args
        self.stop_train = Event()
        self.pause_train = Event()

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    @abstractmethod
    def train(self, examples, num_steps: int) -> Tuple[float, float]:
        pass

    @abstractmethod
    def predict(self, board: np.ndarray) -> Tuple[np.ndarray, float]:
        pass

    @abstractmethod
    def save_checkpoint(self, folder, filename):
        pass

    @abstractmethod
    def load_checkpoint(self, folder, filename):
        pass


class NNetWrapper(BaseWrapper):
    def __init__(self, game_cls, args):
        super().__init__(game_cls, args)
        self.nnet = None
        self._load_nnet(args)
        self.action_size = game_cls.action_size()
        self.optimizer = args.optimizer(self.nnet.parameters(), lr=args.lr, **args.optimizer_args)
        self.scheduler = args.scheduler(self.optimizer, **args.scheduler_args)
        self.verbose = args.scheduler_args.get('verbose')
        if args.cuda:
            self.nnet.cuda()
        self.current_step = 0
        self.total_steps = 0
        self.last_train_planned_steps = 0
        self.last_train_actual_steps = 0
        self.last_train_examples_seen = 0
        self.last_train_learning_rate = float(self.optimizer.param_groups[0]['lr'])
        self.l_pi = 0
        self.l_v = 0
        self.l_ownership = 0
        self.l_score = 0
        self.l_total = 0
        self.step_time = 0
        self.elapsed_time = 0
        self.eta = 0
        self.__loaded = False

    def _load_nnet(self, args):
        if args.nnet_type == 'resnet':
            self.nnet = ResNet(self.game_cls, args)
        elif args.nnet_type == 'fc':
            self.nnet = FullyConnected(self.game_cls, args)
        elif args.nnet_type == 'graph':
            from alphazero.envs.gocube.network import GraphNet
            self.nnet = GraphNet(self.game_cls, args)
        else:
            raise ValueError(f'Unknown NNet type "{args.nnet_type}"')

    @property
    def loaded(self):
        return self.__loaded

    def train(self, batches, train_steps):
        self.total_steps = train_steps
        self.current_step = 0
        self.last_train_planned_steps = int(train_steps)
        self.last_train_actual_steps = 0
        self.last_train_examples_seen = 0
        self.last_train_learning_rate = float(self.optimizer.param_groups[0]['lr'])
        if train_steps <= 0:
            return self.l_pi, self.l_v
        self.nnet.train()
        data_time = AverageMeter()
        batch_time = AverageMeter()
        pi_losses = AverageMeter()
        v_losses = AverageMeter()
        ownership_losses = AverageMeter()
        score_losses = AverageMeter()
        auxiliary = bool(getattr(self.args, 'gocube_auxiliary_targets', False))
        if self.verbose:
            print(f'Current LR: {self.optimizer.param_groups[0]["lr"]}')
        bar = Bar('Training Net', max=train_steps)
        while self.current_step < train_steps and not self.stop_train.is_set():
            for batch_idx, batch in enumerate(batches):
                if self.current_step == train_steps or self.stop_train.is_set():
                    break
                while self.pause_train.is_set():
                    time.sleep(.1)
                start = time.time()
                self.current_step += 1
                ownership_mask = None
                if auxiliary:
                    if len(batch) == 6:
                        boards, target_pis, target_vs, target_scores, target_ownership, ownership_mask = batch
                    elif len(batch) == 5:
                        boards, target_pis, target_vs, target_scores, target_ownership = batch
                    else:
                        raise ValueError(f'Expected 5 or 6 auxiliary tensors, got {len(batch)}')
                else:
                    boards, target_pis, target_vs = batch
                if self.args.cuda:
                    boards = boards.contiguous().cuda()
                    target_pis = target_pis.contiguous().cuda()
                    target_vs = target_vs.contiguous().cuda()
                    if auxiliary:
                        target_scores = target_scores.contiguous().cuda()
                        target_ownership = target_ownership.contiguous().cuda()
                        if ownership_mask is not None:
                            ownership_mask = ownership_mask.contiguous().cuda()
                data_time.update(time.time() - start)
                outputs = self.nnet(boards)
                out_pi, out_v = outputs[0], outputs[1]
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v
                if auxiliary:
                    out_ownership, out_score = outputs[2], outputs[3]
                    l_ownership = self.loss_ownership(target_ownership, out_ownership, ownership_mask)
                    l_score = self.loss_score(target_scores, out_score)
                    total_loss = total_loss + l_ownership + l_score
                    ownership_losses.update(l_ownership.item(), boards.size(0))
                    score_losses.update(l_score.item(), boards.size(0))
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                self.last_train_actual_steps += 1
                self.last_train_examples_seen += int(boards.size(0))
                batch_time.update(time.time() - start)
                self.l_pi = pi_losses.avg
                self.l_v = v_losses.avg
                self.l_ownership = ownership_losses.avg if auxiliary else 0
                self.l_score = score_losses.avg if auxiliary else 0
                self.l_total = self.l_pi + self.l_v + self.l_ownership + self.l_score
                self.step_time = data_time.avg + batch_time.avg
                self.elapsed_time = bar.elapsed_td
                self.eta = bar.eta_td
                suffix = (
                    '({step}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | '
                    'Total: {total:} | ETA: {eta:} | Loss_pi: {lpi:.4f} | Loss_v: {lv:.3f}'
                ).format(step=self.current_step, size=train_steps, data=data_time.avg, bt=batch_time.avg,
                         total=bar.elapsed_td, eta=bar.eta_td, lpi=pi_losses.avg, lv=v_losses.avg)
                if auxiliary:
                    suffix += f' | Loss_owner: {ownership_losses.avg:.3f} | Loss_score: {score_losses.avg:.3f}'
                bar.suffix = suffix
                bar.next()
        scheduler_loss = pi_losses.avg + v_losses.avg
        if auxiliary:
            scheduler_loss += ownership_losses.avg + score_losses.avg
        self.scheduler.step(
            scheduler_loss if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau) else None
        )
        bar.update()
        bar.finish()
        print()
        return pi_losses.avg, v_losses.avg

    def predict(self, board: np.ndarray):
        board = torch.FloatTensor(board.astype(np.float64))
        if self.args.cuda:
            board = board.contiguous().cuda()
        with torch.no_grad():
            self.nnet.eval()
            outputs = self.nnet(board)
            pi, v = outputs[0], outputs[1]
            return torch.exp(pi).data.cpu().numpy()[0], torch.exp(v).data.cpu().numpy()[0]

    def process(self, batch: torch.Tensor):
        batch = batch.type(torch.FloatTensor)
        if self.args.cuda:
            batch = batch.cuda()
        self.nnet.eval()
        with torch.no_grad():
            outputs = self.nnet(batch)
            pi, v = outputs[0], outputs[1]
            return torch.exp(pi), torch.exp(v)

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return -self.args.value_loss_weight * torch.sum(targets * outputs) / targets.size()[0]

    def loss_ownership(self, targets, outputs, mask=None):
        weight = float(getattr(self.args, 'ownership_loss_weight', 0.5))
        point_loss = -torch.sum(targets * outputs, dim=2)
        if mask is None:
            return weight * torch.mean(point_loss)
        mask = mask.to(point_loss.dtype)
        denominator = torch.clamp(mask.sum(), min=1.0)
        return weight * torch.sum(point_loss * mask) / denominator

    def loss_score(self, targets, outputs):
        weight = float(getattr(self.args, 'score_loss_weight', 0.5))
        return weight * torch.mean((targets - outputs) ** 2)

    def _checkpoint_contract(self):
        fields = {}
        mapping = {
            'gocube_terminal_adjudicator': 'TERMINAL_ADJUDICATOR_ID',
            'gocube_observation_schema': 'OBSERVATION_SCHEMA',
            'gocube_topology': None,
            'gocube_size': None,
        }
        if hasattr(self.game_cls, 'TERMINAL_ADJUDICATOR_ID'):
            fields['gocube_terminal_adjudicator'] = self.game_cls.TERMINAL_ADJUDICATOR_ID
        if hasattr(self.game_cls, 'OBSERVATION_SCHEMA'):
            fields['gocube_observation_schema'] = self.game_cls.OBSERVATION_SCHEMA
        if hasattr(self.game_cls, 'topology_kind'):
            fields['gocube_topology'] = self.game_cls.topology_kind()
            fields['gocube_size'] = self.game_cls.board_size()
        if hasattr(self.game_cls, 'rules_fingerprint'):
            fields['gocube_rules_fingerprint'] = self.game_cls.rules_fingerprint()
        return fields

    def _validate_saved_contract(self, saved_args):
        expected = self._checkpoint_contract()
        strict_v3 = expected.get('gocube_terminal_adjudicator') == 'gocube-katago-japanese-v3'
        for key, value in expected.items():
            if key not in saved_args:
                if strict_v3:
                    raise ValueError(f'Checkpoint missing required GoCube V3 metadata: {key}')
                continue
            if saved_args[key] != value:
                raise ValueError(
                    f'Checkpoint GoCube contract mismatch for {key}: '
                    f'saved={saved_args[key]!r}, expected={value!r}'
                )

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar', make_dirs=True):
        filepath = os.path.join(folder, filename)
        if make_dirs and not os.path.exists(folder):
            os.makedirs(folder)
        torch.save({
            'state_dict': self.nnet.state_dict(),
            'opt_state': self.optimizer.state_dict(),
            'sch_state': self.scheduler.state_dict(),
            'args': self.args
        }, filepath, pickle_protocol=pickle.HIGHEST_PROTOCOL)

    def load_checkpoint(
        self,
        folder='checkpoint',
        filename='checkpoint.pth.tar',
        use_saved_args=True,
        device=None,
        load_training_state=True,
    ) -> Optional[dotdict]:
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError("No model in path {}".format(filepath))
        if device not in (None, 'cpu', 'cuda'):
            raise ValueError("device must be one of None, 'cpu', or 'cuda'")
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        checkpoint = torch.load(filepath, map_location=device) if device else torch.load(filepath)
        args_saved = 'args' in checkpoint
        if args_saved:
            self._validate_saved_contract(checkpoint['args'])
        elif self._checkpoint_contract().get('gocube_terminal_adjudicator') == 'gocube-katago-japanese-v3':
            raise ValueError('V3 checkpoint has no saved args/contract metadata')
        if use_saved_args and args_saved:
            saved_args = checkpoint['args']
            if device is not None:
                saved_args = saved_args.copy()
                saved_args.cuda = device == 'cuda'
            self.args = saved_args
            self.__init__(self.game_cls, self.args)
        elif use_saved_args and not args_saved:
            warnings.warn('No args were saved in the checkpoint file, therefore they were not loaded.')
        self.nnet.load_state_dict(checkpoint['state_dict'])
        if load_training_state and 'opt_state' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['opt_state'])
        if load_training_state and 'sch_state' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['sch_state'])
        self.__loaded = True
        if args_saved:
            return self.args if use_saved_args else checkpoint['args']

    @classmethod
    def from_checkpoint(cls, game_cls, *args, **kwargs):
        instance = cls.__new__(cls)
        instance.game_cls = game_cls
        instance.load_checkpoint(*args, **kwargs)
        return instance