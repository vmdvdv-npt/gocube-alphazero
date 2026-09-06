import torch

from alphazero.inference_batching import process_coalesced_inference
from alphazero.search_contract import SearchOutput


class _SearchNet:
    def __init__(self, point_count, action_size):
        self.point_count = point_count
        self.action_size = action_size
        self.calls = 0

    def process_for_search(self, batch):
        self.calls += 1
        rows = batch.size(0)
        base = torch.arange(rows, dtype=torch.float32).view(rows, 1)
        policy = base.repeat(1, self.action_size)
        value = torch.cat((base, base + 1, base + 2), dim=1)
        score = base / 10.0
        ownership = torch.zeros(rows, self.point_count, 3)
        ownership[:, :, 0] = 0.2
        ownership[:, :, 1] = 0.3
        ownership[:, :, 2] = 0.5
        return SearchOutput(policy=policy, value=value, score=score, ownership=ownership)


def test_coalesced_search_inference_splits_all_four_heads_from_one_forward():
    point_count = 6
    action_size = point_count + 1
    net = _SearchNet(point_count, action_size)

    inputs = [torch.zeros(2, 3), torch.zeros(3, 3)]
    policy = [torch.zeros(2, action_size), torch.zeros(3, action_size)]
    value = [torch.zeros(2, 3), torch.zeros(3, 3)]
    score = [torch.zeros(2, 1), torch.zeros(3, 1)]
    ownership = [torch.zeros(2, point_count, 3), torch.zeros(3, point_count, 3)]
    ready = [torch.multiprocessing.Event(), torch.multiprocessing.Event()]

    rows = process_coalesced_inference(
        net,
        [0, 1],
        inputs,
        policy,
        value,
        ready,
        score_tensors=score,
        ownership_tensors=ownership,
    )

    assert rows == 5
    assert net.calls == 1
    assert ready[0].is_set() and ready[1].is_set()
    assert torch.allclose(score[0].flatten(), torch.tensor([0.0, 0.1]))
    assert torch.allclose(score[1].flatten(), torch.tensor([0.2, 0.3, 0.4]))
    assert torch.allclose(ownership[0][0, 0], torch.tensor([0.2, 0.3, 0.5]))
    assert torch.allclose(ownership[1][-1, -1], torch.tensor([0.2, 0.3, 0.5]))
