import numpy as np


_POLICY_EPSILON = 1e-12


def normalize_masked_policy(policy, valids, epsilon=_POLICY_EPSILON):
    """Mask a policy to legal actions and normalize it safely."""
    policy = np.asarray(policy, dtype=np.float32)
    valids = np.asarray(valids, dtype=np.float32)

    if policy.ndim != 1 or valids.ndim != 1 or policy.shape != valids.shape:
        raise ValueError('policy and valids must be 1D arrays with matching shapes')

    legal = valids != 0
    legal_count = int(np.count_nonzero(legal))
    if legal_count == 0:
        raise ValueError('cannot normalize policy: no legal actions')

    masked = np.where(legal, policy, np.float32(0.0)).astype(np.float32, copy=False)
    total = float(np.sum(masked, dtype=np.float64))

    if np.isfinite(total) and total > epsilon:
        masked /= np.float32(total)
        return masked

    masked = np.zeros_like(policy, dtype=np.float32)
    masked[legal] = np.float32(1.0 / legal_count)
    return masked
