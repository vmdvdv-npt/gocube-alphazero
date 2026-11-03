"""Astra entry point: edit FP32 NumPy arrays in place using algorithms.

This template deliberately leaves the zero genesis unchanged; no game strategy.
Names/shapes: `experiment_cli.py schema`. No model execution or data fitting.
"""


def program(weights):
    # Example of the interface only: weights['input_projection.weight'][:] = 0
    # LayerNorm scale is also zero at genesis; explicitly design all needed scales.
    pass
