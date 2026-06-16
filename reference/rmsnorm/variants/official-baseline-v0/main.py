import torch
import flashinfer


def run(hidden_states, weight):
    batch_size, hidden_size = hidden_states.shape

    assert hidden_size == 4096

    EPS = 1e-6

    output = flashinfer.norm.rmsnorm(hidden_states, weight, eps=EPS)

    return output
