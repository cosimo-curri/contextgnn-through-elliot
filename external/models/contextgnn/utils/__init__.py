"""
Disclaimer: A substantial portion of the code contained in the current package is directly taken from
this public GitHub repository: https://github.com/kumo-ai/ContextGNN/tree/xinwei_add_static_data_and_model_v1 and
adapted to work within the framework Elliot (https://elliot.readthedocs.io/en/latest/).
Please refer to the above cited GitHub repository and to the original paper
of ContextGNN (https://arxiv.org/abs/2411.19513) for further details.
"""

from enum import Enum

import torch


class RHSEmbeddingMode(Enum):
    r"""Specifies how to incorporate shallow RHS representations in link
    prediction tasks.
    """
    # Use trainable look-up embeddings (transductive):
    LOOKUP = 'lookup'
    # Purely rely on shallow RHS input features (inductive):
    FEATURE = 'feature'
    # Fuse look-up embeddings and shallow RHS input features (transductive):
    FUSION = 'fusion'


def sparse_matrix_to_sparse_coo(sci_sparse_matrix):
    r"""Converts scipy sparse matrix to sparse coo matrix."""
    sci_sparse_coo = sci_sparse_matrix.tocoo()

    # Get the data, row indices, and column indices
    values = torch.tensor(sci_sparse_coo.data, dtype=torch.int64)
    row_indices = torch.tensor(sci_sparse_coo.row, dtype=torch.int64)
    col_indices = torch.tensor(sci_sparse_coo.col, dtype=torch.int64)

    # Create a PyTorch sparse tensor
    torch_sparse_tensor = torch.sparse_coo_tensor(
        indices=torch.stack([row_indices, col_indices]), values=values,
        size=sci_sparse_matrix.shape)
    return torch_sparse_tensor


__all__ = classes = [
    'RHSEmbeddingMode',
    'sparse_matrix_to_sparse_coo'
]
