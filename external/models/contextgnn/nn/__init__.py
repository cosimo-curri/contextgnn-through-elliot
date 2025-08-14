"""
Disclaimer: A substantial portion of the code contained in the current package is directly taken from
this public GitHub repository: https://github.com/kumo-ai/ContextGNN/tree/xinwei_add_static_data_and_model_v1 and
adapted to work within the framework Elliot (https://elliot.readthedocs.io/en/latest/).
Please refer to the above cited GitHub repository and to the original paper
of ContextGNN (https://arxiv.org/abs/2411.19513) for further details.
"""

from .encoder import HeteroEncoder
from .rhs_embedding import RHSEmbedding

__all__ = classes = ['HeteroEncoder', 'RHSEmbedding']