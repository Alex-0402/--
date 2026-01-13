"""
Model architectures for protein side-chain design.
"""

from .encoder import BackboneEncoder
from .decoder import FragmentDecoder

__all__ = [
    'BackboneEncoder',
    'FragmentDecoder'
]
