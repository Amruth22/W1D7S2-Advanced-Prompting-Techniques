"""
Advanced Prompting Techniques Package
Contains all prompt templates for different techniques
"""

# Import all technique modules
from . import few_shot
from . import chain_of_thought
from . import tree_of_thought
from . import self_consistency
from . import meta_prompting

__all__ = [
    'few_shot',
    'chain_of_thought', 
    'tree_of_thought',
    'self_consistency',
    'meta_prompting'
]