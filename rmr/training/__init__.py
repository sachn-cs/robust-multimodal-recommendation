"""Training package for the GRE-MC model.

This package provides trainers for the different stages of GRE-MC
training, such as modality completion.
"""

from rmr.training.completion_trainer import CompletionTrainer

__all__ = ["CompletionTrainer"]
