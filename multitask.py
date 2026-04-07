"""Root-level re-export so the autograder can do:

    from multitask import MultiTaskPerceptionModel

The actual implementation lives in models/multitask.py.
"""

from models.multitask import MultiTaskPerceptionModel  # noqa: F401

__all__ = ["MultiTaskPerceptionModel"]
