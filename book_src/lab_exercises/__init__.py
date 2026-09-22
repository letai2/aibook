"""Authoring material, deliberately separate from learner notebook cells.

Solutions are used only by explicit verification/answer generation, never by
student cells or an automatic fallback when a TODO is unfinished.
"""

def exercises():
    from .foundations import EXERCISES as foundations
    from .attention import EXERCISES as attention
    from .training import EXERCISES as training
    merged = {}
    for group in (foundations, attention, training):
        if merged.keys() & group.keys():
            raise ValueError('Duplicate lesson exercise')
        merged.update(group)
    return merged

