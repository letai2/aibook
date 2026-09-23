"""Authoring material, deliberately separate from learner notebook cells.

Solutions are used only by explicit verification/answer generation, never by
student cells or an automatic fallback when a TODO is unfinished.
"""

def exercises():
    from .foundations import EXERCISES as foundations
    from .attention import EXERCISES as attention
    from .training import EXERCISES as training
    from .context_memory import EXERCISES as context_memory
    from .tools_reasoning import EXERCISES as tools_reasoning
    from .system_training import EXERCISES as system_training
    merged = {}
    for group in (foundations, attention, training, context_memory, tools_reasoning, system_training):
        if merged.keys() & group.keys():
            raise ValueError('Duplicate lesson exercise')
        merged.update(group)
    return merged
