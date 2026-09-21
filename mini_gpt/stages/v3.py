"""نسخهٔ سه: فقط یک تفاوت با v2؛ آینده پوشانده می‌شود."""

from .v2 import SingleHead


class CausalSingleHead(SingleHead):
    def __init__(self, vocab_size, channels=16):
        super().__init__(vocab_size, channels)
        self.causal = True
