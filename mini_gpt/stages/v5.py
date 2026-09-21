"""نسخهٔ پنج: توجه با یک بلوک شامل شبکهٔ پیش‌خور، نرمال‌سازی و اتصال جمعی جایگزین می‌شود."""

from mini_gpt.transformer import TransformerBlock
from .v4 import MultiHeadModel


class OneBlockModel(MultiHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.attention = TransformerBlock(config)
