import re
from typing import Sequence

from transformers import BertTokenizerFast


class ProtBertTokenizer:
    def __init__(self):
        super().__init__()
        self._tokenizer = BertTokenizerFast.from_pretrained(
            'Rostlab/prot_bert_bfd',
            do_lower_case=False,
        )

    def encode(self, x: str) -> Sequence[int]:
        x = re.sub(r'[UZOB]', 'X', x)
        x = ' '.join(list(x))
        x = self._tokenizer.encode(x)
        # return np.array(x)
        return x
