import threading
import time
from concurrent.futures import ThreadPoolExecutor

from bananabread.models.qwen import BaseQwenModel


class BorrowCheckingTokenizer:
    def __init__(self):
        self._in_use = False
        self.calls = 0

    def __call__(self, batch, **kwargs):
        if self._in_use:
            raise RuntimeError("Already borrowed")
        self._in_use = True
        try:
            time.sleep(0.01)
            self.calls += 1
            return {"input_ids": batch, "kwargs": kwargs}
        finally:
            self._in_use = False


def test_qwen_tokenize_serializes_shared_fast_tokenizer_access():
    model = BaseQwenModel.__new__(BaseQwenModel)
    model.max_length = 128
    model.tokenizer = BorrowCheckingTokenizer()
    model.tokenizer_lock = threading.RLock()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(model.tokenize, [f"text-{i}"], "pt") for i in range(2)]

    assert [future.result()["input_ids"] for future in futures] == [["text-0"], ["text-1"]]
    assert model.tokenizer.calls == 2
