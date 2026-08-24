"""Logging helpers for bitsandbytes integration."""

import logging
import threading

_BNB_AUTOGRAD_LOGGER = "bitsandbytes.autograd._functions"
_MATMUL_CAST_PREFIX = "MatMul8bitLt: inputs will be cast from "


class _MatmulCastFilter(logging.Filter):
    """Replace repeated bitsandbytes cast warnings with one app-level notice."""

    def __init__(self, notice_logger: logging.Logger) -> None:
        super().__init__()
        self._notice_logger = notice_logger
        self._notice_emitted = False
        self._lock = threading.Lock()

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if record.name != _BNB_AUTOGRAD_LOGGER or not message.startswith(_MATMUL_CAST_PREFIX):
            return True

        with self._lock:
            if not self._notice_emitted:
                self._notice_emitted = True
                self._notice_logger.warning(
                    "bitsandbytes reported: %s. Further instances will be suppressed.",
                    message,
                )
        return False


_MATMUL_CAST_FILTER = _MatmulCastFilter(logging.getLogger("BananaBread-Emb"))
_INSTALL_LOCK = threading.Lock()


def install_bnb_matmul_cast_filter() -> None:
    """Install the process-wide filter once, before an 8-bit model runs."""
    bnb_logger = logging.getLogger(_BNB_AUTOGRAD_LOGGER)
    with _INSTALL_LOCK:
        if _MATMUL_CAST_FILTER not in bnb_logger.filters:
            bnb_logger.addFilter(_MATMUL_CAST_FILTER)
