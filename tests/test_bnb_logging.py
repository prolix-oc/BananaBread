import logging
from unittest.mock import Mock

from bananabread import bnb_logging


def _record(message: str, *, name: str = "bitsandbytes.autograd._functions") -> logging.LogRecord:
    return logging.LogRecord(name, logging.WARNING, __file__, 1, message, (), None)


def test_matmul_cast_filter_replaces_only_first_warning_with_app_notice():
    notice_logger = Mock(spec=logging.Logger)
    cast_filter = bnb_logging._MatmulCastFilter(notice_logger)
    message = (
        "MatMul8bitLt: inputs will be cast from torch.bfloat16 " "to float16 during quantization"
    )

    assert cast_filter.filter(_record(message)) is False
    assert cast_filter.filter(_record(message)) is False

    notice_logger.warning.assert_called_once_with(
        "bitsandbytes reported: %s. Further instances will be suppressed.",
        message,
    )


def test_matmul_cast_filter_leaves_other_bitsandbytes_records_alone():
    notice_logger = Mock(spec=logging.Logger)
    cast_filter = bnb_logging._MatmulCastFilter(notice_logger)

    assert cast_filter.filter(_record("A different bitsandbytes warning")) is True
    other_logger_record = _record(
        "MatMul8bitLt: inputs will be cast from torch.bfloat16",
        name="other",
    )
    assert cast_filter.filter(other_logger_record) is True
    notice_logger.warning.assert_not_called()


def test_filter_installation_is_idempotent():
    logger = logging.getLogger("bitsandbytes.autograd._functions")
    original_filters = logger.filters[:]
    try:
        logger.filters.clear()
        bnb_logging.install_bnb_matmul_cast_filter()
        bnb_logging.install_bnb_matmul_cast_filter()
        assert logger.filters.count(bnb_logging._MATMUL_CAST_FILTER) == 1
    finally:
        logger.filters[:] = original_filters
