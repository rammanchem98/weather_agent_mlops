"""Removes API key values from text before it reaches the model, the client or a log.

Two layers:
- the actual OPENWEATHER_API_KEY / GEMINI_API_KEY values, read from the
  environment at call time (so a rotated key is covered without a restart);
- any `appid=<value>` query parameter (OpenWeather puts the key in the URL, and
  requests' connection errors include that URL).

Importing this module also installs a log-record hook, so every log line in the
process (any logger, any handler, including exception tracebacks) is redacted."""
import logging
import os
import re

REDACTED = "[REDACTED]"
SECRET_ENV_VARS = ("OPENWEATHER_API_KEY", "GEMINI_API_KEY")
MIN_SECRET_LENGTH = 6  # don't "redact" empty or trivially short placeholder values
_APPID_PARAM = re.compile(r"(appid=)[^&\s'\"<>]+", re.IGNORECASE)


def redact_secrets(text):
    if not text or not isinstance(text, str):
        return text
    for name in SECRET_ENV_VARS:
        value = os.environ.get(name, "")
        if len(value) >= MIN_SECRET_LENGTH:
            text = text.replace(value, REDACTED)
    return _APPID_PARAM.sub(rf"\1{REDACTED}", text)


_base_factory = logging.getLogRecordFactory()
_exception_formatter = logging.Formatter()


def _redacting_record_factory(*args, **kwargs):
    record = _base_factory(*args, **kwargs)
    try:
        record.msg = redact_secrets(record.getMessage())
        record.args = None
        if record.exc_info:
            # Pre-format the traceback; formatters reuse exc_text instead of re-rendering.
            record.exc_text = redact_secrets(_exception_formatter.formatException(record.exc_info))
    except Exception:  # never let redaction break logging
        pass
    return record


if not getattr(logging, "_weather_agent_redaction_installed", False):
    logging.setLogRecordFactory(_redacting_record_factory)
    logging._weather_agent_redaction_installed = True
