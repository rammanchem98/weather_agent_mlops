"""Removes API key values from text before it reaches the model, the client or a log.

Two layers:
- the actual OPENWEATHER_API_KEY / GEMINI_API_KEY values, read from the
  environment at call time (so a rotated key is covered without a restart);
- any `appid=<value>` query parameter (OpenWeather puts the key in the URL, and
  requests' connection errors include that URL).

Importing this module also installs a log-record hook for every logger and
handler in the process. It redacts the message if it is a string, string values
inside record.args (keeping the tuple/dict shape formatters rely on), and the
exception traceback text. Non-string args (e.g. an exception object passed for
"%s") are left as they are, so code should log str() of anything that may hold
a key, as weather_api does."""
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


def _redact_value(value):
    return redact_secrets(value) if isinstance(value, str) else value


def _redact_args(args):
    """Same shape in, same shape out: formatters such as uvicorn's access log
    unpack record.args as a tuple, so it must never become None or a string."""
    if isinstance(args, tuple):
        return tuple(_redact_value(a) for a in args)
    if isinstance(args, dict):
        return {key: _redact_value(value) for key, value in args.items()}
    return args


def _redacting_record_factory(*args, **kwargs):
    record = _base_factory(*args, **kwargs)
    try:
        # Compute everything first, then assign, so a failure part-way leaves
        # the record exactly as the logging call created it.
        new_msg = _redact_value(record.msg)
        new_args = _redact_args(record.args)
        new_exc_text = None
        if record.exc_info:
            # Pre-format the traceback; formatters reuse exc_text instead of re-rendering.
            new_exc_text = redact_secrets(_exception_formatter.formatException(record.exc_info))
    except Exception:  # never let redaction break or reshape logging
        return record
    record.msg = new_msg
    record.args = new_args
    if new_exc_text is not None:
        record.exc_text = new_exc_text
    return record


if not getattr(logging, "_weather_agent_redaction_installed", False):
    logging.setLogRecordFactory(_redacting_record_factory)
    logging._weather_agent_redaction_installed = True
