import os
import re
import socket
import logging
import inspect
import traceback
from typing import NoReturn
from django.core.mail import send_mail

# Try to use the WWW logger implementation
try:
    from eamllogger.EasyAutoMLLogger import EasyAutoMLLogger as _EasyAutoMLLogger
    
    class EasyAutoMLLogger(_EasyAutoMLLogger):
        """Logger class - uses WWW implementation if available"""
        pass
except ImportError:
    try:
        from colorlog import ColoredFormatter
        from django.core.mail import send_mail
        
        GLOBAL_LOGGER_ALL_LEVELS = {10: "DEBUG", 20: "INFO", 30: "WARNING", 40: "ERROR", 50: "CRITICAL"}
        GLOBAL_LOGGER_DEFAULT_IS_CONSOLE_LEVEL = getattr(logging, os.environ.get("GLOBAL_LOGGER_CONSOLE_LEVEL", "DEBUG"))
        GLOBAL_LOGGER_SAVE_ABOVE_OR_EQUAL_THIS_LEVEL = getattr(logging, os.environ.get("GLOBAL_LOGGER_DB_LEVEL", "ERROR"))
        GLOBAL_LOGGER_EMAIL_ENABLED = True
        
        class EasyAutoMLLogger(logging.Logger):
            """Fallback Logger implementation if WWW is not available"""
            
            def __init__(self, level=GLOBAL_LOGGER_DEFAULT_IS_CONSOLE_LEVEL, **kwargs):
                self.main_module = os.path.normpath(traceback.extract_stack()[0].filename)
                self.logger_name = self.main_module.split(os.path.sep)[-1]
                super().__init__(self.logger_name)
                ch = logging.StreamHandler()
                ch.setLevel(level)
                
                # Use simpler format to avoid missing field errors
                try:
                    _log_format = ColoredFormatter(
                        "%(log_color)s%(levelname)-8s|%(name)s|%(message)s",
                        log_colors={
                            "DEBUG": "blue",
                            "INFO": "green",
                            "WARNING": "yellow",
                            "ERROR": "red",
                            "CRITICAL": "bold_red",
                        },
                    )
                except Exception:
                    # Fallback to standard formatter if ColoredFormatter fails
                    _log_format = logging.Formatter('%(levelname)-8s|%(name)s|%(message)s')
                
                
                ch.setFormatter(_log_format)
                self.addHandler(ch)
            
            def _log(self, level, msg, args, **kwargs) -> NoReturn:
                _trace = traceback.format_stack(inspect.currentframe())[:-2]
                _trace_as_string = "\n".join(_trace)
                p = min(35, max(0, len([s for s in (inspect.stack(0)) if "\\plugins\\" not in s])))
                position_level_spaces_before = ("" if p == 0 else " " * p)
                position_level_spaces_after = ("" if p >= 35 else " " * (35 - p))
                super()._log(level, msg, args, extra={"position_level_spaces_before": position_level_spaces_before, "position_level_spaces_after": position_level_spaces_after}, **kwargs)
            
            def debug(self, msg, *args, **kwargs):
                if self.isEnabledFor(logging.DEBUG):
                    self._log(logging.DEBUG, msg, args, **kwargs)
            
            def info(self, msg, *args, **kwargs):
                if self.isEnabledFor(logging.INFO):
                    self._log(logging.INFO, msg, args, **kwargs)
            
            def warning(self, msg, *args, **kwargs):
                if self.isEnabledFor(logging.WARNING):
                    self._log(logging.WARNING, msg, args, **kwargs)
            
            def error(self, msg, *args, **kwargs):
                if self.isEnabledFor(logging.ERROR):
                    self._log(logging.ERROR, msg, args, **kwargs)
            
            def critical(self, msg, *args, **kwargs):
                if self.isEnabledFor(logging.CRITICAL):
                    self._log(logging.CRITICAL, msg, args, **kwargs)
    except ImportError:
        # Final fallback - use standard logging
        import logging
        
        class EasyAutoMLLogger(logging.Logger):
            """Minimal logger implementation"""
            
            def __init__(self, **kwargs):
                super().__init__('EasyAutoMLLogger')
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
                self.addHandler(handler)
                self.setLevel(logging.DEBUG)
