import logging
CRITICAL = 50
FATAL = CRITICAL
ERROR = 40
WARNING = 30
WARN = WARNING
INFO = 20
DEBUG = 10
NOTSET = 0

__LEVEL = DEBUG


def basicConfig(**kwargs):
    try:
        level = kwargs.get('level', DEBUG)
        if not (level == CRITICAL or level == ERROR or level == WARNING or level == INFO or level == DEBUG or level == INFO or level == NOTSET):
            raise Exception('log level set wrong')
        else:
            __LEVEL = level
    except Exception as e:
        raise Exception(e)


def _log(msg, level):
    if level >= __LEVEL:
        print(msg)

def critical(msg):
    _log(msg, CRITICAL)

def error(msg):
    _log(msg, ERROR)

def warning(msg):
    _log(msg, WARNING)

def info(msg):
    _log(msg, INFO)

def debug(msg):
    _log(msg, DEBUG)
