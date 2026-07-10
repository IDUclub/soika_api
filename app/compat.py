from __future__ import annotations

import inspect
from collections import namedtuple


def patch_inspect_getargspec() -> None:
    if hasattr(inspect, "getargspec"):
        return

    arg_spec = namedtuple("ArgSpec", "args varargs keywords defaults")

    def getargspec(func):
        spec = inspect.getfullargspec(func)
        return arg_spec(spec.args, spec.varargs, spec.varkw, spec.defaults)

    inspect.getargspec = getargspec
