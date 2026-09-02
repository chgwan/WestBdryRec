# -*- coding: utf-8 -*-
"""Configure the NSCC Python runtime, then execute a script or module."""
from __future__ import annotations

import faulthandler
import os
import pathlib
import runpy
import sys
import types
from collections.abc import Sequence

_USAGE = "usage: nscc_python_entry.py TARGET.py [ARGS] | -m MODULE [ARGS]"


def configure_runtime() -> None:
    """Enable fail-fast diagnostics and write-through job log streams."""
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["PYTHONFAULTHANDLER"] = "1"
    faulthandler.enable(all_threads=True)
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(line_buffering=True, write_through=True)


def _set_path0(path: pathlib.Path) -> None:
    value = str(path)
    if sys.path:
        sys.path[0] = value
    else:
        sys.path.append(value)


def _run_module_as_main(module: str, args: Sequence[str]) -> None:
    _resolved_name, spec, code = runpy._get_module_details(module)
    main_module = types.ModuleType("__main__")
    sys.modules["__main__"] = main_module
    sys.argv = [spec.origin, *args]
    runpy._run_code(
        code, main_module.__dict__, mod_name="__main__", mod_spec=spec)


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``TARGET.py [ARGS]`` or ``-m MODULE [ARGS]`` after configuration."""
    restore_state = argv is not None
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        raise SystemExit(_USAGE)

    argv_object = sys.argv
    argv_state = list(sys.argv)
    path_object = sys.path
    path_state = list(sys.path)
    missing_main = object()
    main_state = sys.modules.get("__main__", missing_main)
    try:
        configure_runtime()
        if args[0] == "-m":
            if len(args) < 2 or not args[1]:
                raise SystemExit("-m requires a module name")
            module = args[1]
            _set_path0(pathlib.Path.cwd())
            _run_module_as_main(module, args[2:])
            return 0

        target = pathlib.Path(args[0])
        if not target.exists():
            raise SystemExit(f"target does not exist: {target}")
        if not target.is_file():
            raise SystemExit(f"target is not a file: {target}")
        sys.argv = [str(target), *args[1:]]
        _set_path0(target.absolute().parent)
        runpy.run_path(str(target), run_name="__main__")
        return 0
    finally:
        if restore_state:
            argv_object[:] = argv_state
            sys.argv = argv_object
            path_object[:] = path_state
            sys.path = path_object
            if main_state is missing_main:
                sys.modules.pop("__main__", None)
            else:
                sys.modules["__main__"] = main_state


if __name__ == "__main__":
    raise SystemExit(main())
