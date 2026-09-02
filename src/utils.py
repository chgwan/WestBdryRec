# -*- coding: utf-8 -*-
"""Shared parallel-map and POSIX publication primitives."""
from __future__ import annotations

import fcntl
import os
import pathlib
import stat
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import BinaryIO, Callable, List, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class PublicationIOError(RuntimeError):
    """A no-follow lock/read/publication contract could not be satisfied."""


def pmap(fn: Callable[[T], R], items: Sequence[T], workers: int = 1,
         desc: str = "") -> List[R]:
    """Order-preserving parallel map. ``workers`` <= 1 runs serially."""
    items = list(items)
    if not items:
        return []
    if workers is None or workers <= 1:
        return [fn(x) for x in items]
    out: List[R] = [None] * len(items)  # type: ignore[list-item]
    with ProcessPoolExecutor(max_workers=int(workers)) as ex:
        futs = {ex.submit(fn, x): i for i, x in enumerate(items)}
        for fut in as_completed(futs):
            out[futs[fut]] = fut.result()
    return out


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(bytes(payload))
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write")
        view = view[written:]


def _read_all(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


def _open_real_directory(path: pathlib.Path, *, create: bool,
                         label: str) -> int:
    path = pathlib.Path(path)
    if create and not os.path.lexists(path):
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise PublicationIOError(
                f"cannot create {label} at {path}: {exc}") from exc
    try:
        before = path.lstat()
    except OSError as exc:
        raise PublicationIOError(f"cannot open {label} at {path}: {exc}") from exc
    if not stat.S_ISDIR(before.st_mode):
        raise PublicationIOError(
            f"{label} at {path} must be a real no-follow directory")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PublicationIOError(
            f"{label} at {path} must be a real no-follow directory") from exc
    opened = os.fstat(descriptor)
    if (not stat.S_ISDIR(opened.st_mode)
            or (opened.st_dev, opened.st_ino)
            != (before.st_dev, before.st_ino)):
        os.close(descriptor)
        raise PublicationIOError(f"{label} at {path} changed while opening")
    return descriptor


def _read_leaf_at(directory_fd: int, name: str, *, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise PublicationIOError(
            f"{label} must be a regular no-follow file") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise PublicationIOError(
                f"{label} must be a regular no-follow file")
        return _read_all(descriptor)
    finally:
        os.close(descriptor)


def read_regular_nofollow(path: pathlib.Path | str, *, label: str) -> bytes:
    """Read all bytes from one stable regular file without following a symlink."""
    path = pathlib.Path(path)
    try:
        before = path.lstat()
    except OSError as exc:
        raise PublicationIOError(
            f"{label} at {path} must be a regular no-follow file") from exc
    if not stat.S_ISREG(before.st_mode):
        raise PublicationIOError(
            f"{label} at {path} must be a regular no-follow file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PublicationIOError(
            f"{label} at {path} must be a regular no-follow file") from exc
    try:
        opened = os.fstat(descriptor)
        if (not stat.S_ISREG(opened.st_mode)
                or (opened.st_dev, opened.st_ino)
                != (before.st_dev, before.st_ino)):
            raise PublicationIOError(
                f"{label} at {path} must be a stable regular no-follow file")
        return _read_all(descriptor)
    finally:
        os.close(descriptor)


class PersistentFlock:
    """A persistent no-follow file guarded by nonblocking POSIX ``flock``."""

    def __init__(self, path: pathlib.Path | str, *, shared: bool,
                 state_label: str = "publication state", mode: int = 0o600):
        self.path = pathlib.Path(path)
        self.shared = bool(shared)
        self.state_label = str(state_label)
        self.mode = int(mode)
        self.fd: int | None = None
        self.identity: tuple[int, int] | None = None

    def _path_identity(self):
        try:
            current = self.path.lstat()
        except OSError:
            return None
        if not stat.S_ISREG(current.st_mode):
            return None
        return current.st_dev, current.st_ino

    def __enter__(self):
        directory_fd = _open_real_directory(
            self.path.parent, create=True,
            label=f"{self.state_label} lock parent")
        try:
            if os.path.lexists(self.path):
                try:
                    existing = self.path.lstat()
                except OSError as exc:
                    raise PublicationIOError(
                        f"{self.path}: cannot inspect {self.state_label} lock") \
                        from exc
                if not stat.S_ISREG(existing.st_mode):
                    raise PublicationIOError(
                        f"{self.path}: {self.state_label} lock path must be a "
                        "regular no-follow file")
            flags = os.O_RDWR | os.O_CREAT | os.O_NONBLOCK
            flags |= getattr(os, "O_NOFOLLOW", 0)
            try:
                self.fd = os.open(
                    self.path.name, flags, self.mode, dir_fd=directory_fd)
            except OSError as exc:
                raise PublicationIOError(
                    f"{self.path}: {self.state_label} lock path must be a "
                    "regular no-follow file") from exc
        finally:
            os.close(directory_fd)
        opened = os.fstat(self.fd)
        if not stat.S_ISREG(opened.st_mode):
            os.close(self.fd)
            self.fd = None
            raise PublicationIOError(
                f"{self.path}: {self.state_label} lock path must be a "
                "regular no-follow file")
        self.identity = opened.st_dev, opened.st_ino
        if self._path_identity() != self.identity:
            os.close(self.fd)
            self.fd = None
            self.identity = None
            raise PublicationIOError(
                f"{self.path}: {self.state_label} lock path changed while opening")
        operation = fcntl.LOCK_SH if self.shared else fcntl.LOCK_EX
        try:
            fcntl.flock(self.fd, operation | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(self.fd)
            self.fd = None
            self.identity = None
            raise PublicationIOError(
                f"{self.path}: {self.state_label} lock is held by another "
                "process") from exc
        except OSError as exc:
            os.close(self.fd)
            self.fd = None
            self.identity = None
            raise PublicationIOError(
                f"{self.path}: cannot acquire {self.state_label} lock: {exc}") \
                from exc
        self.assert_held()
        return self

    def assert_held(self) -> None:
        if (self.fd is None or self.identity is None
                or self._path_identity() != self.identity):
            raise PublicationIOError(
                f"{self.path}: {self.state_label} lock path changed while held")

    def __exit__(self, _exc_type, _exc, _traceback):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        self.identity = None
        return False


def _publication_temp_name(destination_name: str) -> str:
    return (
        f".{destination_name}.publish.{os.getpid()}."
        f"{uuid.uuid4().hex}"
    )


def durable_publish_file(
        path: pathlib.Path | str,
        writer: Callable[[BinaryIO], None],
        *,
        validator: Callable[[bytes], object] | None = None,
        state_label: str = "publication state",
        mode: int = 0o644,
) -> str:
    """Publish one complete file by unique temp + fsync + hard-link no-replace.

    The complete temporary bytes are validated before commit and the destination
    is re-read and compared after commit. An exact pre-existing or concurrent
    destination is an ``"unchanged"`` no-op; differing state is never replaced.
    A failure after the hard link never rolls the destination back.
    """
    path = pathlib.Path(path)
    directory_fd = _open_real_directory(
        path.parent, create=True, label=f"{state_label} parent")
    temporary = _publication_temp_name(path.name)
    descriptor: int | None = None
    payload: bytes | None = None
    try:
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(
                temporary, flags, int(mode), dir_fd=directory_fd)
        except OSError as exc:
            raise PublicationIOError(
                f"cannot create unique {state_label} temporary {temporary}: "
                f"{exc}") from exc
        try:
            with os.fdopen(os.dup(descriptor), "wb") as handle:
                writer(handle)
                handle.flush()
                os.fsync(handle.fileno())
            payload = _read_all(descriptor)
            if validator is not None:
                validator(payload)
        except PublicationIOError:
            raise
        except Exception as exc:
            raise PublicationIOError(
                f"cannot write or validate {state_label} temporary: {exc}") \
                from exc
        os.close(descriptor)
        descriptor = None

        if os.path.lexists(path):
            current = _read_leaf_at(
                directory_fd, path.name, label=f"existing {state_label}")
            if current == payload:
                if validator is not None:
                    validator(current)
                return "unchanged"
            raise PublicationIOError(
                f"{path}: existing immutable {state_label} differs")

        try:
            os.link(
                temporary, path.name,
                src_dir_fd=directory_fd, dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            current = _read_leaf_at(
                directory_fd, path.name, label=f"concurrent {state_label}")
            if current == payload:
                if validator is not None:
                    validator(current)
                return "unchanged"
            raise PublicationIOError(
                f"{path}: concurrent immutable {state_label} differs")
        except OSError as exc:
            raise PublicationIOError(
                f"cannot commit {state_label} with hard link: {exc}") from exc

        try:
            os.fsync(directory_fd)
        except OSError as exc:
            raise PublicationIOError(
                f"{state_label} directory fsync failure: {exc}") from exc
        committed = _read_leaf_at(
            directory_fd, path.name, label=f"committed {state_label}")
        if committed != payload:
            raise PublicationIOError(
                f"{path}: committed {state_label} bytes changed")
        if validator is not None:
            validator(committed)
        return "created"
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        except OSError:
            # Never turn cleanup into destination rollback or overwrite.
            pass
        os.close(directory_fd)


def durable_publish_bytes(
        path: pathlib.Path | str,
        payload: bytes,
        *,
        validator: Callable[[bytes], object] | None = None,
        state_label: str = "publication state",
        mode: int = 0o644,
) -> str:
    """Publish exact bytes through :func:`durable_publish_file`."""
    expected = bytes(payload)

    def write(handle: BinaryIO) -> None:
        handle.write(expected)

    def validate(candidate: bytes) -> object:
        if candidate != expected:
            raise PublicationIOError(
                f"{state_label} temporary bytes differ from the full payload")
        return validator(candidate) if validator is not None else None

    return durable_publish_file(
        path, write, validator=validate,
        state_label=state_label, mode=mode)
