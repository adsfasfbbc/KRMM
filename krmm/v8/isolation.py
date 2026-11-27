"""Filesystem isolation for the Oracle writer on the project Linux host."""
import ctypes
import errno
import os
from pathlib import Path

LANDLOCK_CREATE, LANDLOCK_ADD, LANDLOCK_RESTRICT = 444, 445, 446
READ_FILE = 1 << 2
READ_DIR = 1 << 3
EXECUTE = 1 << 0
WRITE_FILE = 1 << 1
READ = READ_FILE | READ_DIR | EXECUTE
WRITE = (1 << 13) - 1


class PathRule(ctypes.Structure):
    _pack_ = 1
    _fields_ = (("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int))


def restrict(read_paths, writable_paths, device_paths):
    libc = ctypes.CDLL(None, use_errno=True)

    def syscall(number, *args):
        status = libc.syscall(number, *args)
        if status < 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code))
        return status

    abi = syscall(LANDLOCK_CREATE, 0, 0, 1)
    if abi < 1:
        raise RuntimeError("Landlock unavailable")
    mask = ctypes.c_uint64(WRITE)
    descriptor = syscall(LANDLOCK_CREATE, ctypes.byref(mask), ctypes.sizeof(mask), 0)
    try:
        for paths, access in ((read_paths, READ), (writable_paths, WRITE), (device_paths, READ | WRITE_FILE)):
            for path in paths:
                path = Path(path)
                if not path.exists():
                    continue
                fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
                try:
                    rights = access if path.is_dir() else access & (READ_FILE | WRITE_FILE | EXECUTE)
                    rule = PathRule(rights, fd)
                    syscall(LANDLOCK_ADD, descriptor, 1, ctypes.byref(rule), 0)
                finally:
                    os.close(fd)
        if libc.prctl(38, 1, 0, 0, 0):
            raise RuntimeError("Unable to set no_new_privileges")
        syscall(LANDLOCK_RESTRICT, descriptor, 0)
    finally:
        os.close(descriptor)
    return abi


def require_denied(path):
    try:
        with Path(path).open("rb"):
            pass
    except OSError as error:
        if error.errno == errno.EACCES:
            return
        raise
    raise RuntimeError(f"Evaluator-only path readable: {path.name}")
