"""Small cross-platform subset of ``fcntl.flock`` used by local coordination."""

import errno
import os
import time

if os.name == "nt":
    import msvcrt

    LOCK_SH = 1
    LOCK_EX = 2
    LOCK_NB = 4
    LOCK_UN = 8

    _CONTENTION_ERRNOS = {
        errno.EACCES,
        errno.EAGAIN,
        errno.EDEADLK,
        errno.EWOULDBLOCK,
    }

    def flock(handle, operation: int) -> None:
        """Lock byte zero; Windows shared requests serialize exclusively for safety."""
        if operation == LOCK_UN:
            _lock_byte(handle, msvcrt.LK_UNLCK)
            return

        if operation & ~(LOCK_SH | LOCK_EX | LOCK_NB) or bool(operation & LOCK_SH) == bool(
            operation & LOCK_EX
        ):
            raise OSError(errno.EINVAL, "invalid file lock operation")

        nonblocking = bool(operation & LOCK_NB)
        while True:
            try:
                _lock_byte(handle, msvcrt.LK_NBLCK)
                return
            except OSError as error:
                if error.errno not in _CONTENTION_ERRNOS:
                    raise
                if nonblocking:
                    raise BlockingIOError(
                        error.errno or errno.EACCES, "file lock is already held"
                    ) from error
                time.sleep(0.05)

    def _lock_byte(handle, mode: int) -> None:
        position = handle.tell()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), mode, 1)
        finally:
            handle.seek(position)
else:
    import fcntl as _fcntl

    LOCK_SH = _fcntl.LOCK_SH
    LOCK_EX = _fcntl.LOCK_EX
    LOCK_NB = _fcntl.LOCK_NB
    LOCK_UN = _fcntl.LOCK_UN

    def flock(handle, operation: int) -> None:
        """Delegate to POSIX flock without changing its semantics."""
        _fcntl.flock(handle, operation)
