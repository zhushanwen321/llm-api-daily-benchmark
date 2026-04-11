"""原子写入工具：temp + fsync + rename 实现 POSIX 原子性。

写入流程：
  1. 在目标文件同目录创建临时文件
  2. 写入内容 + fsync 确保落盘
  3. os.rename 原子替换目标文件

文件锁通过 fcntl.flock 实现 advisory lock，用于多进程/线程间的写入协调。
"""

from __future__ import annotations

import fcntl
import os
import tempfile
from pathlib import Path
from types import TracebackType
from typing import Optional, Type


class FileLock:
    """基于 fcntl.flock 的 advisory file lock。

    使用目标文件的锁文件（同名加 .lock 后缀）进行进程间协调。
    """

    def __init__(self, target_path: str, timeout: float = 30.0) -> None:
        self._lock_path = target_path + ".lock"
        self._timeout = timeout
        self._fd: Optional[int] = None

    def acquire(self) -> FileLock:
        if self._fd is not None:
            return self
        fd = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            self._fd = fd
            return self
        except Exception:
            os.close(fd)
            raise

    def release(self) -> None:
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd)
                self._fd = None

    def __enter__(self) -> FileLock:
        return self.acquire()

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.release()


def acquire_file_lock(target_path: str) -> FileLock:
    """获取文件锁并返回已锁定的 FileLock 对象。

    使用方式：
      # 手动释放
      lock = acquire_file_lock(path)
      try:
          ...
      finally:
          release_file_lock(lock)

      # 或作为 context manager
      with acquire_file_lock(path) as lock:
          ...
    """
    lock = FileLock(target_path)
    lock.acquire()
    return lock


def release_file_lock(lock: FileLock) -> None:
    lock.release()


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def atomic_write(path: str, content: str, encoding: str = "utf-8") -> None:
    """原子写入文本文件。

    Args:
        path: 目标文件路径。
        content: 要写入的文本内容。
        encoding: 文本编码，默认 utf-8。
    """
    _ensure_parent_dir(path)

    dir_name = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, prefix=".tmp_", suffix=".atomic")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def atomic_write_bytes(path: str, content: bytes) -> None:
    """原子写入二进制文件。

    Args:
        path: 目标文件路径。
        content: 要写入的二进制内容。
    """
    _ensure_parent_dir(path)

    dir_name = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, prefix=".tmp_", suffix=".atomic")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
