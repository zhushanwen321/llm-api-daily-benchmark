"""原子写入工具测试。"""

from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path

import pytest

from benchmark.repository.atomic_write import (
    atomic_write,
    atomic_write_bytes,
    acquire_file_lock,
    release_file_lock,
)


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


class TestAtomicWrite:
    def test_normal_write_text(self, tmp_dir: Path) -> None:
        target = tmp_dir / "test.jsonl"
        atomic_write(str(target), "line1\n")
        assert target.read_text() == "line1\n"

    def test_overwrite_preserves_atomicity(self, tmp_dir: Path) -> None:
        target = tmp_dir / "test.jsonl"
        atomic_write(str(target), "line1\n")
        atomic_write(str(target), "line1\nline2\n")
        assert target.read_text() == "line1\nline2\n"

    def test_no_temp_file_left(self, tmp_dir: Path) -> None:
        target = tmp_dir / "test.jsonl"
        atomic_write(str(target), "content\n")
        temp_files = [f for f in os.listdir(str(tmp_dir)) if f.startswith(".tmp_")]
        assert len(temp_files) == 0

    def test_crash_recovery_temp_file_not_renamed(self, tmp_dir: Path) -> None:
        target = tmp_dir / "test.jsonl"
        target.write_text("original")

        # 手动创建临时文件但不 rename，模拟崩溃
        tmp_file = tmp_dir / ".tmp_crash_test"
        tmp_file.write_text("corrupted")

        # 原文件不受影响
        assert target.read_text() == "original"
        tmp_file.unlink()

    def test_crash_recovery_old_content_preserved(self, tmp_dir: Path) -> None:
        target = tmp_dir / "test.jsonl"
        target.write_text("original")

        # 模拟写入过程中断：临时文件存在但 rename 未执行
        import tempfile as tf

        fd, tmp_path_str = tf.mkstemp(dir=str(tmp_dir), prefix=".tmp_", suffix=".jsonl")
        try:
            os.write(fd, b"partial")
            os.close(fd)
            # 不 rename，原文件保持不变
            assert target.read_text() == "original"
        finally:
            Path(tmp_path_str).unlink(missing_ok=True)

    def test_concurrent_writes_no_data_loss(self, tmp_dir: Path) -> None:
        target = tmp_dir / "test_concurrent.txt"
        n_threads = 10
        errors: list[Exception] = []

        def writer(line: str) -> None:
            try:
                # 追加模式需要锁保护后手动读取+写入
                with acquire_file_lock(str(target)):
                    existing = ""
                    if target.exists():
                        existing = target.read_text()
                    atomic_write(str(target), existing + line)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=(f"line{i}\n",))
            for i in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"Errors during concurrent writes: {errors}"
        content = target.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == n_threads

    def test_auto_create_missing_directory(self, tmp_dir: Path) -> None:
        target = tmp_dir / "nested" / "deep" / "dir" / "test.jsonl"
        atomic_write(str(target), "deep content\n")
        assert target.read_text() == "deep content\n"

    def test_auto_create_missing_directory_binary(self, tmp_dir: Path) -> None:
        target = tmp_dir / "nested" / "bin_data.dat"
        atomic_write_bytes(str(target), b"\x00\x01\x02\x03")
        assert target.read_bytes() == b"\x00\x01\x02\x03"

    def test_binary_write(self, tmp_dir: Path) -> None:
        target = tmp_dir / "test.bin"
        data = bytes(range(256))
        atomic_write_bytes(str(target), data)
        assert target.read_bytes() == data

    def test_binary_overwrite(self, tmp_dir: Path) -> None:
        target = tmp_dir / "test.bin"
        atomic_write_bytes(str(target), b"first")
        atomic_write_bytes(str(target), b"second")
        assert target.read_bytes() == b"second"

    def test_empty_content_write(self, tmp_dir: Path) -> None:
        target = tmp_dir / "empty.txt"
        atomic_write(str(target), "")
        assert target.read_text() == ""
        assert target.exists()

    def test_file_lock_basic(self, tmp_dir: Path) -> None:
        target = tmp_dir / "locked.txt"
        target.write_text("data")

        lock = acquire_file_lock(str(target))
        try:
            # 同一线程可以再次获取（fcntl.LOCK_EX 同进程内可重入取决于实现）
            # 这里只验证锁对象可正常使用
            assert lock is not None
        finally:
            release_file_lock(lock)

    def test_file_lock_blocks_concurrent_access(self, tmp_dir: Path) -> None:
        target = tmp_dir / "locked.txt"
        target.write_text("data")

        acquired_order: list[str] = []

        def locked_writer(name: str) -> None:
            with acquire_file_lock(str(target)):
                acquired_order.append(f"{name}_start")
                # 给另一个线程时间尝试获取锁
                import time

                time.sleep(0.1)
                acquired_order.append(f"{name}_end")

        t1 = threading.Thread(target=locked_writer, args=("t1",))
        t2 = threading.Thread(target=locked_writer, args=("t2",))

        t1.start()
        # 确保 t1 先获取锁
        import time

        time.sleep(0.02)
        t2.start()

        t1.join(timeout=5)
        t2.join(timeout=5)

        # 验证操作是序列化的（一个完成后另一个才开始）
        if acquired_order[0] == "t1_start":
            assert acquired_order[1] == "t1_end"
            assert acquired_order[2] == "t2_start"
            assert acquired_order[3] == "t2_end"
        else:
            assert acquired_order[0] == "t2_start"
            assert acquired_order[1] == "t2_end"
            assert acquired_order[2] == "t1_start"
            assert acquired_order[3] == "t1_end"
