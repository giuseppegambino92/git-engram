from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from git_engram.engine import Engram
from git_engram.evals.fixture import FixtureInfo, build_shopfront, populate_memory


@dataclass
class IndexedFixture:
    info: FixtureInfo
    engine: Engram
    failure_ids: dict[str, int]


@pytest.fixture(scope="session")
def shopfront(tmp_path_factory: pytest.TempPathFactory) -> IndexedFixture:
    """The evaluation monorepo, fully indexed, with graveyard + invariants. Treat as read-only."""
    root = tmp_path_factory.mktemp("shopfront") / "repo"
    info = build_shopfront(root)
    engine = Engram(root)
    engine.index()
    ids = populate_memory(engine, info)
    yield IndexedFixture(info, engine, ids)
    engine.close()


class MiniRepo:
    """Tiny git repository builder with deterministic commit dates."""

    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.email", "dev@example.com")
        self.git("config", "user.name", "Dev")
        self.git("config", "commit.gpgsign", "false")
        self.day = 0

    def git(self, *args: str) -> str:
        proc = subprocess.run(["git", "-C", str(self.root), *args], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        return proc.stdout.strip()

    def write(self, path: str, content: str) -> None:
        p = self.root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

    def commit(self, message: str) -> str:
        self.day += 1
        stamp = f"2025-02-{self.day:02d}T12:00:00+0000"
        env = {**os.environ, "GIT_AUTHOR_DATE": stamp, "GIT_COMMITTER_DATE": stamp}
        subprocess.run(["git", "-C", str(self.root), "add", "-A"], check=True)
        subprocess.run(["git", "-C", str(self.root), "commit", "-q", "-m", message], check=True, env=env)
        return self.git("rev-parse", "HEAD")


@pytest.fixture
def mini_repo(tmp_path: Path) -> MiniRepo:
    return MiniRepo(tmp_path / "mini")
