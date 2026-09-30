from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mint_synth.runtime import ConfiguredRoleClient, SqliteResponseCache


class Delegate:
    def __init__(self, value: str):
        self.value = value
        self.calls = 0

    def complete_json(self, **kwargs):
        self.calls += 1
        return {"value": self.value}

    def stats(self):
        return {"requests": self.calls}

    def close(self):
        return None


class RuntimeCacheTests(unittest.TestCase):
    def test_immutable_sqlite_cache_survives_worker_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "responses.sqlite3"
            first_delegate = Delegate("first")
            first = ConfiguredRoleClient(
                first_delegate,
                temperature=0.0,
                max_tokens=100,
                cache=SqliteResponseCache(path, policy="immutable", busy_timeout_seconds=1),
                cache_namespace={"model": "m", "revision": "r1"},
            )
            self.assertEqual(
                first.complete_json(system="system", user="request"),
                {"value": "first"},
            )
            first.close()

            second_delegate = Delegate("second")
            second = ConfiguredRoleClient(
                second_delegate,
                temperature=0.0,
                max_tokens=100,
                cache=SqliteResponseCache(path, policy="immutable", busy_timeout_seconds=1),
                cache_namespace={"model": "m", "revision": "r1"},
            )
            self.assertEqual(
                second.complete_json(system="system", user="request"),
                {"value": "first"},
            )
            self.assertEqual(second_delegate.calls, 0)
            self.assertEqual(second.stats()["cache_hits"], 1)
            second.close()

    def test_cache_key_includes_effective_generation_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            delegate = Delegate("value")
            client = ConfiguredRoleClient(
                delegate,
                temperature=0.2,
                max_tokens=None,
                cache=SqliteResponseCache(
                    Path(temp) / "responses.sqlite3",
                    policy="immutable",
                    busy_timeout_seconds=1,
                ),
                cache_namespace={"model": "m"},
            )
            client.complete_json(system="s", user="u", max_tokens=100)
            client.complete_json(system="s", user="u", max_tokens=200)
            self.assertEqual(delegate.calls, 2)
            client.close()


if __name__ == "__main__":
    unittest.main()
