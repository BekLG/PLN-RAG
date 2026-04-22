import json
import os
import subprocess
import sys
from typing import Any

from config import get_settings
from core.reasoner import Reasoner
from storage.vector_store import VectorStore


class ConceptNetManager:
    def __init__(self):
        self._cfg = get_settings()

    @property
    def enabled(self) -> bool:
        return self._cfg.conceptnet_enabled

    def ensure_loaded(self, reasoner: Reasoner, vector_store: VectorStore):
        if not self.enabled:
            return

        regenerated = self._ensure_artifacts_current()
        if regenerated:
            vector_store.delete_by_source("conceptnet")

        if self._cfg.conceptnet_autoload:
            self.ensure_atomspace_loaded(reasoner)
        if self._cfg.conceptnet_index_on_startup:
            self.ensure_vector_index(vector_store, force=regenerated)

    def ensure_atomspace_loaded(self, reasoner: Reasoner):
        path = self._cfg.conceptnet_atomspace_path
        if not os.path.exists(path):
            self._handle_missing(f"ConceptNet atomspace file not found: {path}")
            return
        reasoner.load_background_file(path)

    def ensure_vector_index(self, vector_store: VectorStore, force: bool = False):
        payload_path = self._cfg.conceptnet_vector_payload_path
        if not os.path.exists(payload_path):
            self._handle_missing(f"ConceptNet vector payload file not found: {payload_path}")
            return

        existing = vector_store.count_by_source("conceptnet")
        if existing > 0 and not force:
            return

        batch: list[dict[str, Any]] = []
        for record in self._iter_records(payload_path):
            batch.append(record)
            if len(batch) >= 100:
                vector_store.store_many(batch)
                batch = []
        if batch:
            vector_store.store_many(batch)

    def restore_after_reset(self, reasoner: Reasoner, vector_store: VectorStore, scope: str):
        if not self.enabled:
            return
        if scope in ("all", "atomspace") and self._cfg.conceptnet_autoload:
            self.ensure_atomspace_loaded(reasoner)
        if (
            scope in ("all", "vectordb")
            and self._cfg.conceptnet_index_on_startup
            and self._cfg.conceptnet_reindex_on_reset
        ):
            self.ensure_vector_index(vector_store, force=True)

    def _iter_records(self, path: str):
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if "nl" not in record or "pln" not in record:
                    continue
                yield record

    def _ensure_artifacts_current(self) -> bool:
        if not self._cfg.conceptnet_auto_rebuild_on_change:
            return False

        required_paths = [
            self._cfg.conceptnet_input_file,
            self._cfg.conceptnet_atomspace_path,
            self._cfg.conceptnet_vector_payload_path,
            self._cfg.conceptnet_manifest_path,
        ]
        if not all(os.path.exists(path) for path in required_paths):
            self._rebuild_artifacts()
            return True

        manifest = self._load_manifest()
        if manifest is None or self._manifest_mismatch(manifest):
            self._rebuild_artifacts()
            return True
        return False

    def _load_manifest(self) -> dict[str, Any] | None:
        path = self._cfg.conceptnet_manifest_path
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _manifest_mismatch(self, manifest: dict[str, Any]) -> bool:
        expected = {
            "source_file": self._normalize_path(self._cfg.conceptnet_input_file),
            "min_weight": float(self._cfg.conceptnet_min_weight),
            "coverage_percent": float(self._cfg.conceptnet_coverage_percent),
            "sample_seed": int(self._cfg.conceptnet_sample_seed),
        }
        for key, value in expected.items():
            current = manifest.get(key)
            if key == "source_file":
                current = self._normalize_path(current)
            if current != value:
                print(
                    f"[ConceptNet] Config change detected for {key}: "
                    f"manifest={manifest.get(key)!r}, current={value!r}"
                )
                return True
        return False

    def _normalize_path(self, path: Any) -> str:
        if not path:
            return ""
        return os.path.abspath(str(path))

    def _rebuild_artifacts(self):
        input_dir = os.path.dirname(self._cfg.conceptnet_atomspace_path)
        os.makedirs(input_dir, exist_ok=True)
        command = [
            sys.executable,
            "scripts/conceptnet/export_conceptnet.py",
            "--input",
            self._cfg.conceptnet_input_file,
            "--atomspace-output",
            self._cfg.conceptnet_atomspace_path,
            "--vector-output",
            self._cfg.conceptnet_vector_payload_path,
            "--manifest-output",
            self._cfg.conceptnet_manifest_path,
            "--min-weight",
            str(self._cfg.conceptnet_min_weight),
            "--coverage-percent",
            str(self._cfg.conceptnet_coverage_percent),
            "--sample-seed",
            str(self._cfg.conceptnet_sample_seed),
        ]
        print("[ConceptNet] Rebuilding background artifacts...")
        subprocess.run(command, check=True)
        print("[ConceptNet] Background artifacts rebuilt.")

    def _handle_missing(self, message: str):
        if self._cfg.conceptnet_startup_fail_open:
            print(f"[ConceptNet] Warning: {message}")
            return
        raise FileNotFoundError(message)
