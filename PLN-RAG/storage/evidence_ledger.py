from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator

from core.evidence.records import ClaimEvidenceRecord, SCHEMA_VERSION, stable_id


class EvidenceLedger:
    """Authoritative local store for documents, evidence, and validated claims."""

    def __init__(self, path: str):
        self.path = str(path)
        self._lock = threading.RLock()
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL UNIQUE,
                    text TEXT NOT NULL,
                    schema_version INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS evidence_units (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    start_pos INTEGER NOT NULL,
                    end_pos INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    coreference_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(document_id, start_pos, end_pos)
                );

                CREATE TABLE IF NOT EXISTS claims (
                    id TEXT PRIMARY KEY,
                    atom TEXT NOT NULL UNIQUE,
                    schema_version INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS claim_evidence (
                    id TEXT PRIMARY KEY,
                    claim_id TEXT NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
                    evidence_id TEXT NOT NULL REFERENCES evidence_units(id) ON DELETE CASCADE,
                    original_text TEXT NOT NULL,
                    retrieval_text TEXT NOT NULL,
                    source_start INTEGER NOT NULL,
                    source_end INTEGER NOT NULL,
                    global_source_start INTEGER NOT NULL,
                    global_source_end INTEGER NOT NULL,
                    validation_state TEXT NOT NULL,
                    validation_errors_json TEXT NOT NULL,
                    lineage_json TEXT NOT NULL,
                    targets_json TEXT NOT NULL,
                    coreference_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS index_outbox (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    destination TEXT NOT NULL,
                    record_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL,
                    UNIQUE(destination, record_id)
                );

                CREATE INDEX IF NOT EXISTS idx_claim_evidence_state
                    ON claim_evidence(validation_state);
                CREATE INDEX IF NOT EXISTS idx_outbox_pending
                    ON index_outbox(destination, status, id);
                """
            )
            connection.commit()

    def record_document(self, text: str) -> str:
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        document_id = stable_id("document", content_hash)
        now = _now()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO documents
                    (id, content_hash, text, schema_version, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (document_id, content_hash, text, SCHEMA_VERSION, now),
            )
            connection.commit()
        return document_id

    def record_evidence(
        self,
        *,
        document_id: str,
        chunk_index: int,
        start_pos: int,
        end_pos: int,
        text: str,
        coreference: dict[str, Any] | None = None,
    ) -> str:
        evidence_id = stable_id(
            "evidence",
            document_id,
            start_pos,
            end_pos,
            text,
        )
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO evidence_units
                    (id, document_id, chunk_index, start_pos, end_pos, text,
                     coreference_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evidence_id,
                    document_id,
                    int(chunk_index),
                    int(start_pos),
                    int(end_pos),
                    text,
                    _json(coreference or {}),
                    _now(),
                ),
            )
            connection.commit()
        return evidence_id

    def record_claims(self, records: Iterable[ClaimEvidenceRecord]) -> None:
        rows = list(records)
        if not rows:
            return
        now = _now()
        with self._lock, self._connect() as connection:
            with connection:
                for record in rows:
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO claims
                            (id, atom, schema_version, created_at)
                        VALUES (?, ?, ?, ?)
                        """,
                        (
                            record.claim_id,
                            record.atom,
                            record.schema_version,
                            now,
                        ),
                    )
                    connection.execute(
                        """
                        INSERT INTO claim_evidence
                            (id, claim_id, evidence_id, original_text,
                             retrieval_text, source_start, source_end,
                             global_source_start, global_source_end,
                             validation_state, validation_errors_json,
                             lineage_json, targets_json, coreference_json,
                             created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(id) DO UPDATE SET
                            original_text = excluded.original_text,
                            retrieval_text = excluded.retrieval_text,
                            validation_state = excluded.validation_state,
                            validation_errors_json = excluded.validation_errors_json,
                            lineage_json = excluded.lineage_json,
                            targets_json = excluded.targets_json,
                            coreference_json = excluded.coreference_json
                        """,
                        (
                            record.claim_evidence_id,
                            record.claim_id,
                            record.evidence_id,
                            record.original_text,
                            record.retrieval_text,
                            record.source_start,
                            record.source_end,
                            record.global_source_start,
                            record.global_source_end,
                            record.validation_state,
                            _json(list(record.validation_errors)),
                            _json(record.lineage),
                            _json([target.to_dict() for target in record.targets]),
                            _json(record.coreference),
                            now,
                        ),
                    )
                    for index_record in record.qdrant_records():
                        connection.execute(
                            """
                            INSERT INTO index_outbox
                                (destination, record_id, payload_json, status,
                                 attempts, last_error, updated_at)
                            VALUES ('qdrant', ?, ?, 'pending', 0, '', ?)
                            ON CONFLICT(destination, record_id) DO UPDATE SET
                                payload_json = excluded.payload_json,
                                status = CASE
                                    WHEN index_outbox.payload_json = excluded.payload_json
                                    THEN index_outbox.status
                                    ELSE 'pending'
                                END,
                                updated_at = excluded.updated_at
                            """,
                            (index_record["index_id"], _json(index_record), now),
                        )

    def pending_index_records(
        self,
        destination: str = "qdrant",
        limit: int = 256,
    ) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, payload_json
                FROM index_outbox
                WHERE destination = ? AND status IN ('pending', 'failed')
                ORDER BY id
                LIMIT ?
                """,
                (destination, max(1, int(limit))),
            ).fetchall()
        return [
            {"outbox_id": int(row["id"]), **json.loads(row["payload_json"])}
            for row in rows
        ]

    def mark_indexed(self, outbox_ids: Iterable[int]) -> None:
        ids = [int(value) for value in outbox_ids]
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        with self._lock, self._connect() as connection:
            connection.execute(
                f"""
                UPDATE index_outbox
                SET status = 'indexed', attempts = attempts + 1,
                    last_error = '', updated_at = ?
                WHERE id IN ({placeholders})
                """,
                (_now(), *ids),
            )
            connection.commit()

    def mark_index_failed(self, outbox_ids: Iterable[int], error: str) -> None:
        ids = [int(value) for value in outbox_ids]
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        with self._lock, self._connect() as connection:
            connection.execute(
                f"""
                UPDATE index_outbox
                SET status = 'failed', attempts = attempts + 1,
                    last_error = ?, updated_at = ?
                WHERE id IN ({placeholders})
                """,
                (str(error)[:1000], _now(), *ids),
            )
            connection.commit()

    def requeue_indexes(self, destination: str = "qdrant") -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE index_outbox
                SET status = 'pending', last_error = '', updated_at = ?
                WHERE destination = ?
                """,
                (_now(), destination),
            )
            connection.commit()

    def reasoner_records(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT c.atom, ce.original_text, ce.source_start, ce.source_end,
                       ce.global_source_start, ce.global_source_end,
                       ce.validation_state, ce.lineage_json, ce.evidence_id,
                       ce.claim_id, ce.id AS claim_evidence_id
                FROM claims c
                JOIN claim_evidence ce ON ce.claim_id = c.id
                WHERE ce.validation_state IN ('accepted', 'derived')
                ORDER BY CASE ce.validation_state WHEN 'accepted' THEN 0 ELSE 1 END,
                         ce.created_at, ce.id
                """
            ).fetchall()
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in rows:
            atom = str(row["atom"])
            if atom in seen:
                continue
            seen.add(atom)
            result.append(
                {
                    "atom": atom,
                    "source": {
                        "text": row["original_text"],
                        "char_interval": {
                            "start_pos": row["source_start"],
                            "end_pos": row["source_end"],
                        },
                        "global_char_interval": {
                            "start_pos": row["global_source_start"],
                            "end_pos": row["global_source_end"],
                        },
                        "validation_state": row["validation_state"],
                        "lineage": json.loads(row["lineage_json"]),
                        "evidence_id": row["evidence_id"],
                        "claim_id": row["claim_id"],
                        "claim_evidence_id": row["claim_evidence_id"],
                    },
                }
            )
        return result

    def reset(self) -> None:
        with self._lock, self._connect() as connection:
            with connection:
                connection.execute("DELETE FROM index_outbox")
                connection.execute("DELETE FROM claim_evidence")
                connection.execute("DELETE FROM claims")
                connection.execute("DELETE FROM evidence_units")
                connection.execute("DELETE FROM documents")

    @property
    def claim_count(self) -> int:
        return self._count("claims")

    @property
    def evidence_count(self) -> int:
        return self._count("evidence_units")

    @property
    def document_count(self) -> int:
        return self._count("documents")

    @property
    def pending_count(self) -> int:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM index_outbox
                WHERE status IN ('pending', 'failed')
                """
            ).fetchone()
        return int(row["count"] if row else 0)

    def _count(self, table: str) -> int:
        if table not in {"documents", "evidence_units", "claims"}:
            raise ValueError("unsupported ledger table")
        with self._connect() as connection:
            row = connection.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()
        return int(row["count"] if row else 0)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
