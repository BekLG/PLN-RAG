import os
import re
import threading
from typing import List
from config import get_settings

from pettachainer.pettachainer import PeTTaChainer


class Reasoner:
    """
    Owns all atomspace operations. Nothing else in the system
    calls add_atom or query directly.

    Responsibilities:
    - Load atomspace from disk on startup
    - Add statements coming from the parser
    - Execute queries and return proof traces
    - Persist new atoms to disk
    """

    def __init__(self):
        cfg = get_settings()
        self._atomspace_path = cfg.atomspace_path
        self._query_timeout = cfg.chaining_timeout
        self._lock = threading.Lock()
        self._handler = PeTTaChainer()
        self._background_files: set[str] = set()
        self._load_from_disk()

    def _load_from_disk(self):
        if not os.path.exists(self._atomspace_path):
            os.makedirs(os.path.dirname(self._atomspace_path), exist_ok=True)
            return
        print(f"[Reasoner] Loading atomspace from {self._atomspace_path}...")
        self._load_file(self._atomspace_path)
        print("[Reasoner] Atomspace loaded.")

    def _load_file(self, path: str):
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                atom = line.strip()
                if atom:
                    try:
                        self._handler.add_atom(atom)
                    except Exception as e:
                        print(f"[Reasoner] Warning: skipping atom '{atom}': {e}")

    def load_background_file(self, path: str):
        normalized = os.path.abspath(path)
        if normalized in self._background_files:
            return
        print(f"[Reasoner] Loading background atomspace from {path}...")
        self._load_file(path)
        self._background_files.add(normalized)
        print("[Reasoner] Background atomspace loaded.")

    def add_statements(self, statements: List[str]) -> List[str]:
        """
        Add parsed MeTTa statements to the atomspace and persist them.
        Returns the list of successfully added atoms.
        """
        added = []
        with self._lock:
            with open(self._atomspace_path, "a", encoding="utf-8") as f:
                for stmt in statements:
                    clean = " ".join(stmt.split())
                    try:
                        self._handler.add_atom(clean)
                        f.write(clean + "\n")
                        added.append(clean)
                    except Exception as e:
                        print(f"[Reasoner] Failed to add atom '{clean}': {e}")
        return added

    def query(self, pln_query: str) -> List[str]:
        """
        Run a PLN query and return proof traces.
        Try exact fact lookup first for grounded queries, then fall back to
        PeTTaChainer proof search with the configured timeout.
        """
        exact = self._query_exact_fact(pln_query)
        if exact:
            return exact
        try:
            result = self._handler.query(pln_query, timeout_sec=self._query_timeout)
            return result if result else []
        except Exception as e:
            print(f"[Reasoner] Query failed for '{pln_query}': {e}")
            return []

    def _query_exact_fact(self, pln_query: str) -> List[str]:
        target = self._extract_grounded_query_atom(pln_query)
        if not target:
            return []

        for path in self._fact_sources():
            match = self._find_exact_atom_in_file(path, target)
            if match:
                return [match]
        return []

    def _extract_grounded_query_atom(self, pln_query: str) -> str:
        match = re.fullmatch(r"\(:\s+[$?][^\s]+\s+(\(.+\))\s+[$?][^\s]+\)", pln_query.strip())
        if not match:
            return ""
        atom = match.group(1)
        if "$" in atom or "?" in atom:
            return ""
        return " ".join(atom.split())

    def _fact_sources(self) -> List[str]:
        paths = [self._atomspace_path, *sorted(self._background_files)]
        return [path for path in paths if os.path.exists(path)]

    def _find_exact_atom_in_file(self, path: str, target: str) -> str:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                atom = line.strip()
                if not atom:
                    continue
                body = self._extract_statement_body(atom)
                if body == target:
                    return atom
        return ""

    def _extract_statement_body(self, statement: str) -> str:
        match = re.fullmatch(
            r"\(:\s+[^\s]+\s+(\(.+\))\s+\(STV\s+[^\s]+\s+[^\s]+\)\)",
            statement.strip(),
        )
        if not match:
            return ""
        return " ".join(match.group(1).split())

    def reset(self):
        """Clear the in-memory atomspace and wipe the persistence file."""
        with self._lock:
            self._handler = PeTTaChainer()
            self._background_files = set()
            if os.path.exists(self._atomspace_path):
                os.remove(self._atomspace_path)
        print("[Reasoner] Atomspace reset.")

    @property
    def size(self) -> int:
        """Approximate atom count (line count of persistence file)."""
        if not os.path.exists(self._atomspace_path):
            return 0
        with open(self._atomspace_path) as f:
            return sum(1 for line in f if line.strip())

    @property
    def background_size(self) -> int:
        total = 0
        for path in self._background_files:
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as handle:
                total += sum(1 for line in handle if line.strip())
        return total
