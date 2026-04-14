import os
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
        self._lock = threading.Lock()
        self._handler = PeTTaChainer()
        self._load_from_disk()

    def _load_from_disk(self):
        if not os.path.exists(self._atomspace_path):
            os.makedirs(os.path.dirname(self._atomspace_path), exist_ok=True)
            return
        print(f"[Reasoner] Loading atomspace from {self._atomspace_path}...")
        with open(self._atomspace_path, "r", encoding="utf-8") as f:
            for line in f:
                atom = line.strip()
                if atom:
                    try:
                        self._handler.add_atom(atom)
                    except Exception as e:
                        print(f"[Reasoner] Warning: skipping atom '{atom}': {e}")
        print("[Reasoner] Atomspace loaded.")

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
        PeTTaChainer already runs reasoning in a subprocess with timeout.
        """
        try:
            result = self._handler.query(pln_query)
            return result if result else []
        except Exception as e:
            print(f"[Reasoner] Query failed for '{pln_query}': {e}")
            return []

    def reset(self):
        """Clear the in-memory atomspace and wipe the persistence file."""
        with self._lock:
            self._handler = PeTTaChainer()
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
