from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from core.senf import SENF, SENFEntity
from core.symbol_normalization import equivalent_symbol

@dataclass
class Weave:
    id: str
    sa_id: str
    sb_id: str
    cost: float
    distortion: float
    entity_map: Dict[str, str] = field(default_factory=dict)
    kind_map: Dict[str, str] = field(default_factory=dict)
    exemplar_map: Dict[str, str] = field(default_factory=dict)
    
    def to_metta_strings(self) -> List[str]:
        """Serialize TransWeave data to MeTTa atoms."""
        atoms = []
        atoms.append(f"(: weave_{self.id} (Weave {self.id} {self.sa_id} {self.sb_id}) (STV 1.0 1.0))")
        atoms.append(f"(: weave_cost_{self.id} (WeaveCost {self.id} {self.cost:.2f}) (STV 1.0 1.0))")
        atoms.append(f"(: weave_dist_{self.id} (WeaveDistortion {self.id} {self.distortion:.2f}) (STV 1.0 1.0))")
        
        for i, (e_a, e_b) in enumerate(self.entity_map.items()):
            atoms.append(f"(: map_entity_{self.id}_{i} (MapEntity {self.id} {e_a} {e_b}) (STV 1.0 1.0))")
            
        for i, (k_a, k_b) in enumerate(self.kind_map.items()):
            atoms.append(f"(: map_kind_{self.id}_{i} (MapKind {self.id} {k_a} {k_b}) (STV 1.0 1.0))")
            
        return atoms

class TransWeaveAligner:
    """
    Implements a single-shot alignment algorithm (Algorithm 1) to find
    structural and semantic mappings between two SENFs.
    """
    
    def build_weaves(self, senf_a: SENF, senf_b: SENF, weave_id: str = "W1", sa_id: str = "SA", sb_id: str = "SB", top_k: int = 1) -> List[Weave]:
        """
        Build top-k weaves between two SENFs. For this MVP, we implement
        a greedy single-shot matcher that prefers same-kind and close exemplars.
        """
        candidate_pairs: List[Tuple[SENFEntity, SENFEntity, float]] = []
        
        # Cross product of entities to find candidate matches
        for ent_a in senf_a.entities.values():
            for ent_b in senf_b.entities.values():
                if ent_a.id == ent_b.id:
                    continue
                cost = self._score_entity_pair(ent_a, ent_b, senf_a, senf_b)
                if cost < 1.0: # threshold to consider a match
                    candidate_pairs.append((ent_a, ent_b, cost))
                    
        # Sort by lowest cost (best match)
        candidate_pairs.sort(key=lambda x: x[2])
        
        # Greedy assignment to build one weave
        assigned_a = set()
        assigned_b = set()
        
        weave = Weave(id=weave_id, sa_id=sa_id, sb_id=sb_id, cost=0.0, distortion=0.0)
        
        for ent_a, ent_b, cost in candidate_pairs:
            if ent_a.id not in assigned_a and ent_b.id not in assigned_b:
                # Add to weave
                weave.entity_map[ent_a.id] = ent_b.id
                
                if ent_a.kind and ent_b.kind:
                    weave.kind_map[ent_a.kind] = ent_b.kind
                    
                # Add cost
                weave.cost += cost
                
                # Mark as assigned
                assigned_a.add(ent_a.id)
                assigned_b.add(ent_b.id)
                
        # Only return a weave if we mapped something
        if weave.entity_map:
            return [weave]
        return []

    def _score_entity_pair(self, ent_a: SENFEntity, ent_b: SENFEntity, senf_a: SENF, senf_b: SENF) -> float:
        return 0.0 if equivalent_symbol(ent_a.id, ent_b.id) else 1.0

    def _get_best_exemplar(self, entity_id: str, senf: SENF):
        exs = [ex for ex in senf.exemplars if ex.entity_id == entity_id]
        if not exs:
            return None
        return min(exs, key=lambda x: x.distance)
