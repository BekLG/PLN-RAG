from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import re

@dataclass
class SENFEntity:
    id: str
    kind: str = ""
    properties: List[str] = field(default_factory=list)

@dataclass
class SENFFrame:
    id: str
    head: str
    roles: Dict[str, str] = field(default_factory=dict) # role_name -> entity_id

@dataclass
class SENFExemplar:
    entity_id: str
    kind: str
    prototype: str
    distance: float

@dataclass
class IdentityEdge:
    e1: str
    e2: str
    cost: float
    reasons: List[str]

@dataclass
class IdentityEdgePlus(IdentityEdge):
    pass

@dataclass
class IdentityEdgeMinus(IdentityEdge):
    pass

@dataclass
class SENF:
    frames: List[SENFFrame] = field(default_factory=list)
    entities: Dict[str, SENFEntity] = field(default_factory=dict)
    exemplars: List[SENFExemplar] = field(default_factory=list)
    id_plus_edges: List[IdentityEdgePlus] = field(default_factory=list)
    id_minus_edges: List[IdentityEdgeMinus] = field(default_factory=list)
    raw_atoms: List[str] = field(default_factory=list)
    
    def to_metta_strings(self) -> List[str]:
        """Serialize SENF data back to MeTTa atoms."""
        atoms = list(self.raw_atoms) # Keep the original canonical atoms
        
        # Add exemplar data
        for ex in self.exemplars:
            atoms.append(f"(exemplar {ex.entity_id} {ex.kind} {ex.prototype} {ex.distance:.2f})")
            
        # Add nearest-ex (assuming we pick the minimum distance for now)
        if self.exemplars:
            # Group by entity
            entity_exs = {}
            for ex in self.exemplars:
                if ex.entity_id not in entity_exs:
                    entity_exs[ex.entity_id] = []
                entity_exs[ex.entity_id].append(ex)
                
            for entity_id, exs in entity_exs.items():
                best_ex = min(exs, key=lambda x: x.distance)
                atoms.append(f"(nearest-ex {entity_id} {best_ex.kind} {best_ex.prototype})")
                
        # Add Identity Edges
        for i, edge in enumerate(self.id_plus_edges):
            reasons_str = " ".join(edge.reasons)
            atoms.append(f"(: id_plus_{i} (IdPlus {edge.e1} {edge.e2} {edge.cost:.2f} (reasons {reasons_str})) (STV 1.0 1.0))")
            
        for i, edge in enumerate(self.id_minus_edges):
            reasons_str = " ".join(edge.reasons)
            atoms.append(f"(: id_minus_{i} (IdMinus {edge.e1} {edge.e2} {edge.cost:.2f} (reasons {reasons_str})) (STV 1.0 1.0))")
                
        return atoms

def build_senf_from_atoms(atoms: List[str]) -> SENF:
    senf = SENF(raw_atoms=atoms)
    
    for atom in atoms:
        # Look for (IsA <entity> <Kind>) or (Inheritance <entity> <Kind>)
        isa_match = re.search(r"\((?:IsA|Inheritance)\s+([A-Za-z0-9_]+)\s+([A-Za-z0-9_]+)\)", atom)
        if isa_match:
            ent_id = isa_match.group(1)
            kind = isa_match.group(2)
            if not ent_id.startswith("$") and not ent_id.startswith("?"): # Skip variables
                if ent_id not in senf.entities:
                    senf.entities[ent_id] = SENFEntity(id=ent_id)
                senf.entities[ent_id].kind = kind
                
        # Also extract entities from other simple binary predicates
        # like (Predicate <ent1> <ent2>) just so they exist in senf.entities
        binary_match = re.search(r"\([A-Za-z0-9_]+\s+([A-Za-z0-9_]+)\s+([A-Za-z0-9_]+)\)", atom)
        if binary_match:
            for ent_id in (binary_match.group(1), binary_match.group(2)):
                if not ent_id.startswith("$") and not ent_id.startswith("?"):
                    if ent_id not in senf.entities:
                        senf.entities[ent_id] = SENFEntity(id=ent_id)
                        
        # Extract entities from unary predicates like (Healthy football)
        unary_match = re.search(r"\([A-Z][A-Za-z0-9_]*\s+([a-z0-9_]+)\)", atom)
        if unary_match:
            ent_id = unary_match.group(1)
            if not ent_id.startswith("$") and not ent_id.startswith("?"):
                if ent_id not in senf.entities:
                    senf.entities[ent_id] = SENFEntity(id=ent_id)

    return senf
