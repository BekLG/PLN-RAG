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
class SENF:
    frames: List[SENFFrame] = field(default_factory=list)
    entities: Dict[str, SENFEntity] = field(default_factory=dict)
    exemplars: List[SENFExemplar] = field(default_factory=list)
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
                
        return atoms

def build_senf_from_atoms(atoms: List[str]) -> SENF:
    """Algorithm 3: Build SENF from canonical PLN atoms."""
    senf = SENF(raw_atoms=atoms)
    
    for atom in atoms:
        # Look for (IsA <entity> <Kind>)
        isa_match = re.search(r"\(IsA\s+([A-Za-z0-9_]+)\s+([A-Za-z0-9_]+)\)", atom)
        if isa_match:
            ent_id = isa_match.group(1)
            kind = isa_match.group(2)
            if not ent_id.startswith("$") and not ent_id.startswith("?"): # Skip variables
                if ent_id not in senf.entities:
                    senf.entities[ent_id] = SENFEntity(id=ent_id)
                senf.entities[ent_id].kind = kind
                
    return senf
