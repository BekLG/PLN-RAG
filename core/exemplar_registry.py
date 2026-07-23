from typing import List, Dict, Tuple
from core.senf import SENF, SENFExemplar, SENFEntity

class ExemplarScorer:
    """
    For each entity-kind pair, score distances to a small exemplar set.
    """
    
    # Small exemplar registry for common and domain-relevant kinds:
    REGISTRY: Dict[str, List[str]] = {
        "Camera": ["professional_camera", "consumer_camera", "phone_camera", "security_camera"],
        "Game": ["chess_game", "football_game", "childrens_game", "video_game"],
        "Bird": ["robin", "eagle", "penguin", "ostrich"],
        "Treatment": ["drug_treatment", "surgical_treatment", "behavioral_treatment"]
    }
    
    # Simple lexical cues for the MVP
    CUES: Dict[str, Dict[str, float]] = {
        "nikon": {"professional_camera": 0.12, "consumer_camera": 0.48},
        "lens": {"professional_camera": 0.2, "consumer_camera": 0.5},
        "strategy": {"chess_game": 0.05, "football_game": 0.80},
        "exhausting": {"football_game": 0.06, "chess_game": 0.82}
    }

    def score(self, senf: SENF, context_text: str = "") -> None:
        """
        Populate exemplars in the SENF object based on entities and context.
        """
        context_lower = context_text.lower()
        
        for ent_id, entity in senf.entities.items():
            if not entity.kind:
                continue
                
            # If we don't have exemplars for this kind, skip
            if entity.kind not in self.REGISTRY:
                continue
                
            prototypes = self.REGISTRY[entity.kind]
            
            # Simple heuristic distance calculation for the MVP
            for proto in prototypes:
                distance = 0.5 # default moderate distance
                
                # Check lexical cues in the context
                for cue, distances in self.CUES.items():
                    if cue in context_lower:
                        if proto in distances:
                            distance = distances[proto]
                            break
                            
                senf.exemplars.append(
                    SENFExemplar(
                        entity_id=ent_id,
                        kind=entity.kind,
                        prototype=proto,
                        distance=distance
                    )
                )
