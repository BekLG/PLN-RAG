from typing import List
from core.transweave import Weave

class PLNBridgeGenerator:
    
    def generate_bridges(self, weave: Weave) -> List[str]:
        """
        Takes a Weave object and emits PLN atoms.
        """
        atoms = []
        
        # Calculate confidence based on the linear decay proposed in the MVP plan
        # Confidence = max(0.01, 1.0 - (Cost * 0.5))
        confidence = max(0.01, 1.0 - (weave.cost * 0.5))
        
        # Format the truth value string
        # For simplicity, we just use a generic strength/confidence TV format
        tv_str = f"(TruthValue {confidence:.2f} 0.90)" # Using 0.90 as a fixed weight/count for now
        
        # Generate SimilarityLinks for mapped entities
        for e_a, e_b in weave.entity_map.items():
            atoms.append(f"(SimilarityLink {e_a} {e_b} {tv_str})")
            
        # Optional: Emit the ContextLink encapsulation
        # This tells the reasoner that statements in SB are true within the context of SA
        atoms.append(f"(ContextLink {weave.sa_id} {weave.sb_id} {tv_str})")
        
        return atoms
