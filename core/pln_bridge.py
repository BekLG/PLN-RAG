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
        tv_str = f"(STV {confidence:.2f} 0.90)" # Using 0.90 as a fixed weight/count for now
        
        # Generate SimilarityLinks for mapped entities
        for i, (e_a, e_b) in enumerate(weave.entity_map.items()):
            atoms.append(f"(: sim_link_{weave.id}_{i} (SimilarityLink {e_a} {e_b}) {tv_str})")
            
        # Optional: Emit the ContextLink encapsulation
        atoms.append(f"(: ctx_link_{weave.id} (ContextLink {weave.sa_id} {weave.sb_id}) {tv_str})")
        
        return atoms
