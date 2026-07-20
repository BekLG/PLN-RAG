from core.senf import SENF, IdentityEdgePlus, IdentityEdgeMinus

class IdentityGraphBuilder:
    """
    Phase 3: Identity Graph.
    Builds costed positive (IdPlus) and negative (IdMinus) evidence 
    edges between entities in the same SENF batch.
    """
    
    def build_graph(self, senf: SENF) -> None:
        """
        Populate the id_plus_edges and id_minus_edges in the SENF object.
        Compares every pair of entities in the SENF.
        """
        entities = list(senf.entities.values())
        
        # O(N^2) comparison for local window
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                ent1 = entities[i]
                ent2 = entities[j]
                
                # Rule-based scoring
                self._evaluate_pair(ent1, ent2, senf)
                
    def _evaluate_pair(self, ent1, ent2, senf: SENF) -> None:
        plus_reasons = []
        minus_reasons = []
        plus_cost = 0.0
        minus_cost = 0.0
        
        # 1. Kind compatibility
        if ent1.kind and ent2.kind:
            if ent1.kind == ent2.kind:
                plus_reasons.append("same-kind")
                plus_cost += 0.2
            else:
                minus_reasons.append("incompatible-kind")
                minus_cost += 0.8
                
        # 2. Exemplar matching
        # Find exemplars for these entities if they exist
        ex1 = self._get_best_exemplar(ent1.id, senf)
        ex2 = self._get_best_exemplar(ent2.id, senf)
        
        if ex1 and ex2:
            if ex1.prototype == ex2.prototype:
                plus_reasons.append("exemplar-match")
                plus_cost += 0.1
            else:
                minus_reasons.append("exemplar-mismatch")
                minus_cost += 0.6
                
        # Only emit edges if we have substantial evidence
        if plus_reasons:
            senf.id_plus_edges.append(
                IdentityEdgePlus(ent1.id, ent2.id, plus_cost, plus_reasons)
            )
            
        if minus_reasons:
            senf.id_minus_edges.append(
                IdentityEdgeMinus(ent1.id, ent2.id, minus_cost, minus_reasons)
            )

    def _get_best_exemplar(self, entity_id: str, senf: SENF):
        exs = [ex for ex in senf.exemplars if ex.entity_id == entity_id]
        if not exs:
            return None
        return min(exs, key=lambda x: x.distance)
