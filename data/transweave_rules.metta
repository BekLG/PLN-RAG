;; Phase 6: TransWeave & PLN Bridge Rules
;; These rules instruct the PeTTaChainer how to traverse SimilarityLinks.
;; When a query asks for a fact about $x, but the knowledge base only has
;; a fact about $y, the chainer can legally substitute $y for $x IF there
;; is a SimilarityLink between them. The final TruthValue is multiplied
;; by the SimilarityLink's confidence, inherently discounting analogies.

(: SimilarityTransportRule (Implication (Premises (SimilarityLink $x $y) (Inheritance $x $z)) (Conclusions (Inheritance $y $z))) (STV 1.0 1.0))
(: SimilarityTransportRuleRev (Implication (Premises (SimilarityLink $x $y) (Inheritance $z $x)) (Conclusions (Inheritance $z $y))) (STV 1.0 1.0))
(: SimilarityTransportRelation (Implication (Premises (SimilarityLink $x $y) (Evaluation $rel (List $x $z))) (Conclusions (Evaluation $rel (List $y $z)))) (STV 1.0 1.0))
(: SimilarityTransportRelationRev (Implication (Premises (SimilarityLink $x $y) (Evaluation $rel (List $z $x))) (Conclusions (Evaluation $rel (List $z $y)))) (STV 1.0 1.0))
(: SimilarityTransportRuleIsA (Implication (Premises (SimilarityLink $x $y) (IsA $x $z)) (Conclusions (IsA $y $z))) (STV 1.0 1.0))
(: SimilarityTransportRuleIsARev (Implication (Premises (SimilarityLink $x $y) (IsA $z $x)) (Conclusions (IsA $z $y))) (STV 1.0 1.0))
(: SimilarityTransportUnary (Implication (Premises (SimilarityLink $x $y) ($P $x)) (Conclusions ($P $y))) (STV 1.0 1.0))
(: SimilarityTransportRuleHealthy (Implication (Premises (SimilarityLink $x $y) (Healthy $x)) (Conclusions (Healthy $y))) (STV 1.0 1.0))
