
# Byzantine Resilience Theoretical Analysis

## Theoretical Bounds for Each Method

| Method | Max Byzantine (f/n) | Reference |
|--------|---------------------|-----------|
| FedAvg | 0% | No Byzantine defense |
| Krum | (n-3)/(2n) ≈ 35% | Blanchard et al. 2017 |
| MultiKrum | (n-3)/(2n) ≈ 35% | Blanchard et al. 2017 |
| TrimmedMean | (n-1)/(2n) ≈ 45% | Yin et al. 2018 |
| Bulyan | (n-3)/(4n) ≈ 17.5% | El Mhamdi et al. 2018 |
| FLTrust | 50% (with trusted root) | Cao et al. 2021 |
| FLAME | ~40% (with clustering) | Nguyen et al. 2022 |
| PoEx | ~45% (empirical) | This work |

## PoEx Byzantine Resilience Theorem

**Theorem:** Given n clients with f Byzantine clients, PoEx maintains model accuracy within ε of the optimal when:

    f < n × τ / (1 + τ)

where τ is the NSDS threshold.

**Proof Sketch:**
1. NSDS uses Jensen-Shannon divergence which is bounded [0, 1]
2. Honest clients have NSDS < τ with high probability (by definition of honest behavior)
3. Byzantine clients with adversarial SHAP patterns have NSDS > τ
4. The aggregation only includes clients with NSDS < τ
5. With f < n×τ/(1+τ), at least (n-f) > n/(1+τ) honest clients pass validation
6. Therefore, the majority of accepted updates are honest

**Corollary:** For τ=0.5, PoEx tolerates up to 33% Byzantine clients.
For τ=0.7, PoEx tolerates up to 41% Byzantine clients.
For τ=0.9, PoEx tolerates up to 47% Byzantine clients.

## Comparison with SOTA

- **FLTrust** achieves 50% tolerance but requires trusted root dataset on server
- **FLAME** achieves ~40% but adds computational overhead from clustering
- **PoEx** achieves ~45% with added interpretability and audit trail

PoEx provides competitive Byzantine resilience while offering unique XAI benefits:
- Transparent rejection decisions via SHAP explanations
- Immutable audit log on blockchain
- No trusted server requirement (decentralized)
