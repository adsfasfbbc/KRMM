# KRMM v0.5 Evidence Support Structure Taxonomy

## 1. Eight-Class Support Taxonomy (`krmm/v5/support_structure.py`)
To eliminate the v0.3 ambiguity around intra-model redundancy, v0.5 distinguishes 8 support classes:
1. `single_decisive`: A single atomic attribute uniquely determines the target category.
2. `single_chain`: One multi-step reasoning chain determines the candidate (higher fragility).
3. `redundant_same_source`: Multiple supporting facts share high surface/semantic overlap or stem from one correlated attribute.
4. `redundant_distinct_attributes`: Multiple semantically distinct attributes within the same model independently converge on the same candidate (strictly labeled intra-model, NOT external independent sources).
5. `independent_source_support`: Reserved for verifiable distinct external data sources (`count = 0` in single-model experiments; never conflated with class 4).
6. `conflicting_support`: Extracted facts imply mutually inconsistent candidates.
7. `insufficient_support`: Extracted facts cannot uniquely determine a candidate.
8. `ambiguous_or_unverifiable`: Target entity or predicate lacks a well-defined groundable resolution.

## 2. Empirical Stratification on Brand-New v0.5 Entities (`G2-A`, `n=22`)
| Support Class | Count | Candidate Correct | Accuracy | Evidence-Gated Accepted |
|---|---:|---:|---:|---:|
| `single_decisive` | 6 | 6 | **100.0%** | 6/6 |
| `redundant_distinct_attributes` | 10 | 9 | **90.0%** | 10/10 |
| `single_chain` | 4 | 2 | 50.0% | 0/4 (gated out) |
| `conflicting_support` | 2 | 0 | 0.0% | 0/2 (gated out) |
| `independent_source_support` | 0 | 0 | N/A | 0/0 |

**Takeaway:** Support structure cleanly separates high-reliability candidates (`15/16 = 93.75%` across `single_decisive` + `redundant_distinct_attributes`) from low-reliability candidates (`2/6 = 33.3%` across `single_chain` + `conflicting_support`). Moreover, observing `9/10` (`giant anteater` misclassified despite `redundant_distinct_attributes`) confirms that intra-model redundancy is high-precision rather than infallible.
