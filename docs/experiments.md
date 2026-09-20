# Experiment log (pilot)

Everything below was measured on the **pilot target**: the estimated
*total* of an IT tender, from PNCP, predicted from the tender's object text.
The project has since moved to *paid unit prices* per item (ADR-13). The
methodological findings stand; the absolute numbers will be re-measured.

Data: 682 IT tenders (Apr–Sep 2026), 70/30 dev/holdout with seed 12345.
The 204 holdout rows have never been scored. Sectors declared up front:
IT (682), medicines (81), fuels (58); the last two are below the crossover
below and were not modelled.

## 1. The ladder (dev, 5-fold CV, n=478)

| Rung | RMSE | MAE | within 2x | within 5x |
|---|---|---|---|---|
| 1. mean | 0.982 | 0.810 | 22.4% | 49.8% |
| 3. deterministic + GB | 0.820 | 0.646 | 30.5% | 60.7% |
| 4. TF-IDF + GB | 0.859 | 0.676 | 28.5% | 61.1% |
| 5. LLM direct, no model | 0.791 | 0.634 | 27.4% | 61.5% |
| 6. LLM features + det (RF) | 0.667 | 0.511 | 36.6% | 73.6% |
| **7. LLM features + det (GB)** | **0.664** | **0.510** | **41.4%** | 73.2% |
| 8. LLM features + det (MLP) | 1.031 | 0.798 | 25.9% | 52.5% |
| 9. LLM + TF-IDF + det (GB) | 0.663 | 0.507 | 39.5% | 73.4% |

Rungs 6–9 exclude the direct-price columns, so the comparison with rung 5
is clean. Reproduce with `python -m precojusto.ladder`.

## 2. Learning curve (the main finding)

| train n | pipeline (7) | LLM alone (5) | gain |
|---|---|---|---|
| 40 | 0.679 | 0.634 | −0.044 |
| 80 | 0.632 | 0.634 | +0.002 |
| 160 | 0.563 | 0.634 | +0.071 |
| 240 | 0.556 | 0.634 | +0.079 |
| 320 | 0.540 | 0.634 | +0.094 |
| 382 | 0.528 | 0.634 | +0.106 |

Crossover at n≈80. Below it, training a model is worse than asking the LLM.
Above it the advantage grows and had not saturated. A first pilot with
n=76 was negative; it sat just under the crossover, and this was predicted
before being measured.

## 3. False positives (open problem)

Residual: median −0.014, sd 0.658. Detection of *injected* overpricing at a
1% false-positive budget:

| Method | detects 10x | detects 50x |
|---|---|---|
| global residual | 14.3% | 45.4% |
| per-category z-score (clustering) | 12.6% | 44.5% |
| above the 90% quantile | 16.0% | 43.7% |

Three attempts, no gain. Recorded on purpose.

## 4. Selective prediction (what worked)

Correlation of candidate uncertainty signals with |error|:

| Signal | corr |
|---|---|
| `escala_quantidade__disp` | +0.068 |
| `duracao__disp` | +0.041 |
| `especificidade__disp` | −0.098 |
| 10–90% interval width | +0.011 |

None predicts the *typical* error. But interval width predicts *tail* risk:

| Coverage | MAE | within 2x | FP per 5 days @50x | if random |
|---|---|---|---|---|
| 30% | 0.487 | 38.5% | ~13 | ~27 |
| 50% | 0.478 | 41.8% | ~26 | ~46 |
| 70% | 0.490 | 40.4% | ~39 | ~64 |
| 100% | 0.504 | 40.8% | ~92 | ~92 |

MAE is flat across the curve while false positives halve. Predicting typical
error and predicting tail risk are different problems; an anomaly queue only
needs the second. Abstaining on the LLM's semantic uncertainty (`__disp`)
was worse than random.

## 5. What the LLM features do

Permutation importance, GB on dev, top 12:

```
+0.302  mod_Pregão - Eletrônico       +0.075  duracao__disp
+0.137  escala_quantidade__disp       +0.070  cat__outro
+0.135  escopo__nivel                 +0.061  abrangencia__nivel
+0.114  duracao__nivel                +0.058  abrangencia__disp
+0.104  mod_Concorrência - Eletrônica +0.058  padronizacao__disp
                                      +0.052  mod_Dispensa
```

Four `__disp` columns in the top 12, in every run.

## 6. Mistakes made and fixed

1. `faixa_preco__nivel` was inside rung 7's matrix, so rung 7 secretly
   contained rung 5. Fixed by excluding `faixa_preco__*` and `direct__*`.
2. The cache fingerprint was prefixed to the *state* sent to the model, not
   only to the cache key. The LLM was reading a hash. Fixed.
3. Unit-price bands were applied to tender totals, penalising rung 5
   unfairly. Fixed with total-value bands at the time; bands are unit-price
   again now that the target is unit price.
4. The IT filter on the tender object alone let in lock repairs and hotel
   nights. Fixed with an item-level filter.
5. (Rewrite) `--final` cross-validated inside the holdout. Fixed: it fits
   on dev and predicts the holdout.
6. (Rewrite) Splits were by row, while the docs said "by purchase". With
   one row per tender this made no difference in the pilot; it would have
   on item-level data. Fixed with group-aware splits.

## 7. First item-level observation

Catalogue code 481548 (all-in-one desktop): 100 purchases, median
R$ 4 765, min R$ 290, max R$ 49 900 (10.5x the median). The LLM's direct
guess for one of them was R$ 3 266 against R$ 4 899 paid: a 1.5x miss,
versus ~3x typical at tender level. Early sign that unit prices are more
predictable, as expected.
