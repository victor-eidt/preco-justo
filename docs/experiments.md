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

---

# Item level: paid unit prices (2026-11-04)

The re-measurement the notes above promised. Target = log10 of the unit
price actually **paid** (Compras.gov.br, group 70, IT equipment, 2024–2026).

**Data.** 85 972 purchases → cleaning (`items.clean_paid`, reported, not
silent): 3 532 dropped for a unit other than `UN`, 1 680 outside
R$ 5–300 000, 3 964 more than 10x from their catalogue code's median (mostly
lot totals typed as unit prices; see the dirty-outlier notes). 76 796 kept,
aggregated to 11 077 (code, description) rows with the median paid price.
3 000 of the 5 604 rows with at least two purchases were sampled (seed 7)
and read by the LLM: 0 failures, ~0.2 s per item.

**Split.** Grouped by **catalogue code**: the model is always scored on
catalogue items it never saw, which is the situation of a PNCP line with no
code. 70/30 with seed 12345 → dev 2 099 rows, holdout 901. Rung 2
(catalogue median) therefore always falls back to the global median: by
construction there is no code to look up, and that is the point.

## 1. The ladder (dev CV and the holdout, scored once)

| Rung | dev RMSE | holdout RMSE | holdout within 2x |
|---|---|---|---|
| 1. mean | 0.877 | 0.833 | 23.8% |
| 2. catalogue median (no code seen) | 0.877 | 0.834 | 23.9% |
| 3. deterministic + GB | 0.673 | 0.629 | 43.1% |
| 4. TF-IDF + GB | 0.529 | 0.477 | 53.6% |
| 5. LLM direct, no model | 0.506 | 0.483 | 52.7% |
| 6. LLM features + det (RF) | 0.454 | 0.412 | 60.7% |
| 7. LLM features + det (GB) | 0.458 | 0.417 | 60.2% |
| 8. LLM features + det (MLP) | 8.125* | 0.455 | 55.2% |
| 9. LLM + TF-IDF + det (GB) | 0.459 | 0.405 | 61.0% |
| 10a. 7 + similar paid purchases, strict | 0.417 | 0.389 | 64.7% |
| **10b. 7 + similar paid purchases (app model)** | **0.394** | **0.351** | **69.0%** |

\* one fold produced an extreme prediction (overflow); MAE was 0.548. The MLP
is there to show the deep model does not win on 36 columns and ~2k rows;
on the holdout it lands between TF-IDF and the shallow models.

The pilot's hypothesis holds on the new target: LLM features + a shallow
model beat both the model without the LLM (rungs 3–4) and the LLM without
the model (rung 5).

**Rung 10 adds five columns** from a text search over all 11 077 paid rows:
weighted median/mean/spread of the 10 most similar paid prices, the top
neighbour's price and its similarity. On training rows the search must not
see the row's own price. Two ways to hide it, both reported:

- **10a, strict:** hide every purchase of the row's catalogue code.
- **10b, app model:** hide only the row's own (code, description) entry;
  other descriptions of the same catalogue item stay visible.

10b is the honest model for auditing: a PNCP line is compared against the
whole paid database, where its real catalogue item, under another wording,
usually is. 10a was tried first, and it broke the demo: having never seen
a near-identical neighbour in training, the model ignored one at audit time
(a 48-port 25 Gb SFP28 switch predicted at R$ 4 138 while its paid twin,
similarity 0.99, cost R$ 116 000).

## 2. The range and the abstention

Two extra quantile models (10% and 90%), widened by conformal calibration on
out-of-fold errors, give an 80% range. The widest 20% of dev ranges set the
abstention cut (12.5x wide).

| | dev CV (10b) | holdout (10b) |
|---|---|---|
| paid price inside the range | 80.1% | 82.8% |
| median range width | 7.9x | 7.4x |
| answered (range under the cut) | 80.0% | 84.6% |
| within 2x, answered only | 71.4% | 73.8% |

## 3. The audit queue (PNCP, estimated prices)

1 213 IT-equipment tenders listed → 1 605 material items from the item
endpoint → 1 429 after dedup and the price range, read by the LLM.

Verdict rules (`model.verdict`), fixed before the queue was read item by
item: **needs review** when the estimate exceeds the top of the range by
1.5x (the auction discount: estimates sit above paid prices as a rule);
**abstain** when the item is not IT equipment (LLM category), the
description is too generic (LLM specificity < 1.5; every catalogue row
reads ≥ 2), the price is not per unit, the range is wider than the cut, or
no paid purchase reaches similarity 0.3.

Result: **20 need review, 90 within the expected range, 1 319 abstain.**
The abstention is large on purpose: most lines in "IT" tenders are not IT
equipment (fuses, folders, conduit boxes) or are worded nothing like the
catalogue. Each gate was added after seeing a wrong verdict it would have
stopped:

- generic text ("Microcomputador", "Notebook, conforme Termo de
  Referência") was predicted at ~R$ 250, far outside anything in training;
- with no similar paid purchase, a 21.5" monitor came out at R$ 3 895 and a
  Samsung smartphone at R$ 153.

**Known false positives among the 20:** Canon GPR-53 *original* toners at
R$ 1 440 are flagged because the catalogue code mixes originals with
compatibles at ~R$ 40 (see the dirty-outlier notes). The flag is a request
to look, and the sheet shows the evidence that makes a person discard it.

Reproduce: `python -m precojusto.model train`, `... final` (done once),
`... queue`. Numbers: `data/items/results_*.csv`.
