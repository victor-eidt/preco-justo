# Architecture

## One item's path

1. **Collect.** `sources/comprasgov.py` walks a catalogue group (70 = IT
   equipment) and pulls, for each catalogue code, up to 100 recorded
   purchases: description, quantity, brand, buyer, city, date, and the unit
   price paid. `sources/pncp.py` walks tender listings and keeps the object
   text and the estimated total. Both append as they go and resume from
   what is on disk.
2. **Ask.** `features.py` sends each description, with quantity, unit and
   the tender object, to the LLM together with all 19 questions from
   `questions.py`. The answer is cached on disk under
   `fingerprint(questions) + "|" + state`.
3. **Flatten.** Each typed answer becomes columns (`features.to_features`):

   | Type | Columns | Meaning |
   |---|---|---|
   | `Score` (9 feature questions) | `<q>__nivel`, `<q>__disp` | expected level on the scale; standard deviation of the distribution |
   | `Noul` (8 questions) | `<q>__p` | probability the statement holds |
   | `Choice` (1 question) | `cat__<option>` ×9, `cat__conf` | probability per category; confidence |
   | direct price (`faixa_preco`) | `faixa_preco__nivel`, `faixa_preco__disp`, `direct__log10`, `direct__conf` | the LLM's own guess as a band; **never a feature** |

   That is 40 columns, 36 of them features. Deterministic columns (description length, digit
   count, largest number, one-hot sphere and modality) are added in
   `ladder.deterministic_features`.
4. **Split.** `splits.holdout_split` locks 30% of the *groups* (a group is a
   tender or a Compras.gov purchase id). `splits.group_folds` does 5-fold
   CV on the rest, also by group.
5. **Climb.** `ladder.build_ladder` builds nine rungs. Each is a
   `(train_idx, test_idx) -> predictions` callable, so dev CV and the final
   holdout run share one code path.
6. **Audit (designed, not built).** Apply the model to PNCP estimates,
   rank by `log10(estimate) - prediction`, model the auction discount so
   the ranking is centred, and cut the queue by prediction-interval width
   (the only signal that predicted tail risk in the pilot).

## Why the spread column matters

A `Score` answer is a distribution over levels, not a point. The pipeline
keeps both moments. Four `__disp` columns landed in the top 12 permutation
importances in the pilot: when the LLM is unsure how large or how long a
contract is, that uncertainty is itself a price signal (vague descriptions
belong to a different population of purchases). Note the distinction the
pilot forced: `__disp` works as an *attribute*, not as a *confidence*
signal for abstention. Abstaining on high `__disp` was worse than random.

## Why the ladder shares one code path for dev and final

The original script cross-validated *inside* the holdout when asked for the
final number, which trains on holdout rows. Rewriting every rung as
`fit_predict(train_idx, test_idx)` makes "fit on all of dev, predict the
holdout" and "5-fold on dev" the same call with different index arrays.
There is nothing the final mode can do that dev mode cannot test.

## Where the numbers live

| Path | Contents | In git |
|---|---|---|
| `data/pilot/dataset.jsonl` | 682 IT tenders, pilot input | yes |
| `data/pilot/features.jsonl` | same rows + `jev` feature dict | yes |
| `data/pilot/results_dev.csv` | the recorded pilot ladder | yes |
| `data/raw/comprasgov_grupo70.jsonl` | item-level paid prices, growing | no |
| `data/raw/pncp_{ti,medicamentos,combustiveis}.jsonl` | tender listings, three sectors | no |
| `data/raw/jev_cache.json` | every LLM answer ever received | no |
