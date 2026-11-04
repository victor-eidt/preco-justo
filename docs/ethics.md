# Ethics and limitations

A system that marks public purchases "to review" points, sooner or later, at
people: the civil servant who wrote the estimate, the agency that signed it.
This page sets out who can be hurt and how, what the design does about it,
and what was measured rather than assumed. Numbers come from the audit
queue (1,429 PNCP items, 407 tenders) and the paid-price holdout (901 rows,
scored once). Reproduce the bias table with `python -m precojusto.model bias`.

## 1. Who bears the errors

The two errors do not cost the same.

- **A false positive** casts suspicion on an estimate that was fine, and
  indirectly on whoever wrote it.
- **A false negative** lets an inflated estimate through. Other controls can
  still catch it (the bidding itself, the courts of accounts, the press).

So the design leans toward silence over accusation:

- an estimate is flagged only when it clears the **top** of the 80% range by
  1.5x (`MARGIN`), because estimates sit above paid prices as a rule;
- the system **abstains** whenever it has no basis to judge. It abstains on
  1,319 of 1,429 items and says why in each case;
- the label is "revisar" (review), never "sobrepreço" or "fraude". A price
  gap is a reason to look, not a finding. Freight, small quantities,
  warranty terms and urgency all raise a legitimate estimate.

The output is a queue for a person, not a decision. Nothing is published,
sent or blocked automatically.

## 2. Privacy and the LGPD

The LGPD **does apply**. Public procurement data is public, but some of it
is personal data: supplier names include sole traders and MEIs, who are
natural persons. A flag also points indirectly at the servant who made the
estimate.

Art. 7, §3 allows processing personal data made public, provided the
processing respects the purpose and the public interest that justified
publishing it. Checking public spending is that purpose. Within it, the
project keeps only what it needs:

- **Supplier names are dropped at ingestion** (`items.py`, `aggregate_paid`).
  They never reach the model, the LLM, the app, or the committed data.
- **The external LLM receives only the item**: its description, unit,
  quantity and material/service type (`features.py`, `state_for`). It never
  receives the agency, the tender, a supplier, or a person's name.
- **No ranking of people or agencies.** The queue is ordered by item and by
  the size of the price gap. It does not score buyers.

## 3. Data security

- Only open data: the PNCP and Compras.gov.br public APIs, no login, no
  scraping behind access controls.
- The API key lives outside the repository (`TYPESAFE_API_KEY` or
  `~/.typesafe_key`) and is never logged.
- Raw crawls, the model file, the queue and the LLM answer cache are
  gitignored. The data committed under `data/items/` contains no supplier
  names.

## 4. Transparency

Every verdict carries its evidence, in the app's item sheet:

- the predicted range, with the estimate drawn on it;
- the **similar purchases already paid**, each with its price, agency, city
  and date, so a reader can check the comparison by hand;
- how the LLM read the description (category, specificity, technical tier,
  brand, certification), each answer with its probability;
- for an abstention, the reason in plain words.

The method is public too: the code, the question schema, every experiment
including the failed ones (`docs/experiments.md`), and the decisions
(`docs/decisions.md`). The holdout was scored once, after the model was
fixed, so the reported accuracy was not tuned on it.

## 5. Bias, measured

The worry from the proposal: small municipalities write shorter, vaguer
descriptions, so their items might be flagged more. Measured on the audit
queue:

| Sphere | Items | Judged | Flagged, of judged | No similar paid purchase |
|---|---|---|---|---|
| municipal | 1,050 | 47 (4%) | 8 (17%) | 29% |
| federal | 173 | 45 (26%) | 9 (20%) | 6% |
| state | 170 | 18 (11%) | 3 (17%) | 21% |

The model learned from paid purchases that are 52% federal, 25% state and
23% municipal. The audit side is 73% municipal.

What this shows:

- **The bias appears as blindness, not suspicion.** Among items the system
  does judge, the flag rate is about the same in every sphere (17 to 20%).
  But it judges only 4% of municipal items against 26% of federal ones,
  mostly because there is no comparable paid purchase or the description is
  too generic. Where most public buying happens, the system protects least.
- **Errors cluster.** The 20 flags fall in 10 tenders, and the top 3 hold
  11. Seven of the 8 municipal flags are toner cartridges from two small
  buyers. For the four Canon toners (GPR-53) this is a known false positive:
  the paid catalogue code mixes original cartridges (around R$ 1,400) with
  compatible ones (around R$ 40). The three Brother TN-419 flags show the same
  pattern (paid neighbours at R$ 19 to 92), likely for the same reason, not
  verified. One catalogue quirk can put four alerts on one small buyer.

The counts are small (47, 45, 18 judged items), so the rates are indicative,
not conclusive. They are the right thing to keep measuring.

## 6. The LLM's limits

LLMs make things up. Here the LLM never gives the verdict:

- it answers 19 **typed** questions, each a probability over a small scale,
  and those answers become columns for a model trained on prices actually
  paid. The model learns how much each answer is worth;
- the LLM's own price guess is kept only as a baseline. On the holdout it
  is worse than the model: RMSE 0.483 against 0.351 (log10), and within 2x
  of the paid price for 53% of items against 69%.

## 7. Limitations

- **Scope:** IT equipment priced per unit. Services, licences and lots are
  out of scope and are abstained on.
- **Free text is unstable.** The model was trained on catalogue-style
  descriptions. A hand-written description of the same product can predict
  a different price.
- **The range misses 17% of the time** by construction (an 80% range
  covered 82.8% of held-out paid prices).
- **Catalogue codes mix products**, such as originals and compatibles, and
  the model inherits that confusion.
- **The paid price is not the "right" price**, only the price someone paid in
  an auction. An agency that paid too much also feeds the training data.
- **Not validated with auditors.** Whether a flagged item is worth a
  reviewer's time has not been tested with the people who would do the
  reviewing.

## 8. Before any real use

1. Validate the flags with auditors (for example a court of accounts team)
   and measure precision on their verdicts, not on ours.
2. Group flags by tender, so one catalogue quirk reads as one problem and
   not four accusations.
3. Ask the LLM whether an item is original or compatible, and compare it
   only with purchases of the same kind.
4. Add municipal paid purchases to the training data before extending the
   system to the buyers it is blind to today.
