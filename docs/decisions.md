# Decision records

Short, dated, and kept even when superseded. Newest last.

## ADR-1 · Predict in log10 space
Prices in one sector span R$ 5 to R$ 300 000. An error of R$ 500 is a
rounding error on a server and a 10x miss on a mouse. All metrics are on
log10(price); "within 2x" is reported because it is what an auditor
understands.

## ADR-2 · Ablation ladder, one claim per rung
A single accuracy number cannot say whether the LLM helped, the model
helped, or the text helped. Rungs: mean; catalogue median; deterministic
text stats; TF-IDF; LLM direct; LLM features with RF, GB and MLP;
everything. Rung 5 (LLM direct) exists so rung 7 has something honest to
beat.

## ADR-3 · The LLM answers typed questions; it does not name a price
Alternatives were sentence embeddings (opaque columns, needs more rows) or
fine-tuning (needs far more rows and gives no explanation). Typed questions
give named columns with a written rubric, work at n≈100, and let a flagged
item be explained as "the model read this as enterprise-scale and
recurring". Cost: the 19 questions are a hand-designed bottleneck.

## ADR-4 · A `Score` becomes two columns
Expected level and spread. The spread turned out to be a top feature. See
architecture.md.

## ADR-5 · The direct-price columns are excluded from every learned rung
Found the hard way: rung 7 once contained `faixa_preco__nivel` and was
therefore rung 5 plus noise. `ladder.DIRECT_COLUMNS` is the single list
that enforces this.

## ADR-6 · Locked 30% holdout, seed 12345, by group
Set aside before any model was trained. Never scored during iteration.
Split by purchase, not by row, because items of one purchase have
correlated prices. `--final` fits on dev and scores the holdout; it does
not cross-validate inside it (the pilot script did; fixed in the rewrite).

## ADR-7 · Three sectors declared up front
IT, medicines, fuels. All three are reported regardless of outcome.
Selecting the best-looking sector after the fact is p-hacking.

## ADR-8 · Keyword filters at crawl time, LLM at item time
Regexes are free and run over every listing row. A false positive costs
one LLM call later; a false negative is lost. Precision is recovered at
the item level in `sources/clean.py`.

## ADR-9 · One polite crawler, resumable, append-only
PNCP rate-limits within seconds of a second concurrent process. One
process, one request every 5 s, each page tested against all sectors.
Every puller appends and flushes per page and skips what is already on
disk, so a crash costs one page.

## ADR-10 · Cache keyed by question fingerprint + state
Editing a question must invalidate its answers, and the fingerprint must
not reach the model. An earlier version prefixed the fingerprint to the
state string itself; the LLM was reading a hash. Fixed.

## ADR-11 · Shallow models first
37 dense columns and a few hundred rows. Gradient boosting and random
forests win; the MLP rung is kept as evidence, not as a candidate.
Revisit past ~10k rows.

## ADR-12 · Question texts and column names stay in Portuguese
The rubric is read together with Portuguese descriptions. Translating it
would move the rubric away from the text it judges. Column names are the
domain's terms (`especificidade`, `escopo`, `recorrencia`).

## ADR-13 · Train on the paid price, audit the estimated one (Sep 2026)
The pilot trained on PNCP's estimated tender totals because PNCP's item
endpoint was down. A second API (Compras.gov.br) exposes the unit price
actually paid, with quantity, brand and a national catalogue code. That
became the training source, and the PNCP estimate became the audit
target. Training on the estimate is circular: a commonly inflated
estimate is predicted correctly and never flagged. Consequence: the
pilot's absolute numbers must be re-measured on the new target.

## ADR-14 · Package layout, not scripts (Sep 2026)
Six root-level scripts sharing state through the working directory became
a package with explicit paths, one HTTP helper, and one ladder code path.
The pilot numbers reproduce to the third decimal after the rewrite; the
group-aware splits degenerate to the original row-level splits when every
group has one row, and a test pins that.
