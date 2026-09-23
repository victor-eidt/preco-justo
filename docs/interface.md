# Interface plan

Status: **planned, not started** (2026-09-23). Target: a working demo by
early November 2026.

Today the project is code and logs. This is the plan for the surface a
person actually uses, and the reasoning behind it.

## Positioning: a triage inbox, not a chatbot

A chatbot or a detector answers what someone asks. Preço Justo reads the
tenders nobody asked about and points at the few that deserve a look. The
interface should make that visible: it looks like an inbox or a ticket
queue, not a chat box. Competing on "has a chat window" would hide the one
thing the project does that the others do not.

## Four views

**1. The queue (home).** The tenders published in a period, already read.
A header line sets the scale: "1,240 items read, 7 need review, 38 without
enough confidence". Each row: description, buyer, city, estimated price,
predicted range, ratio. Sorted by divergence; only rows that pass the
confidence gate are flagged.

**2. The item sheet.** Explains the flag instead of only stating it:

```
Microcomputador All in One, SSD 310–500 GB, garantia on site 12 meses
Pref. Municipal de X · pregão eletrônico · 13 unidades

  estimated  R$ 49.900 ●
  predicted  R$ 3.900 ├──────■──────┤ R$ 6.800
             └─ 8.4x above the range → NEEDS REVIEW

  How the LLM read it (Jev)            What drove the price
  categoria      hardware   1.00       e_hardware        ▲▲▲
  especificidade 3/4 full spec         tier_tecnico      ▲▲
  e_hardware                0.96       escala            ▼
  ...

  Similar purchases already paid (Compras.gov.br)
  R$ 4.899 · Dona Francisca/RS · 2026   R$ 4.650 · …   R$ 5.120 · …
```

- The Jev answers show the LLM at work: named columns, readable values.
- "What drove the price" shows the model at work: TreeSHAP on the GB model.
- The similar paid purchases are the most intuitive element on the page:
  anyone understands "other agencies paid this". Find them by distance in
  the same 36 LLM feature columns, so the explanation matches the model;
  fall back to the catalogue code when there is one.

**3. "Try it yourself".** Paste any description and a price, get the item
sheet in seconds. This is the live moment of the demo: someone else types
an item. It shows input → processing → output working instead of narrated.

**4. "How we know it works".** The ablation ladder, the learning-curve
crossover, flag rate by government sphere, calibration. Charts, not
tables. This is what separates a measured system from a demo.

## Decisions

- **Ethics shows up in the UI.** The label is "needs review", never
  "fraud", and not alarm red. There is an explicit "not confident enough"
  state where the system abstains. The uncertainty range is always drawn.
  Supplier names are not prominent. Each choice can be pointed at on
  screen.
- **Optional: a plain-language note.** A short Portuguese paragraph
  generated from the numbers already computed ("read as standard hardware;
  similar purchases cost R$ 3.900–6.800; the estimate is 8.4x above"). Hard
  rule: it only restates facts that are on the screen, never adds any.
- **The demo never depends on PNCP being up.** The API has gone down more
  than once. The queue is computed by a daily batch job into a database;
  the pages only read from it. Only "try it yourself" runs live, and it
  needs only the LLM and the trained model.
- **Stack.** A small Python API (FastAPI) around the model, a real web
  front end (Next.js), a simple database. Not Streamlit: faster to build,
  but it looks like every other course project, which is what this is
  trying not to be.

## Scope and order

Solo, about six weeks, with ML work still open (the item-level rerun on
paid prices, the auction discount). So: no login, no accounts, three screens
plus the evidence page.

1. Queue and item sheet on the pilot data. Works with no new data.
2. "Try it yourself".
3. Evidence page, filled in as the ladder results on item data arrive.

The final presentation is built around this app: the app carries the demo,
and the slides only frame it.
