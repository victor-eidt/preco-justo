# Data

| Path | What | Committed |
|---|---|---|
| `pilot/dataset.jsonl` | 682 IT tenders from PNCP (Apr–Sep 2026): object text + estimated total | yes |
| `pilot/features.jsonl` | the same rows with a `jev` dict of 40 LLM-derived columns (36 features + 4 direct-price) | yes |
| `pilot/results_dev.csv` | the recorded pilot ladder, as produced at the time | yes |
| `raw/comprasgov_grupo70.jsonl` (+ `.done`) | paid unit prices per catalogue item, group 70 | no |
| `raw/pncp_{ti,medicamentos,combustiveis}.jsonl` | tender listings, three sectors | no |
| `raw/pncp_items_v1.jsonl` | the few item-level PNCP rows collected before the endpoint went down | no |
| `raw/jev_cache.json` | every LLM answer, keyed by question fingerprint + state | no |
| `raw/comprasgov-openapi.json` | the API's OpenAPI document, for reference | no |

The pilot files are committed so the ladder runs without an API key. Raw
files are regenerable with the pullers in `precojusto/sources/`.

## Row schema (pilot / PNCP)

```
tender        PNCP control number (group key for splits)
item_desc     free text the LLM reads
valor_unit    the target (tender total in the pilot; unit price at item level)
orgao, esfera, poder, uf, modalidade
objeto, quantidade, unidade, unidade_n, tipo, setor
jev           (features only) the 40 LLM columns
```

Item-level rows from Compras.gov.br additionally carry `idCompra` (group
key), `codigoItemCatalogo`, `marca`, `municipio`, `dataCompra` and the other
fields listed in `docs/data-sources.md`.

All content is public procurement data published by the Brazilian
government. No personal data is collected.
