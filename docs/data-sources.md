# Data sources and their quirks

Both APIs are public and keyless. Both are flaky in different ways.

## Compras.gov.br open data (training side: paid unit prices)

Base: `https://dadosabertos.compras.gov.br`. OpenAPI at `/v3/api-docs`.

Two calls, in `sources/comprasgov.py`:

1. `/modulo-material/4_consultarItemMaterial?codigoGrupo=70` lists the
   catalogue items of a group (70 = IT equipment, 15 142 items).
   `tamanhoPagina` has a floor of 10 and a ceiling of 500; below 10 is a 400.
2. `/modulo-pesquisa-preco/1_consultarMaterial?tipo=codigoItemCatalogo&codigo=<c>`
   returns recorded purchases for that code. `tipo` is an enum: only
   `codigoItemCatalogo` or `codigoPdm`. `tamanhoPagina=100` works, so one
   request per code is enough for the 100-purchase cap.

Useful fields per purchase: `precoUnitario`, `quantidade`,
`descricaoDetalhadaItem`, `marca`, `codigoItemCatalogo`, `nomePdm`,
`nomeClasse`, `municipio`, `estado`, `esfera`, `poder`, `nomeUasg`,
`nomeFornecedor`, `dataCompra`, `idCompra`.

`codigoItemCatalogo` groups identical items nationwide, which is what makes
the "national median per code" baseline (rung 2) possible. `idCompra` is the
group key for splits.

## PNCP (audit side: estimated prices)

`https://pncp.gov.br/api/consulta/v1/contratacoes/publicacao` works but is
unstable: 429, 500 and timeouts during testing. One process with 5 s between
requests runs clean; three in parallel were throttled immediately. Each
request returns 50 tenders, so every page is matched against all sectors
in one pass (`sources/pncp.py`).

`https://pncp.gov.br/api/pncp/v1/orgaos/{cnpj}/compras/{ano}/{seq}/itens`
gives item-level *estimated* unit prices, which is the audit target. It
returned 502/503 for the whole pilot. `sources/pncp_items.py` is kept and
should be re-tested periodically.

## Sectors

Declared in `sectors.py`: `ti`, `medicamentos`, `combustiveis`. Keyword
filters with an exclusion list each ("sistema de esgoto" is not IT).

## Resuming crawls

Every puller appends per page and skips what is already on disk, so it is
safe to restart at any time:

```sh
nohup python -u -m precojusto.sources.comprasgov --group 70 > logs/comprasgov.log 2>&1 &
nohup python -u -m precojusto.sources.pncp --year 2026 --months 4-9 > logs/pncp.log 2>&1 &
wc -l data/raw/*.jsonl
```

Processes started with `nohup` are reparented to launchd on macOS and
survive the terminal closing. Only a reboot or `pkill` stops them.
