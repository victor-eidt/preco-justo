"""Item-level unit prices from the Compras.gov.br open-data API.

Gives what the training side needs: the unit price actually paid, the
quantity as its own column, the free-text description, the brand, and a
catalogue code that groups identical items across the whole country.

Two steps: list every catalogue item in a group (70 = IT equipment), then
pull the purchases recorded for each. Resumable: codes already processed
are listed in ``<out>.done`` and skipped.

API notes (learned the hard way, see docs/data-sources.md):
- ``tamanhoPagina`` has a floor of 10 and a ceiling of 500 on the catalogue
  endpoint; the price endpoint accepts at least 100.
- ``tipo`` is an enum: only ``codigoItemCatalogo`` or ``codigoPdm``.

Usage::

    python -m precojusto.sources.comprasgov --group 70
"""

import argparse
import json
import os
import sys
import time

from .http import get_json

BASE = "https://dadosabertos.compras.gov.br"

KEEP = ("descricaoDetalhadaItem", "descricaoItem", "precoUnitario", "quantidade",
        "marca", "codigoItemCatalogo", "nomePdm", "nomeClasse", "codigoClasse",
        "siglaUnidadeFornecimento", "nomeUasg", "nomeOrgao", "municipio",
        "estado", "esfera", "poder", "dataCompra", "modalidade",
        "nomeFornecedor", "idCompra")


def _log(msg):
    print(msg, file=sys.stderr, flush=True)


def catalogue(group, gap):
    """Every catalogue item in ``group``, paged 500 at a time."""
    out, page = [], 1
    while True:
        d = get_json(f"{BASE}/modulo-material/4_consultarItemMaterial"
                     f"?pagina={page}&tamanhoPagina=500&codigoGrupo={group}", timeout=70)
        time.sleep(gap)
        if not d or not d.get("resultado"):
            break
        out.extend(d["resultado"])
        total = d.get("totalRegistros") or 0
        _log(f"  catalogue: {len(out)}/{total}")
        if len(out) >= total:
            break
        page += 1
    return out


def purchases(code, date_from, date_to, gap, page_size=100, max_pages=1):
    """Purchases recorded for one catalogue code, capped at ``page_size*max_pages``."""
    out, page = [], 1
    while page <= max_pages:
        d = get_json(f"{BASE}/modulo-pesquisa-preco/1_consultarMaterial"
                     f"?pagina={page}&tamanhoPagina={page_size}&tipo=codigoItemCatalogo"
                     f"&codigo={code}&dataCompraInicio={date_from}&dataCompraFim={date_to}",
                     timeout=70)
        time.sleep(gap)
        res = (d or {}).get("resultado") or []
        if not res:
            break
        out.extend(res)
        if len(out) >= (d.get("totalRegistros") or 0):
            break
        page += 1
    return out


def to_row(c):
    price = c.get("precoUnitario")
    desc = (c.get("descricaoDetalhadaItem") or c.get("descricaoItem") or "").strip()
    if not price or float(price) <= 0 or len(desc) < 20:
        return None
    row = {k: c.get(k) for k in KEEP}
    row["item_desc"] = " ".join(desc.split())[:700]
    row["valor_unit"] = float(price)
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description="Pull paid unit prices per catalogue item.")
    ap.add_argument("--group", default="70", help="catalogue group (70 = IT equipment)")
    ap.add_argument("--out", default=None, help="default data/raw/comprasgov_grupo<G>.jsonl")
    ap.add_argument("--from", dest="date_from", default="2024-01-01")
    ap.add_argument("--to", dest="date_to", default="2026-09-01")
    ap.add_argument("--gap", type=float, default=0.6, help="seconds between requests")
    ap.add_argument("--per-code", type=int, default=100, help="max purchases per code")
    args = ap.parse_args(argv)

    out = args.out or f"data/raw/comprasgov_grupo{args.group}.jsonl"
    done_path = out + ".done"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(done_path):
        with open(done_path, encoding="utf-8") as fh:
            done = {line.strip() for line in fh if line.strip()}
    _log(f"group {args.group} · {len(done)} codes already done · output {out}")

    items = catalogue(args.group, args.gap)
    _log(f"{len(items)} catalogue items in group {args.group}")

    n = 0
    with open(out, "a", encoding="utf-8") as f, open(done_path, "a", encoding="utf-8") as fd:
        for i, it in enumerate(items, 1):
            code = str(it.get("codigoItem"))
            if code in done:
                continue
            for c in purchases(code, args.date_from, args.date_to, args.gap,
                               page_size=min(args.per_code, 100),
                               max_pages=max(1, args.per_code // 100)):
                row = to_row(c)
                if row:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n += 1
            f.flush()
            fd.write(code + "\n")
            fd.flush()
            if i % 25 == 0:
                _log(f"  {i}/{len(items)} codes · {n} purchases collected")
    _log(f"DONE: {n} purchases appended to {out}")


if __name__ == "__main__":
    main()
