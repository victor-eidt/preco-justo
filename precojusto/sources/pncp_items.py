"""Item-level ESTIMATED prices from PNCP: the number the system audits.

Lists tenders whose object looks like IT, then fetches each tender's items
with ``valorUnitarioEstimado``. This is the endpoint that was returning
502/503 for the whole pilot, which is why training moved to Compras.gov.br.
Kept because it is the audit side of the design; re-test it periodically.

Usage::

    python -m precojusto.sources.pncp_items 20260901 20260910 --out data/raw/pncp_items.jsonl
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from ..sectors import matches
from .http import get_json

BASE = "https://pncp.gov.br/api"
MODALITIES = (6, 8)  # pregão eletrônico and dispensa carry most IT buying


def _log(msg):
    print(msg, file=sys.stderr, flush=True)


def find_tenders(d0, d1, pages, sector="ti"):
    hits = []
    for mod in MODALITIES:
        for page in range(1, pages + 1):
            d = get_json(f"{BASE}/consulta/v1/contratacoes/publicacao"
                         f"?dataInicial={d0}&dataFinal={d1}"
                         f"&codigoModalidadeContratacao={mod}"
                         f"&pagina={page}&tamanhoPagina=50", tries=3, timeout=45)
            if d is None:
                _log(f"    throttled at modality {mod} page {page}; backing off")
                time.sleep(10)
                continue
            if not d.get("data"):
                break
            hits += [t for t in d["data"] if matches(sector, t.get("objetoCompra") or "")]
            if page >= (d.get("totalPaginas") or 1):
                break
        _log(f"  modality {mod}: {len(hits)} tenders so far")
    return hits


def fetch_items(t):
    org = t.get("orgaoEntidade") or {}
    cnpj, ano, seq = org.get("cnpj"), t.get("anoCompra"), t.get("sequencialCompra")
    if not (cnpj and ano and seq):
        return []
    d = get_json(f"{BASE}/pncp/v1/orgaos/{cnpj}/compras/{ano}/{seq}/itens"
                 f"?pagina=1&tamanhoPagina=50", tries=3, timeout=45)
    items = d if isinstance(d, list) else (d or {}).get("data") or []
    out = []
    for it in items:
        desc = (it.get("descricao") or "").strip()
        price = it.get("valorUnitarioEstimado")
        if not desc or not price or price <= 0 or len(desc) < 12:
            continue
        out.append({
            "tender": t.get("numeroControlePNCP"),
            "orgao": org.get("razaoSocial"),
            "esfera": org.get("esferaId"),
            "poder": org.get("poderId"),
            "uf": (t.get("unidadeOrgao") or {}).get("ufSigla"),
            "modalidade": t.get("modalidadeNome"),
            "objeto": (t.get("objetoCompra") or "")[:300],
            "item_desc": desc,
            "quantidade": it.get("quantidade"),
            "unidade": it.get("unidadeMedida"),
            "tipo": it.get("materialOuServicoNome"),
            "valor_unit": float(price),
        })
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Pull item-level estimated prices from PNCP.")
    ap.add_argument("date_from")
    ap.add_argument("date_to")
    ap.add_argument("--out", default="data/raw/pncp_items.jsonl")
    ap.add_argument("--pages", type=int, default=12)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)

    tenders = find_tenders(args.date_from, args.date_to, args.pages)
    seen, uniq = set(), []
    for t in tenders:
        k = t.get("numeroControlePNCP")
        if k and k not in seen:
            seen.add(k)
            uniq.append(t)
    _log(f"{len(uniq)} IT tenders; fetching items...")

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, got in enumerate(ex.map(fetch_items, uniq), 1):
            rows.extend(got)
            if i % 25 == 0:
                _log(f"  {i}/{len(uniq)} tenders -> {len(rows)} items")
    with open(args.out, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    _log(f"APPENDED {args.out}: {len(rows)} items from {len(uniq)} tenders")


if __name__ == "__main__":
    main()
