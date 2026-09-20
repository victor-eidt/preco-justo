"""Tender listings from PNCP, three sectors filled in one polite pass.

PNCP rate-limits hard (429 within seconds of a second process). So: one
process, one request at a time, seconds apart, and every listing row is
tested against ALL sector filters so three datasets fill from one crawl.

Rows are tender-level: the free-text ``objetoCompra`` and the ESTIMATED
total. This is the audit side of the system, and it was also the pilot's
training set while the item endpoint was down (see docs/experiments.md).

Resumable: tenders already on disk are skipped.

Usage::

    python -m precojusto.sources.pncp --months 4-9 --year 2026
"""

import argparse
import json
import os
import sys
import time

from ..sectors import SECTORS, matches
from .http import get_json

BASE = "https://pncp.gov.br/api/consulta/v1/contratacoes/publicacao"
MODALITIES = (6, 8, 4, 9)  # pregão eletrônico, dispensa, concorrência eletrônica, inexigibilidade
VALUE_RANGE = (1000.0, 20_000_000.0)


def _log(msg):
    print(msg, file=sys.stderr, flush=True)


def to_row(t, sector):
    obj = " ".join((t.get("objetoCompra") or "").split())
    val, key = t.get("valorTotalEstimado"), t.get("numeroControlePNCP")
    if not obj or not val or not key or len(obj) < 40:
        return None
    if not (VALUE_RANGE[0] <= float(val) <= VALUE_RANGE[1]):
        return None
    if not matches(sector, obj):
        return None
    org = t.get("orgaoEntidade") or {}
    return {"tender": key, "item_desc": obj[:700], "valor_unit": float(val),
            "orgao": org.get("razaoSocial"), "esfera": org.get("esferaId"),
            "poder": org.get("poderId"),
            "uf": (t.get("unidadeOrgao") or {}).get("ufSigla"),
            "modalidade": t.get("modalidadeNome"), "objeto": "",
            "quantidade": 1.0, "unidade": None, "unidade_n": "na",
            "tipo": "Licitacao", "setor": sector}


def windows(year, months):
    for m in months:
        for a, b in (("01", "10"), ("11", "20"), ("21", "28")):
            yield f"{year}{m:02d}{a}", f"{year}{m:02d}{b}"


def crawl(out_dir, year, months, gap, pages=30, sectors=SECTORS):
    os.makedirs(out_dir, exist_ok=True)
    paths = {s: os.path.join(out_dir, f"pncp_{s}.jsonl") for s in sectors}
    seen = {s: set() for s in sectors}
    for s, p in paths.items():
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        seen[s].add(json.loads(line)["tender"])
                    except (ValueError, KeyError):
                        pass
    _log("resuming: " + ", ".join(f"{s}={len(v)}" for s, v in seen.items()))
    files = {s: open(p, "a", encoding="utf-8") for s, p in paths.items()}

    try:
        for d0, d1 in windows(year, months):
            for mod in MODALITIES:
                for page in range(1, pages + 1):
                    d = get_json(f"{BASE}?dataInicial={d0}&dataFinal={d1}"
                                 f"&codigoModalidadeContratacao={mod}"
                                 f"&pagina={page}&tamanhoPagina=50", tries=7,
                                 on_wait=lambda w, e: _log(f"  wait {w}s ({e})"))
                    time.sleep(gap)
                    data = (d or {}).get("data") or []
                    if not data:
                        break
                    for t in data:
                        for s in sectors:
                            row = to_row(t, s)
                            if row and row["tender"] not in seen[s]:
                                seen[s].add(row["tender"])
                                files[s].write(json.dumps(row, ensure_ascii=False) + "\n")
                    for f in files.values():
                        f.flush()
                    if page >= (d.get("totalPaginas") or 1):
                        break
                _log(f"{d0}-{d1} mod{mod}: " +
                     ", ".join(f"{s}={len(v)}" for s, v in seen.items()))
    finally:
        for f in files.values():
            f.close()
    return {s: len(v) for s, v in seen.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Crawl PNCP tender listings for all sectors.")
    ap.add_argument("--out-dir", default="data/raw")
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--months", default="4-9", help="inclusive range, e.g. 4-9")
    ap.add_argument("--gap", type=float, default=5.0, help="seconds between requests")
    ap.add_argument("--pages", type=int, default=30, help="max pages per window/modality")
    args = ap.parse_args(argv)
    lo, hi = (int(x) for x in args.months.split("-"))
    crawl(args.out_dir, args.year, range(lo, hi + 1), args.gap, args.pages)


if __name__ == "__main__":
    main()
