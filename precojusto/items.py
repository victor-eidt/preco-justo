"""Item-level datasets: paid prices to learn from, estimated prices to audit.

Two builders, one per side of the design:

- ``train-set``: Compras.gov.br purchases (unit price PAID) -> cleaned,
  aggregated to one row per (catalogue code, description), target = the
  median paid price of that description. Also writes the full aggregated
  table as the "similar purchases already paid" index used by the app.
- ``audit-set``: PNCP tender items (unit price ESTIMATED) for the IT
  tenders already listed in ``data/raw/pncp_ti.jsonl``.

Cleaning is reported, never silent (see ``clean_paid``). Supplier names are
dropped here: they are not needed to model a price, and some are natural
persons (MEI, sole traders), so they never reach the LLM or the app.

Usage::

    python -m precojusto.items train-set
    python -m precojusto.items audit-set
    python -m precojusto.items audit-prep
"""

import argparse
import json
import math
import os
import random
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from .sources.clean import PRICE_RANGE

OUTLIER_LOG10 = 1.0   # drop a purchase 10x above/below its code's median
MIN_CODE_N = 5        # ...only when the code has at least this many purchases


def _log(msg):
    print(msg, file=sys.stderr, flush=True)


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def _write(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- paid side

def clean_paid(rows, lo=PRICE_RANGE[0], hi=PRICE_RANGE[1]):
    """Filter Compras.gov.br purchases. Returns (kept, report Counter).

    1. unit of supply must be ``UN``: boxes, rolls and metres price a
       different quantity than the description names;
    2. unit price inside ``PRICE_RANGE``;
    3. within a catalogue code with >= MIN_CODE_N purchases, drop prices 10x
       away from the code median. These are mostly data errors (lot totals
       typed as unit prices, another product filed under the code), checked
       by hand on 2026-09-23.
    """
    rep = Counter(raw=len(rows))
    step = []
    for r in rows:
        if (r.get("siglaUnidadeFornecimento") or "").upper() != "UN":
            rep["drop: unit is not UN"] += 1
            continue
        p = float(r["valor_unit"])
        if not (lo <= p <= hi):
            rep["drop: price outside range"] += 1
            continue
        step.append(dict(r, valor_unit=p))

    by_code = defaultdict(list)
    for r in step:
        by_code[r["codigoItemCatalogo"]].append(math.log10(r["valor_unit"]))
    med = {c: st.median(v) for c, v in by_code.items() if len(v) >= MIN_CODE_N}

    kept = []
    for r in step:
        m = med.get(r["codigoItemCatalogo"])
        if m is not None and abs(math.log10(r["valor_unit"]) - m) > OUTLIER_LOG10:
            rep["drop: 10x from code median"] += 1
            continue
        kept.append(r)
    rep["kept"] = len(kept)
    return kept, rep


def aggregate_paid(rows):
    """One row per (code, description): median paid price and context."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["codigoItemCatalogo"], r["item_desc"])].append(r)
    out = []
    for (code, desc), rs in groups.items():
        prices = sorted(r["valor_unit"] for r in rs)
        qty = [float(r.get("quantidade") or 1) for r in rs]
        recent = max(rs, key=lambda r: r.get("dataCompra") or "")
        out.append({
            "group": f"cat{code}",
            "codigoItemCatalogo": code,
            "item_desc": desc,
            "valor_unit": st.median(prices),
            "n_compras": len(rs),
            "p25": prices[len(prices) // 4],
            "p75": prices[(3 * len(prices)) // 4],
            "quantidade": st.median(qty),
            "unidade": "UN",
            "tipo": "Material",
            "objeto": "",
            "nomePdm": recent.get("nomePdm"),
            "nomeClasse": recent.get("nomeClasse"),
            # one example purchase, for display. No supplier name.
            "ex_orgao": recent.get("nomeOrgao"),
            "ex_municipio": recent.get("municipio"),
            "ex_uf": recent.get("estado"),
            "ex_esfera": recent.get("esfera"),
            "ex_data": recent.get("dataCompra"),
            "ex_preco": recent["valor_unit"],
        })
    return out


def train_set(src, out_train, out_index, n, min_purchases, seed):
    rows = _read(src)
    kept, rep = clean_paid(rows)
    agg = aggregate_paid(kept)
    _write(sorted(agg, key=lambda r: (r["group"], r["item_desc"])), out_index)

    eligible = [r for r in agg if r["n_compras"] >= min_purchases]
    rng = random.Random(seed)
    rng.shuffle(eligible)
    sample = eligible[:n]
    _write(sample, out_train)

    _log("cleaning report (purchases):")
    for k, v in rep.items():
        _log(f"  {k:32s} {v:7d}")
    _log(f"aggregated: {len(agg)} (code, description) rows -> {out_index}")
    _log(f"eligible (>= {min_purchases} purchases): {len(eligible)}; "
         f"sampled {len(sample)} -> {out_train}")
    _log(f"codes in sample: {len({r['group'] for r in sample})}")


# --------------------------------------------------------------- audit side

def _tender_ref(listing_row):
    """``82827148000169-1-000034/2026`` -> the dict ``pncp_items`` expects."""
    m = re.match(r"^(\d{14})-\d+-(\d+)/(\d{4})$", listing_row.get("tender") or "")
    if not m:
        return None
    cnpj, seq, ano = m.groups()
    return {
        "numeroControlePNCP": listing_row["tender"],
        "orgaoEntidade": {"cnpj": cnpj, "razaoSocial": listing_row.get("orgao"),
                          "esferaId": listing_row.get("esfera"),
                          "poderId": listing_row.get("poder")},
        "anoCompra": int(ano),
        "sequencialCompra": int(seq),
        "unidadeOrgao": {"ufSigla": listing_row.get("uf")},
        "modalidadeNome": listing_row.get("modalidade"),
        "objetoCompra": listing_row.get("item_desc"),
    }


HARDWARE_OBJECT = re.compile(
    r"\b(computador|microcomputador|notebook|desktop|all in one|tablet|"
    r"impressora|multifuncional|toner|cartucho|monitor|switch|roteador|"
    r"nobreak|no-break|servidor|storage|scanner|projetor|webcam|teclado|mouse|"
    r"equipamentos? de inform[áa]tica|materia(?:l|is) de inform[áa]tica|"
    r"perif[ée]ricos?|suprimentos? de inform[áa]tica|hd externo|ssd|pen ?drive)",
    re.I)


def audit_set(listing, out, workers, limit):
    from .sources.pncp_items import fetch_items

    tenders = [t for t in _read(listing) if HARDWARE_OBJECT.search(t.get("item_desc") or "")]
    refs = [r for r in map(_tender_ref, tenders) if r][:limit]
    _log(f"{len(refs)} IT-equipment tenders to fetch")
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, got in enumerate(ex.map(fetch_items, refs), 1):
            rows.extend(got)
            if i % 25 == 0:
                _log(f"  {i}/{len(refs)} tenders -> {len(rows)} items")
    mats = [r for r in rows if (r.get("tipo") or "").lower().startswith("mat")]
    _write(mats, out)
    _log(f"WROTE {out}: {len(mats)} material items ({len(rows)} total) "
         f"from {len(refs)} tenders")


def _per_unit(u):
    u = (u or "").strip().lower()
    return u.startswith(("un", "pç", "pc", "peç", "pec")) or u in ("", "und")


def audit_prep(src, dst, lo=PRICE_RANGE[0], hi=PRICE_RANGE[1]):
    """Raw PNCP items -> rows the LLM reads exactly like the training rows.

    The training rows carry no tender object, so the model reads only the
    item description here too; the object is kept for display only.
    """
    seen, out, rep = set(), [], Counter()
    for r in _read(src):
        desc = " ".join((r.get("item_desc") or "").split())
        key = (r.get("tender"), desc.lower(), round(float(r["valor_unit"]), 2))
        if key in seen:
            rep["duplicate"] += 1
            continue
        seen.add(key)
        if not (lo <= float(r["valor_unit"]) <= hi):
            rep["price outside range"] += 1
            continue
        out.append(dict(r, item_desc=desc[:700], objeto_licitacao=r.get("objeto"), objeto="",
                        unidade_original=r.get("unidade"),
                        unidade="UN" if _per_unit(r.get("unidade")) else r.get("unidade")))
    _write(out, dst)
    _log(f"{dst}: kept {len(out)} · dropped {dict(rep)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the item-level datasets.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train-set")
    t.add_argument("--src", default="data/raw/comprasgov_grupo70.jsonl")
    t.add_argument("--out", default="data/items/train.jsonl")
    t.add_argument("--index", default="data/items/paid_index.jsonl")
    t.add_argument("-n", type=int, default=3000)
    t.add_argument("--min-purchases", type=int, default=2)
    t.add_argument("--seed", type=int, default=7)
    a = sub.add_parser("audit-set")
    a.add_argument("--listing", default="data/raw/pncp_ti.jsonl")
    a.add_argument("--out", default="data/raw/pncp_audit_items.jsonl")
    a.add_argument("--workers", type=int, default=2)
    a.add_argument("--limit", type=int, default=100000)
    p = sub.add_parser("audit-prep")
    p.add_argument("--src", default="data/raw/pncp_audit_items.jsonl")
    p.add_argument("--out", default="data/items/audit.jsonl")
    args = ap.parse_args(argv)
    if args.cmd == "train-set":
        train_set(args.src, args.out, args.index, args.n, args.min_purchases, args.seed)
    elif args.cmd == "audit-set":
        audit_set(args.listing, args.out, args.workers, args.limit)
    else:
        audit_prep(args.src, args.out)


if __name__ == "__main__":
    main()
