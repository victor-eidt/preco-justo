"""Clean raw item rows into a modelling dataset.

Tenders whose OBJECT looked like IT still drag in non-IT items ("manutenção
predial e sistemas" brings lock repairs). Filter again at the ITEM level,
drop duplicates, and drop prices too extreme to model.

Usage::

    python -m precojusto.sources.clean data/raw/pncp_items.jsonl data/pilot/dataset.jsonl
"""

import argparse
import json
import re
from collections import Counter

ITEM_TI = re.compile(
    r"\b(software|sistema|licen[çc]|inform[áa]tic|computador|notebook|desktop|"
    r"servidor|impressora|toner|cartucho|switch|roteador|firewall|storage|"
    r"nobreak|no-break|scanner|monitor|teclado|mouse|estabilizador|"
    r"internet|link dedicado|nuvem|cloud|erp|hd externo|ssd|mem[óo]ria ram|"
    r"processador|cabo de rede|rack|webcam|headset|tablet|smartphone|"
    r"antiv[íi]rus|backup|dom[íi]nio|datacenter|data center|"
    r"desenvolvimento de sistema|suporte t[ée]cnico|ti\b)\b",
    re.I,
)

# Words that match the IT filter but mean something else entirely.
ITEM_NOT_TI = re.compile(
    r"\b(quarto|hotel|di[áa]ria|caf[ée] da manh|leito|apartamento|"
    r"sistema de esgoto|sistema vi[áa]rio|sistema de ilumina|"
    r"sistema de irriga|sistema de ar condicionado|sistema construtivo|"
    r"fechadura|hidr[áa]ulic|sanit[áa]ri)\b",
    re.I,
)

PRICE_RANGE = (5.0, 300_000.0)  # unit prices we try to model


def norm_unit(u):
    u = (u or "").strip().lower()
    if not u:
        return "na"
    if u.startswith(("mes", "mês")):
        return "mes"
    if u.startswith(("serv", "sv")):
        return "servico"
    if u.startswith(("un", "pc", "peç", "pec")):
        return "unidade"
    if u.startswith(("cx", "caixa")):
        return "caixa"
    if u.startswith(("hora", "h/", "hr")):
        return "hora"
    return "outro"


def clean(rows, lo=PRICE_RANGE[0], hi=PRICE_RANGE[1]):
    seen, out = set(), []
    for r in rows:
        desc = " ".join(r["item_desc"].split())
        key = (r.get("tender"), desc.lower(), round(r["valor_unit"], 2))
        if key in seen:
            continue
        seen.add(key)
        if not ITEM_TI.search(desc) or ITEM_NOT_TI.search(desc) or len(desc) < 20:
            continue
        if not (lo <= r["valor_unit"] <= hi):
            continue
        r = dict(r, item_desc=desc[:600], unidade_n=norm_unit(r.get("unidade")))
        try:
            r["quantidade"] = float(r.get("quantidade") or 1) or 1.0
        except (TypeError, ValueError):
            r["quantidade"] = 1.0
        out.append(r)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Filter raw item rows into a dataset.")
    ap.add_argument("src")
    ap.add_argument("dst")
    args = ap.parse_args(argv)
    with open(args.src, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    out = clean(rows)
    with open(args.dst, "w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tenders = {r.get("tender") for r in out}
    print(f"raw {len(rows)} -> kept {len(out)} items from {len(tenders)} tenders")
    print("unidade:", Counter(r["unidade_n"] for r in out).most_common())
    p = sorted(r["valor_unit"] for r in out)
    if p:
        q = lambda x: p[int(x * (len(p) - 1))]  # noqa: E731
        print(f"price  : p05 {q(.05):.2f}  p25 {q(.25):.2f}  p50 {q(.5):.2f}  "
              f"p75 {q(.75):.2f}  p95 {q(.95):.2f}")


if __name__ == "__main__":
    main()
