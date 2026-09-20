"""Pre-declared sectors and the keyword filters that select them.

Three sectors were declared before any result was seen: ``ti``,
``medicamentos``, ``combustiveis``. All three are reported, whatever the
numbers say. Picking the winner after the fact would be p-hacking.

Keyword filters are a deliberate low-tech choice for the crawl stage: they
are free, run on every listing row, and a false positive here only costs an
LLM call later. Precision is recovered at the item level (``sources.clean``).
"""

import re

TI_RX = re.compile(
    r"\b(software|sistema|licen[çc]|inform[áa]tic|tecnologia da informa|"
    r"computador|notebook|desktop|servidor|micro ?computador|"
    r"impressora|switch|firewall|storage|datacenter|data center|"
    r"nobreak|no-break|scanner|monitor|link de internet|"
    r"hospedagem|nuvem|cloud|erp|help ?desk|suporte t[ée]cnico)\b",
    re.I,
)

# Objects that trip the keyword net but are not IT ("sistema de esgoto").
TI_SKIP_RX = re.compile(
    r"\b(esgoto|abastecimento de [áa]gua|ilumina[çc][ãa]o p[úu]blica|"
    r"sistema vi[áa]rio|drenagem|g[áa]s medicinal|ar condicionado)\b",
    re.I,
)

SECTORS = {
    "ti": (TI_RX, TI_SKIP_RX),
    "medicamentos": (
        re.compile(r"\b(medicament|f[aá]rmac|material m[eé]dico|"
                   r"insumo hospitalar|material hospitalar|correlatos|"
                   r"vacina|soro fisiol|antibi[oó]tic|farm[aá]cia b[aá]sica)\b", re.I),
        re.compile(r"\b(veterin[aá]ri|constru[cç][aã]o|reforma)\b", re.I)),
    "combustiveis": (
        re.compile(r"\b(combust[ií]ve|gasolina|etanol|[oó]leo diesel|"
                   r"diesel s-?10|arla 32|g[aá]s liquefeito|glp|"
                   r"querosene|lubrificante)\b", re.I),
        re.compile(r"\b(usina|refinaria|explora[cç][aã]o)\b", re.I)),
}


def matches(sector, text):
    keep, skip = SECTORS[sector]
    return bool(keep.search(text)) and not skip.search(text)
