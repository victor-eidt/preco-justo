"""Short, readable titles for catalogue-style item descriptions.

Compras.gov.br and many PNCP items follow the catalogue pattern
``<Name> <attribute>: <value>, <attribute>: <value>, ...``, which is
unreadable in a list. ``short_title`` returns ``(title, spec)``: the
product name with its quantity attribute folded in ("Switch 16 portas") and
the next attribute values as a one-line spec. Free text falls back to its
first clause. Display only; the model never sees these.
"""

import html
import re

ATTR_START = (
    "quantidade", "tipo", "capacidade", "velocidade", "aplicação", "aplicacao",
    "características", "caracteristicas", "cor", "referência", "referencia",
    "refêrencia", "tamanho", "material", "modelo", "formato", "interface",
    "resolução", "resolucao", "padrão", "padrao", "tensão", "tensao",
    "alimentação", "alimentacao", "potência", "potencia", "largura",
    "comprimento", "núcleos", "nucleos", "armazenamento", "frequência",
    "frequencia", "compatibilidade", "conexão", "conexao", "tecnologia",
    "processador", "memória ram", "memoria ram", "tela", "bateria",
    "comprimento", "rendimento", "voltagem", "função", "funcao", "uso",
)
# zero-width, so overlapping candidates are all found ("Memória Ram referência:")
_ATTR_RX = re.compile(
    r"(?=\s(?:" + "|".join(re.escape(a) for a in ATTR_START) + r")\b[^:,]{0,40}:)", re.I)
_PAIR_RX = re.compile(r"([^:,]{2,50}):\s*([^:]+?)(?=,\s*[^,:]{2,50}:|$)")


_CODE_RX = re.compile(r"\b(?=[a-z0-9-]*\d)(?=[a-z0-9-]*[a-z])[a-z0-9]+(?:-[a-z0-9]+)*\b", re.I)
_ACRO_RX = re.compile(r"\b(usb|ssd|hdd|ram|ddr\d?|hd|led|lcd|wi-?fi|poe\+?|sfp\+?|rj-?45|cat ?\d|"
                      r"cftv|dvd|cd|vga|hdmi|ti|a\d)\b", re.I)
_TITLE_KEYS = ("cor", "refer", "refêr", "capacidade")
_PREP_END = re.compile(r"\b(de|da|do|das|dos|para|com|e|em)$", re.I)


def _sentence(s):
    s = html.unescape(s).strip(" .;-–")
    letters = [c for c in s if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.7:
        s = s.lower()
        s = s[:1].upper() + s[1:]
    s = _ACRO_RX.sub(lambda m: m.group(0).upper(), s)
    return s


def _codes_upper(v):
    return _CODE_RX.sub(lambda m: m.group(0).upper(), v)


def _numeric(v):
    return bool(re.match(r"^(\d|superior|inferior|até|ate|acima|entre)", v, re.I))


def short_title(desc, max_spec=70):
    d = html.unescape(" ".join((desc or "").split()))
    d = re.sub(r"^\(?\d{4,}\)?\s*[-–]\s*", "", d)
    d = re.sub(r"^(lote\s*\d+\s*[-–:]\s*)", "", d, flags=re.I)
    m = next((m for m in _ATTR_RX.finditer(" " + d)
              if m.start() > 0 and not _PREP_END.search(d[: m.start()].strip(" ,"))), None)
    if m:
        name = d[: m.start()].strip(" ,")
        rest = d[m.start():].strip()
        pairs = [(k.strip().lower(), v.strip()) for k, v in _PAIR_RX.findall(rest)]
        title = _sentence(name)
        extra, spec_vals, qty_done = [], [], False
        for k, v in pairs:
            v = _sentence(v)
            if k.startswith("quantidade ") and re.match(r"^\d+([.,]\d+)?$", v) and not qty_done:
                title, qty_done = f"{title} {v} {k[len('quantidade '):]}", True
            elif k.startswith(_TITLE_KEYS) and len(v) <= 14 and len(extra) < 2:
                extra.append(_codes_upper(v))
            else:
                spec_vals.append(f"{k} {v}" if _numeric(v) else v)
        if extra:
            title = f"{title} {' '.join(extra)}"
        spec = " · ".join(spec_vals)
    else:
        parts = re.split(r"\s[-–]\s|[,;(:]", d, maxsplit=1)
        title = _codes_upper(_sentence(parts[0]))
        spec = _sentence(parts[1]) if len(parts) > 1 else ""
    title = re.sub(r"^Cartucho Toner Impressora\b", "Toner", title, flags=re.I)
    title = re.sub(r"^Cartucho Tinta Impressora\b", "Cartucho de tinta", title, flags=re.I)
    title = re.sub(r"^Cartucho de toner\b", "Toner", title, flags=re.I)
    if len(title) > 60:
        title = title[:57].rsplit(" ", 1)[0] + "…"
    spec = spec.strip(" .;,)")
    if len(spec) > max_spec:
        spec = spec[: max_spec - 1].rsplit(" ", 1)[0] + "…"
    return title, spec
