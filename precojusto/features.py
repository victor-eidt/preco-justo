"""Ask the LLM the typed questions about every item and write a feature row.

One request per item carries all the questions (much cheaper than one call
per question). Answers are cached on disk, keyed by the question-set
fingerprint plus the exact state sent to the model, so re-runs cost nothing
and editing a question only re-asks that set.

Usage::

    python -m precojusto.features --input data/pilot/dataset.jsonl \
        --output data/pilot/features.jsonl --cache data/raw/jev_cache.json

Needs ``TYPESAFE_API_KEY`` (or ``~/.typesafe_key``) unless everything is cached.
"""

import argparse
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from .questions import BAND_LOG10, CATEGORIES, NOULS, QUESTIONS, QUESTIONS_FINGERPRINT, SCORES

DEFAULT_MODEL = os.environ.get("JEV_MODEL", "jev-latest")


def state_for(row):
    """What the model reads. Kept minimal so answers depend on the text."""
    return json.dumps({
        "descricao_do_item": row["item_desc"],
        "unidade_de_medida": row.get("unidade"),
        "quantidade": row.get("quantidade"),
        "material_ou_servico": row.get("tipo"),
        "objeto_da_licitacao": (row.get("objeto") or "")[:300],
    }, ensure_ascii=False)


def cache_key(row):
    """Fingerprint-prefixed state. The prefix never reaches the model."""
    return QUESTIONS_FINGERPRINT + "|" + state_for(row)


def _probs(answer):
    p = getattr(answer, "probabilities", None)
    return p if isinstance(p, dict) else {}


def to_features(answers):
    """Typed answers -> flat dict of floats.

    Score -> ``<q>__nivel`` (expected level) and ``<q>__disp`` (std dev).
    Noul  -> ``<q>__p``.
    Choice -> one ``cat__<option>`` per option and ``cat__conf``.
    Plus the direct-price rung: ``direct__log10`` and ``direct__conf``.
    """
    f = {}
    for k in SCORES:
        pairs = [(float(lvl), p) for lvl, p in _probs(answers[k]).items()]
        tot = sum(p for _, p in pairs) or 1.0
        mean = sum(lvl * p for lvl, p in pairs) / tot
        var = sum(p * (lvl - mean) ** 2 for lvl, p in pairs) / tot
        f[f"{k}__nivel"] = mean
        f[f"{k}__disp"] = math.sqrt(max(var, 0.0))
    for k in NOULS:
        f[f"{k}__p"] = float(answers[k].noul)

    cat = answers["categoria"]
    cp = _probs(cat)
    for opt in CATEGORIES:
        f[f"cat__{opt}"] = float(cp.get(opt, 0.0))
    f["cat__conf"] = float(getattr(cat, "confidence", 0.0) or 0.0)

    fp = answers["faixa_preco"]
    probs = _probs(fp)
    tot = sum(probs.values()) or 1.0
    f["direct__log10"] = sum(
        BAND_LOG10[int(lvl)] * p for lvl, p in probs.items()
        if 0 <= int(lvl) < len(BAND_LOG10)
    ) / tot
    f["direct__conf"] = float(getattr(fp, "confidence", 0.0) or 0.0)
    return f


def api_key():
    k = os.environ.get("TYPESAFE_API_KEY")
    if k:
        return k.strip()
    path = os.path.expanduser("~/.typesafe_key")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return fh.read().strip()
    sys.exit("No API key. Set TYPESAFE_API_KEY or write it to ~/.typesafe_key")


def load_cache(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def save_cache(cache, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(cache, fh)
    os.replace(tmp, path)


def featurize(rows, cache, client, model=DEFAULT_MODEL, workers=8,
              cache_path=None, log=print):
    """Fill ``cache`` with features for every row missing one. Returns #errors."""
    todo = [r for r in rows if cache_key(r) not in cache]
    log(f"{len(rows)} items; {len(todo)} need inference, {len(rows) - len(todo)} cached")
    if not todo:
        return 0

    def ask(row):
        key = cache_key(row)
        try:
            resp = client.system_one(state=state_for(row), questions=QUESTIONS,
                                     model=model)
            return key, to_features(resp.answers), None
        except Exception as e:  # noqa: BLE001 - one bad item must not stop the run
            return key, None, f"{type(e).__name__}: {e}"

    errors = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (key, feats, err) in enumerate(ex.map(ask, todo), 1):
            if err:
                errors += 1
                if errors <= 3:
                    log(f"  ERROR {err}")
            else:
                cache[key] = feats
            if i % 50 == 0:
                log(f"  {i}/{len(todo)} ({errors} errors)")
                if cache_path:
                    save_cache(cache, cache_path)
    if cache_path:
        save_cache(cache, cache_path)
    return errors


def write_features(rows, cache, path):
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            feats = cache.get(cache_key(r))
            if feats is not None:
                fh.write(json.dumps({**r, "jev": feats}, ensure_ascii=False) + "\n")
                n += 1
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", default="data/pilot/dataset.jsonl")
    ap.add_argument("--output", default="data/pilot/features.jsonl")
    ap.add_argument("--cache", default="data/raw/jev_cache.json")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)

    with open(args.input, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    cache = load_cache(args.cache)

    log = lambda m: print(m, file=sys.stderr)  # noqa: E731
    if any(cache_key(r) not in cache for r in rows):
        from typesafe_sdk import TypeSafeClient
        client = TypeSafeClient(api_key=api_key())
        errors = featurize(rows, cache, client, args.model, args.workers,
                           cache_path=args.cache, log=log)
    else:
        errors = 0
        log(f"{len(rows)} items, all cached")
    n = write_features(rows, cache, args.output)
    log(f"WROTE {args.output}: {n} items ({errors} failed)")


if __name__ == "__main__":
    main()
