"""The price model behind the app: learn paid prices, audit estimated ones.

What it produces for one item:

- a predicted unit price (the median the model expects someone to PAY);
- a range around it, calibrated so that ~80% of held-out paid prices fall
  inside (conformalised quantile regression: two quantile models, widened
  by the error they made on cross-validated rows);
- a verdict on the agency's ESTIMATE, from four states:

  ``revisar``       estimate above the range by more than ``MARGIN``
  ``abaixo``        estimate below the range by more than ``MARGIN``
  ``ok``            otherwise
  ``sem_confianca`` the system abstains: the range is too wide to judge
                    (the widest ``ABSTAIN_FRAC`` of cross-validated rows),
                    the item is not IT equipment, its description is too
                    generic to price, it is not priced per unit, or no paid
                    purchase is similar enough to compare against.

``MARGIN`` exists because a paid price comes out of an auction and an
estimate goes in before it: estimates sit above paid prices as a rule. An
estimate is only flagged when it clears the top of the range by 1.5x.

Usage::

    python -m precojusto.model train            # dev only; holdout untouched
    python -m precojusto.model final            # score the holdout, ONCE
    python -m precojusto.model queue            # batch: audit items -> queue.json
    python -m precojusto.model bias             # ethics: judged/flagged rate by sphere
"""

import argparse
import json
import math
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .ladder import DIRECT_COLUMNS, deterministic_features, group_ids, load
from .splits import group_folds, holdout_split

COVERAGE = 0.80        # target coverage of the range
ABSTAIN_FRAC = 0.20    # abstain on the widest 20% of ranges
MARGIN = 1.5           # an estimate must clear the range by this factor
EQUIP_MIN = 0.5        # P(IT equipment) below this -> out of scope
MIN_SIM = 0.3          # no paid purchase this similar -> nothing to compare against.
                       # Half the training rows have one at >= 0.83.
SPEC_MIN = 1.5         # LLM-read specificity below this -> too generic to price.
                       # Every training row (catalogue text) reads >= ~2.
HOLDOUT_SEED = 12345

BUNDLE = "data/items/model.pkl"
TEXT_COLUMNS = ["desc_len", "desc_words", "n_digits", "n_upper", "n_meses", "maior_numero"]
EQUIP_CATS = ["cat__hardware_computador", "cat__periferico_suprimento",
              "cat__infraestrutura_rede"]


def _log(msg):
    print(msg, file=sys.stderr, flush=True)


# ------------------------------------------------------------------ matrix

def jev_columns(jev):
    return [c for c in jev.columns if c not in DIRECT_COLUMNS]


def feature_matrix(descs, jev, columns, index=None, groups=None, regime=None):
    """[LLM answers | cheap text statistics | similar paid purchases].

    On training rows the neighbour search must not see the row's own price.
    ``regime`` says how much is hidden (``groups`` = each row's catalogue code):

    - ``"own"``: only the row's own (code, description) entry. Other
      descriptions of the same catalogue item stay visible, as they are when
      a PNCP item is audited against the full paid-price database. The app
      model uses this.
    - ``"code"``: every purchase of the row's catalogue code. The strict
      reading: an item whose catalogue entry was never priced at all.
    - ``None``: nothing hidden (audit time).
    """
    descs = list(descs)
    det = deterministic_features(pd.DataFrame({"item_desc": descs}))[TEXT_COLUMNS]
    parts = [jev[columns].to_numpy(float), det.to_numpy(float)]
    if index is not None:
        parts.append(index.knn_features(descs, groups, regime))
    return np.hstack(parts)


def gb(loss="squared_error", quantile=None):
    kw = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=15,
              min_samples_leaf=5, l2_regularization=1.0, random_state=0)
    if quantile is not None:
        kw.update(loss="quantile", quantile=quantile)
    return HistGradientBoostingRegressor(**kw)


class PriceModel:
    """Point model + two quantile models + the calibration constants."""

    def __init__(self, columns):
        self.columns = columns
        lo_q = (1 - COVERAGE) / 2
        self.models = {"mid": gb(), "lo": gb(quantile=lo_q), "hi": gb(quantile=1 - lo_q)}
        self.qhat = 0.0          # conformal widening, log10 units
        self.width_cut = np.inf  # abstain above this range width (log10)

    def fit(self, X, y):
        for m in self.models.values():
            m.fit(X, y)
        return self

    def raw(self, X):
        return {k: m.predict(X) for k, m in self.models.items()}

    def predict(self, X):
        p = self.raw(X)
        lo = np.minimum(p["lo"], p["mid"]) - self.qhat
        hi = np.maximum(p["hi"], p["mid"]) + self.qhat
        return p["mid"], lo, hi


def calibrate(X, y, groups, columns, folds=5):
    """Cross-validate inside the training rows to set ``qhat`` and the cut.

    Returns the fitted-on-everything model and the out-of-fold predictions.
    """
    oof = {k: np.zeros(len(y)) for k in ("mid", "lo", "hi")}
    for tr, te in group_folds(groups, folds, seed=0):
        m = PriceModel(columns).fit(X[tr], y[tr])
        for k, v in m.raw(X[te]).items():
            oof[k][te] = v
    lo = np.minimum(oof["lo"], oof["mid"])
    hi = np.maximum(oof["hi"], oof["mid"])
    scores = np.maximum(lo - y, y - hi)
    n = len(y)
    level = min(1.0, math.ceil((n + 1) * COVERAGE) / n)
    qhat = float(np.quantile(scores, level, method="higher"))

    model = PriceModel(columns).fit(X, y)
    model.qhat = qhat
    widths = (hi + qhat) - (lo - qhat)
    model.width_cut = float(np.quantile(widths, 1 - ABSTAIN_FRAC))
    return model, {"mid": oof["mid"], "lo": lo - qhat, "hi": hi + qhat, "width": widths}


def report(name, y, mid, lo, hi, width_cut):
    err = mid - y
    ratio = 10 ** np.abs(err)
    inside = (y >= lo) & (y <= hi)
    keep = (hi - lo) <= width_cut
    r = {
        "conjunto": name, "n": int(len(y)),
        "RMSE_log10": float(np.sqrt(np.mean(err ** 2))),
        "mediana_erro_x": float(np.median(ratio)),
        "dentro_2x_%": float(100 * np.mean(ratio <= 2)),
        "cobertura_faixa_%": float(100 * inside.mean()),
        "respondidos_%": float(100 * keep.mean()),
        "dentro_2x_respondidos_%": float(100 * np.mean(ratio[keep] <= 2)),
        "cobertura_respondidos_%": float(100 * inside[keep].mean()),
        "largura_mediana_x": float(np.median(10 ** (hi - lo))),
    }
    for k, v in r.items():
        _log(f"  {k:26s} {v:.3f}" if isinstance(v, float) else f"  {k:26s} {v}")
    return r


# ---------------------------------------------------------------- neighbours

class PaidIndex:
    """'Similar purchases already paid': text search over Compras.gov.br."""

    def __init__(self, rows):
        self.rows = rows
        self.vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=1,
                                   strip_accents="unicode", max_features=60000)
        self.mat = self.vec.fit_transform([self._text(r) for r in rows])
        self.groups = np.array([r["group"] for r in rows])
        self.pos = {(r["group"], r["item_desc"]): i for i, r in enumerate(rows)}
        self.logp = np.log10([r["valor_unit"] for r in rows])

    @staticmethod
    def _text(r):
        return f"{r.get('nomePdm') or ''} {r['item_desc']}"

    def knn_features(self, descs, groups=None, regime=None, k=10):
        """Weighted median / mean / spread of the k most similar paid prices,
        the top neighbour's price, and how similar it is. log10 units.
        See ``feature_matrix`` for ``regime``."""
        sims = linear_kernel(self.vec.transform(descs), self.mat)
        out = np.zeros((len(descs), 5))
        for i, s in enumerate(sims):
            if regime == "code":
                s = np.where(self.groups == groups[i], -1.0, s)
            elif regime == "own":
                j = self.pos.get((groups[i], descs[i]))
                if j is not None:
                    s = s.copy()
                    s[j] = -1.0
            top = np.argpartition(-s, k)[:k]
            top = top[np.argsort(-s[top])]
            w = np.clip(s[top], 1e-3, None)
            p = self.logp[top]
            o = np.argsort(p)
            wmed = p[o][np.searchsorted(np.cumsum(w[o]) / w.sum(), 0.5)]
            mean = np.average(p, weights=w)
            sd = math.sqrt(np.average((p - mean) ** 2, weights=w))
            out[i] = [wmed, mean, sd, p[0], s[top[0]]]
        return out

    def similar(self, desc, k=5, min_sim=0.15):
        q = self.vec.transform([desc])
        sims = linear_kernel(q, self.mat).ravel()
        idx = np.argsort(-sims)[: k * 3]
        out, seen = [], set()
        for i in idx:
            if sims[i] < min_sim:
                break
            r = self.rows[i]
            key = r["item_desc"][:80]
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "descricao": r["item_desc"], "pdm": r.get("nomePdm"),
                "mediana": r["valor_unit"], "n_compras": r["n_compras"],
                "p25": r["p25"], "p75": r["p75"],
                "orgao": r.get("ex_orgao"), "municipio": r.get("ex_municipio"),
                "uf": r.get("ex_uf"), "data": r.get("ex_data"),
                "similaridade": float(sims[i]),
            })
            if len(out) == k:
                break
        return out


# ------------------------------------------------------------------ verdict

def _x(r):
    """1.53 -> '1,5×' (Brazilian decimal comma)."""
    return (f"{r:.0f}" if r >= 10 else f"{r:.1f}").replace(".", ",") + "×"


def equipment_prob(f):
    return min(1.0, sum(f.get(c, 0.0) for c in EQUIP_CATS))


def verdict(estimate, mid, lo, hi, width, width_cut, f, per_unit=True, top_sim=1.0):
    """The four states. ``f`` is the dict of LLM answers for this item."""
    price = {"previsto": 10 ** mid, "faixa_min": 10 ** lo, "faixa_max": 10 ** hi}
    reasons = []
    if equipment_prob(f) < EQUIP_MIN or f.get("e_hardware__p", 1.0) < EQUIP_MIN:
        reasons.append("o item não parece equipamento de TI, que é o que o modelo conhece")
    elif f.get("especificidade__nivel", 4.0) < SPEC_MIN:
        reasons.append("a descrição é genérica demais para estimar um preço: "
                       "nomeia o tipo de produto, sem especificação")
    if not per_unit:
        reasons.append("o preço não é por unidade")
    if top_sim < MIN_SIM:
        reasons.append("não há compra parecida já paga para comparar")
    if width > width_cut:
        reasons.append("a faixa prevista ficou larga demais para julgar")
    out = dict(price, razao=(estimate / price["previsto"]) if estimate else None)
    if reasons:
        return dict(out, estado="sem_confianca", motivos=reasons)
    if estimate is None:
        return dict(out, estado="ok", motivos=[])
    if estimate > price["faixa_max"] * MARGIN:
        return dict(out, estado="revisar", motivos=[
            "a estimativa está " + _x(estimate / price["faixa_max"]) + " acima do teto da faixa"])
    if estimate < price["faixa_min"] / MARGIN:
        return dict(out, estado="abaixo", motivos=[
            "a estimativa está " + _x(price["faixa_min"] / estimate) + " abaixo do piso da faixa"])
    return dict(out, estado="ok", motivos=[])


# ------------------------------------------------------------------ commands

def _split(df):
    groups = group_ids(df)
    dev, hold = holdout_split(groups, 0.30, HOLDOUT_SEED)
    return groups, dev, hold


def load_index(path):
    with open(path, encoding="utf-8") as fh:
        return PaidIndex([json.loads(line) for line in fh])


REGIMES = {"own": "busca sem a própria descrição (modelo do app)",
           "code": "busca sem o código de catálogo inteiro (estrito)"}


def cmd_train(features, index_path, out):
    df, jev = load(features)
    columns = jev_columns(jev)
    index = load_index(index_path)
    y = np.log10(df["valor_unit"].to_numpy(float))
    groups, dev, hold = _split(df)
    _log(f"{len(df)} rows · {len(np.unique(groups))} catalogue codes · "
         f"dev {len(dev)} · holdout {len(hold)} (locked)")

    bundle = {"index": index, "n_train": int(len(dev)), "models": {}, "dev_reports": {}}
    for regime, label in REGIMES.items():
        X = feature_matrix(df["item_desc"], jev, columns, index, df["group"].tolist(), regime)
        model, oof = calibrate(X[dev], y[dev], groups[dev], columns)
        _log(f"[{regime}] {label}")
        _log(f"  qhat {model.qhat:.3f} log10 · abstain when range wider than "
             f"{10 ** model.width_cut:.1f}x · dev 5-fold group CV:")
        bundle["models"][regime] = model
        bundle["dev_reports"][regime] = report(f"dev CV · {regime}", y[dev], oof["mid"],
                                               oof["lo"], oof["hi"], model.width_cut)
    bundle["model"], bundle["dev_report"] = bundle["models"]["own"], bundle["dev_reports"]["own"]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as fh:
        pickle.dump(bundle, fh)
    _log(f"saved {out}")


def cmd_final(features, bundle_path, out_csv):
    """Score the locked holdout with the models fitted on dev. Meant to run once."""
    _log("*** FINAL: scoring the locked item-level holdout. Once. ***")
    df, jev = load(features)
    with open(bundle_path, "rb") as fh:
        bundle = pickle.load(fh)
    y = np.log10(df["valor_unit"].to_numpy(float))
    _, _, hold = _split(df)
    rows, bundle["holdout_reports"] = [], {}
    for regime, model in bundle["models"].items():
        X = feature_matrix(df["item_desc"], jev, model.columns, bundle["index"],
                           df["group"].tolist(), regime)
        mid, lo, hi = model.predict(X[hold])
        _log(f"[{regime}] {REGIMES[regime]}")
        r = report(f"holdout · {regime}", y[hold], mid, lo, hi, model.width_cut)
        bundle["holdout_reports"][regime] = r
        rows += [bundle["dev_reports"][regime], r]
    bundle["holdout_report"] = bundle["holdout_reports"]["own"]
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    with open(bundle_path, "wb") as fh:
        pickle.dump(bundle, fh)
    _log(f"wrote {out_csv}")


def unit_is_each(u):
    u = (u or "").strip().lower()
    return u.startswith(("un", "pç", "pc", "peç", "pec")) or u in ("", "und")


def cmd_queue(features, bundle_path, out):
    """Batch job: score every audit item, write the queue the app reads."""
    with open(bundle_path, "rb") as fh:
        bundle = pickle.load(fh)
    model, index = bundle["model"], bundle["index"]
    df, jev = load(features)
    X = feature_matrix(df["item_desc"], jev, model.columns, index)
    mid, lo, hi = model.predict(X)
    def clean(v):
        if isinstance(v, float) and not math.isfinite(v):
            return None
        return v

    items = []
    for i, r in df.iterrows():
        r = r.map(clean) if hasattr(r, "map") else r
        f = jev.iloc[i].to_dict()
        v = verdict(float(r["valor_unit"]), mid[i], lo[i], hi[i], hi[i] - lo[i],
                    model.width_cut, f, per_unit=unit_is_each(r.get("unidade")),
                    top_sim=float(X[i, -1]))
        items.append({
            "id": int(i), "descricao": r["item_desc"], "orgao": r.get("orgao"),
            "uf": r.get("uf"), "esfera": r.get("esfera"), "modalidade": r.get("modalidade"),
            "objeto": r.get("objeto_licitacao"), "tender": r.get("tender"),
            "quantidade": r.get("quantidade"), "unidade": r.get("unidade"),
            "estimado": float(r["valor_unit"]), **v, "jev": f,
            "llm_direto": 10 ** f["direct__log10"],
            "parecidas": index.similar(r["item_desc"]),
        })
    order = {"revisar": 0, "abaixo": 1, "ok": 2, "sem_confianca": 3}
    items.sort(key=lambda it: (order[it["estado"]],
                               -abs(math.log10(it["razao"] or 1))))
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"items": items, "dev_report": bundle["dev_report"],
                   "holdout_report": bundle.get("holdout_report")}, fh, ensure_ascii=False)
    counts = pd.Series([it["estado"] for it in items]).value_counts().to_dict()
    _log(f"WROTE {out}: {len(items)} items · {counts}")


SPHERES = {"F": "federal", "E": "estadual", "M": "municipal", "D": "distrital",
           "N": "não se aplica"}


def bias_table(items):
    """Per government sphere: how often the system judges, and how often it flags.

    Two rates, because the risk runs two ways. A higher flag rate among judged
    items would put suspicion on one kind of buyer; a lower judged rate means
    the system is blind to that buyer and protects nobody there.
    """
    df = pd.DataFrame({"esfera": [it.get("esfera") or "?" for it in items],
                       "estado": [it["estado"] for it in items],
                       "motivo": [(it.get("motivos") or [None])[0] for it in items]})
    rows = []
    for code, g in df.groupby("esfera"):
        judged = g[g.estado != "sem_confianca"]
        rows.append({"esfera": SPHERES.get(code, code), "itens": len(g),
                     "julgados": len(judged), "pct_julgados": len(judged) / len(g),
                     "revisar": int((g.estado == "revisar").sum()),
                     "pct_revisar_dos_julgados": ((judged.estado == "revisar").mean()
                                                  if len(judged) else float("nan")),
                     # share of the sphere's items with no paid purchase like them:
                     # the training data's blind spot, not the buyer's fault
                     "pct_sem_compra_parecida": g.motivo.fillna("").str.startswith(
                         "não há compra parecida").mean()})
    return pd.DataFrame(rows).sort_values("itens", ascending=False)


def cmd_bias(queue_path, index_path, out_csv):
    """Ethics check: judged and flagged rates by sphere, on the audit queue."""
    with open(queue_path, encoding="utf-8") as fh:
        items = json.load(fh)["items"]
    t = bias_table(items)
    t.to_csv(out_csv, index=False)
    with pd.option_context("display.width", 160, "display.max_colwidth", 60):
        _log(t.to_string(index=False))
    paid = pd.Series([r.get("ex_esfera") for r in load_index(index_path).rows]).value_counts()
    _log("paid descriptions by sphere (what the model learned from): "
         + ", ".join(f"{SPHERES.get(k, k)} {v / paid.sum():.0%}" for k, v in paid.items()))
    flagged = pd.Series([it.get("tender") for it in items if it["estado"] == "revisar"])
    top = flagged.value_counts()
    _log(f"{len(flagged)} flags in {flagged.nunique()} tenders; the top 3 tenders hold "
         f"{top.head(3).sum()}")
    _log(f"WROTE {out_csv}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train, evaluate and batch-score the price model.")
    ap.add_argument("cmd", choices=["train", "final", "queue", "bias"])
    ap.add_argument("--features", default=None)
    ap.add_argument("--index", default="data/items/paid_index.jsonl")
    ap.add_argument("--bundle", default=BUNDLE)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "train":
        cmd_train(a.features or "data/items/features.jsonl", a.index, a.bundle)
    elif a.cmd == "final":
        cmd_final(a.features or "data/items/features.jsonl", a.bundle,
                  a.out or "data/items/results_final.csv")
    elif a.cmd == "queue":
        cmd_queue(a.features or "data/items/audit_features.jsonl", a.bundle,
                  a.out or "data/items/queue.json")
    else:
        cmd_bias("data/items/queue.json", a.index, a.out or "data/items/results_bias.csv")


if __name__ == "__main__":
    # Run through the real module so pickled classes resolve as
    # ``precojusto.model.PriceModel``, not ``__main__.PriceModel``.
    from precojusto import model as _self
    _self.main()
