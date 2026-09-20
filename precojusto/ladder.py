"""The ablation ladder, evaluated on a locked, group-aware holdout.

Every rung answers one question:

    1. mean               nothing learned? then nothing beats this
    2. catalogue median   what an auditor does today (needs catalogue codes)
    3. deterministic+GB   cheap text statistics, no LLM
    4. TF-IDF+GB          the classic way to use text
    5. LLM direct         what the LLM knows with no training at all
    6-8. LLM feats+model  the hypothesis: typed LLM features + shallow model
    9. everything         does raw text still add anything on top?

Method:

- 30% of the rows are set aside ONCE with a fixed seed, by group (tender or
  purchase id), and never touched while iterating. Default output is
  5-fold group CV on the other 70%.
- ``--final`` fits every learned rung on ALL of dev and predicts the holdout.
  Run it once, at the end. It never cross-validates inside the holdout.
- Target is log10(price). Prices span four orders of magnitude, so the
  error has to be proportional to mean anything.
- The direct-price columns are excluded from every learned rung, otherwise
  rung 6 would silently contain rung 5.

Usage::

    python -m precojusto.ladder                       # dev, CV
    python -m precojusto.ladder --final               # holdout, once
    python -m precojusto.ladder --features path.jsonl --quick
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .splits import group_folds, holdout_split

#: Columns that answer "what is the price?" directly. Never features.
DIRECT_COLUMNS = ["direct__log10", "direct__conf", "faixa_preco__nivel", "faixa_preco__disp"]
GROUP_KEYS = ("tender", "idCompra")
CATALOGUE_KEY = "codigoItemCatalogo"


# --------------------------------------------------------------------- data

def load(path):
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    df = pd.DataFrame(rows).reset_index(drop=True)
    jev = pd.DataFrame([r["jev"] for r in rows]).reset_index(drop=True)
    return df, jev


def group_ids(df):
    """Rows of one purchase share a group. Falls back to one row per group."""
    for k in GROUP_KEYS:
        if k in df.columns and df[k].notna().any():
            return df[k].fillna(df.index.to_series().map(lambda i: f"row{i}")).astype(str).to_numpy()
    return df.index.astype(str).to_numpy()


def deterministic_features(df):
    """Cheap text statistics plus one-hot sphere and modality. No LLM."""
    desc = df["item_desc"]
    det = pd.DataFrame({
        "desc_len": desc.str.len(),
        "desc_words": desc.str.split().str.len(),
        "n_digits": desc.str.count(r"\d"),
        "n_upper": desc.str.count(r"[A-ZÀ-Ý]"),
        "n_meses": desc.str.count(r"(?i)\bmes(?:es)?\b"),
        "maior_numero": desc.str.findall(r"\d+").apply(
            lambda xs: np.log10(max([int(x) for x in xs] or [1]) + 1)),
    }).astype(float)
    dummies = []
    for col, prefix in (("esfera", "esf"), ("modalidade", "mod")):
        if col in df.columns:
            dummies.append(pd.get_dummies(df[col].fillna("na"), prefix=prefix).astype(float))
    return pd.concat([det, *dummies], axis=1)


# ------------------------------------------------------------------- models

def make_models(quick=False):
    seed = 0
    trees = 50 if quick else 500
    iters = 40 if quick else 300
    return {
        "gb": lambda: HistGradientBoostingRegressor(
            max_iter=iters, learning_rate=0.06, max_leaf_nodes=15,
            min_samples_leaf=5, l2_regularization=1.0, random_state=seed),
        "rf": lambda: RandomForestRegressor(
            n_estimators=trees, min_samples_leaf=2, random_state=seed, n_jobs=-1),
        "mlp": lambda: make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(32,), alpha=1.0,
                         max_iter=300 if quick else 3000, early_stopping=True,
                         n_iter_no_change=30, random_state=seed)),
    }


def metrics(name, pred, y):
    err = pred - y
    ratio = 10 ** np.abs(err)
    return {"rung": name,
            "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "MAE": float(np.mean(np.abs(err))),
            "within_2x_%": float(100 * np.mean(ratio <= 2)),
            "within_5x_%": float(100 * np.mean(ratio <= 5))}


# -------------------------------------------------------------------- rungs

class Rung:
    """One rung = a way to produce predictions for test rows from train rows."""

    def __init__(self, name, fit_predict):
        self.name = name
        self.fit_predict = fit_predict  # (tr_idx, te_idx) -> predictions for te


def rung_mean(y):
    return Rung("1. mean", lambda tr, te: np.full(len(te), y[tr].mean()))


def rung_catalogue_median(y, codes):
    """Median log-price of TRAIN rows with the same catalogue code.

    Falls back to the train median when the code was never seen. This is
    the baseline an auditor already has; it only exists for catalogued items.
    """
    codes = np.asarray(codes)

    def fp(tr, te):
        table = pd.Series(y[tr]).groupby(codes[tr]).median()
        fallback = float(np.median(y[tr]))
        return np.array([table.get(c, fallback) for c in codes[te]])
    return Rung("2. catalogue median", fp)


def rung_direct(direct):
    return Rung("5. LLM direct (no model)", lambda tr, te: direct[te])


def rung_model(name, make_model, y, matrix=None, text=None):
    """Fit ``make_model()`` on train rows of [tfidf(text) | matrix]."""
    def fp(tr, te):
        parts_tr, parts_te = [], []
        if text is not None:
            vec = TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2),
                                  max_features=20000, strip_accents="unicode")
            parts_tr.append(vec.fit_transform(text[tr]).toarray())
            parts_te.append(vec.transform(text[te]).toarray())
        if matrix is not None:
            parts_tr.append(matrix[tr])
            parts_te.append(matrix[te])
        m = make_model()
        m.fit(np.hstack(parts_tr), y[tr])
        return m.predict(np.hstack(parts_te))
    return Rung(name, fp)


def build_ladder(df, jev, y, quick=False):
    models = make_models(quick)
    det = deterministic_features(df).to_numpy(float)
    core = jev.drop(columns=[c for c in DIRECT_COLUMNS if c in jev.columns]).to_numpy(float)
    core_det = np.hstack([core, det])
    text = df["item_desc"].to_numpy()
    direct = jev["direct__log10"].to_numpy(float)

    ladder = [rung_mean(y)]
    if CATALOGUE_KEY in df.columns and df[CATALOGUE_KEY].notna().any():
        ladder.append(rung_catalogue_median(y, df[CATALOGUE_KEY].astype(str)))
    ladder += [
        rung_model("3. deterministic + GB", models["gb"], y, matrix=det),
        rung_model("4. TF-IDF + GB", models["gb"], y, text=text),
        rung_direct(direct),
        rung_model("6. LLM features + det (RF)", models["rf"], y, matrix=core_det),
        rung_model("7. LLM features + det (GB)", models["gb"], y, matrix=core_det),
        rung_model("8. LLM features + det (MLP)", models["mlp"], y, matrix=core_det),
        rung_model("9. LLM + TF-IDF + det (GB)", models["gb"], y, matrix=core_det, text=text),
    ]
    return ladder, core_det


# ------------------------------------------------------------------ running

def run_dev(ladder, y, groups, dev, folds=5, seed=0):
    """Cross-validate inside dev. Groups never straddle a fold."""
    results = []
    splits = list(group_folds(groups[dev], folds, seed))
    for rung in ladder:
        pred = np.zeros(len(dev))
        for tr, te in splits:
            pred[te] = rung.fit_predict(dev[tr], dev[te])
        results.append(metrics(rung.name, pred, y[dev]))
    return results


def run_final(ladder, y, dev, hold):
    """Fit on all of dev, predict the holdout. Once."""
    return [metrics(r.name, r.fit_predict(dev, hold), y[hold]) for r in ladder]


def feature_importance(make_model, matrix, columns, y, seed=0, top=12):
    from sklearn.inspection import permutation_importance
    m = make_model().fit(matrix, y)
    pi = permutation_importance(m, matrix, y, n_repeats=10, random_state=seed,
                                scoring="neg_mean_absolute_error")
    return (pd.Series(pi.importances_mean, index=columns)
            .sort_values(ascending=False).head(top))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run the ablation ladder.")
    ap.add_argument("--features", default="data/pilot/features.jsonl")
    ap.add_argument("--out", default=None, help="CSV path (default derived from mode)")
    ap.add_argument("--final", action="store_true", help="evaluate on the holdout, ONCE")
    ap.add_argument("--quick", action="store_true", help="small models, for smoke tests")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--holdout", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=12345, help="holdout seed; never change it")
    args = ap.parse_args(argv)

    df, jev = load(args.features)
    y = np.log10(df["valor_unit"].to_numpy(float))
    groups = group_ids(df)
    dev, hold = holdout_split(groups, args.holdout, args.seed)
    ladder, core_det = build_ladder(df, jev, y, args.quick)

    if args.final:
        print("\n*** FINAL: fitting on dev, scoring the locked holdout. "
              "This is meant to happen once. ***", file=sys.stderr)
        res = run_final(ladder, y, dev, hold)
        label, n = "HOLDOUT (final)", len(hold)
    else:
        res = run_dev(ladder, y, groups, dev, args.folds)
        label, n = f"DEV, {args.folds}-fold group CV", len(dev)

    out = pd.DataFrame(res)
    print(f"\n{n} rows · {label} · {len(np.unique(groups))} groups · "
          f"holdout locked = {len(hold)} rows\n")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    path = args.out or ("results_final.csv" if args.final else "results_dev.csv")
    out.to_csv(path, index=False)

    if not args.final and not args.quick:
        det_cols = list(deterministic_features(df).columns)
        jev_cols = [c for c in jev.columns if c not in DIRECT_COLUMNS]
        imp = feature_importance(make_models()["gb"], core_det[dev],
                                 jev_cols + det_cols, y[dev])
        print("\nmost useful features (permutation importance, GB on dev):")
        for k, v in imp.items():
            print(f"  {v:+.4f}  {k}")


if __name__ == "__main__":
    main()
