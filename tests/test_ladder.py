import numpy as np
import pandas as pd

from precojusto.ladder import (
    build_ladder,
    deterministic_features,
    group_ids,
    run_dev,
    run_final,
    rung_catalogue_median,
)
from precojusto.splits import holdout_split


def toy(n=90, seed=0):
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 6, n)
    price = 10 ** (2 + codes * 0.4 + rng.normal(0, 0.1, n))
    df = pd.DataFrame({
        "item_desc": [f"item tipo {c} com {rng.integers(1, 99)} unidades" for c in codes],
        "valor_unit": price,
        "esfera": rng.choice(["F", "M"], n),
        "modalidade": rng.choice(["Pregão", "Dispensa"], n),
        "idCompra": rng.integers(0, 30, n),
        "codigoItemCatalogo": codes,
    })
    jev = pd.DataFrame({
        "especificidade__nivel": codes + rng.normal(0, 0.2, n),
        "especificidade__disp": rng.random(n),
        "direct__log10": np.log10(price) + rng.normal(0, 0.3, n),
        "direct__conf": rng.random(n),
        "faixa_preco__nivel": codes.astype(float),
        "faixa_preco__disp": rng.random(n),
    })
    return df, jev


def test_group_ids_prefer_purchase_id():
    df, _ = toy()
    assert len(set(group_ids(df))) == len(set(df["idCompra"]))


def test_catalogue_median_uses_train_rows_only():
    y = np.array([1.0, 1.0, 5.0, 9.0])
    codes = np.array(["a", "a", "a", "b"])
    rung = rung_catalogue_median(y, codes)
    pred = rung.fit_predict(np.array([0, 1, 3]), np.array([2]))
    assert pred[0] == 1.0  # median of a's in train, the 5.0 itself is unseen
    pred = rung.fit_predict(np.array([0, 1]), np.array([3]))
    assert pred[0] == 1.0  # code b never seen -> train median


def test_ladder_runs_dev_and_final_without_direct_leak():
    df, jev = toy()
    y = np.log10(df["valor_unit"].to_numpy())
    ladder, core_det = build_ladder(df, jev, y, quick=True)
    assert core_det.shape[1] == 2 + deterministic_features(df).shape[1]
    groups = group_ids(df)
    dev, hold = holdout_split(groups, 0.3, 12345)
    res = run_dev(ladder, y, groups, dev, folds=3)
    assert [r["rung"][:2] for r in res] == ["1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9."]
    final = run_final(ladder, y, dev, hold)
    assert all(np.isfinite(r["RMSE"]) for r in final)
    assert final[1]["RMSE"] < final[0]["RMSE"]  # catalogue median beats the mean here
