from types import SimpleNamespace

from precojusto.features import cache_key, state_for, to_features
from precojusto.questions import (
    BAND_LOG10,
    CATEGORIES,
    NOULS,
    QUESTIONS,
    QUESTIONS_FINGERPRINT,
    SCORES,
    questions_fingerprint,
)


def fake_answers(spread=False):
    ans = {}
    for k in SCORES:
        n = len(QUESTIONS[k].criteria)
        probs = {str(i): 1.0 / n for i in range(n)} if spread else {"0": 1.0}
        ans[k] = SimpleNamespace(probabilities=probs, confidence=0.9)
    for k in NOULS:
        ans[k] = SimpleNamespace(noul=0.25)
    ans["categoria"] = SimpleNamespace(
        probabilities={"hardware_computador": 0.7, "outro": 0.3}, confidence=0.8)
    return ans


def test_score_becomes_level_and_spread():
    f = to_features(fake_answers())
    assert f["especificidade__nivel"] == 0.0
    assert f["especificidade__disp"] == 0.0
    g = to_features(fake_answers(spread=True))
    assert g["especificidade__nivel"] == 2.0  # uniform over 5 levels
    assert g["especificidade__disp"] > 0


def test_direct_price_is_band_midpoint():
    f = to_features(fake_answers())
    assert f["direct__log10"] == BAND_LOG10[0]


def test_choice_expands_to_one_column_per_option():
    f = to_features(fake_answers())
    for opt in CATEGORIES:
        assert f"cat__{opt}" in f
    assert f["cat__hardware_computador"] == 0.7
    assert f["cat__outro"] == 0.3


def test_column_count_is_stable():
    f = to_features(fake_answers())
    expected = 2 * len(SCORES) + len(NOULS) + len(CATEGORIES) + 1 + 2
    assert len(f) == expected == 40


def test_fingerprint_is_in_cache_key_but_not_in_state():
    row = {"item_desc": "notebook i5 8gb", "quantidade": 2}
    assert cache_key(row).startswith(QUESTIONS_FINGERPRINT + "|")
    assert QUESTIONS_FINGERPRINT not in state_for(row)


def test_changing_a_question_changes_the_fingerprint():
    from typesafe_sdk import Noul
    q = dict(QUESTIONS, tem_marca=Noul(instructions="something else"))
    assert questions_fingerprint(q) != QUESTIONS_FINGERPRINT
