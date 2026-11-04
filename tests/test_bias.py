from precojusto.model import bias_table


def _it(esfera, estado, motivo=None):
    return {"esfera": esfera, "estado": estado, "motivos": [motivo] if motivo else []}


def test_rates_are_per_sphere_and_flags_count_only_judged_items():
    items = ([_it("M", "revisar"), _it("M", "ok")]
             + [_it("M", "sem_confianca", "não há compra parecida já paga")] * 2
             + [_it("F", "ok")] * 3 + [_it("F", "revisar")])
    t = bias_table(items).set_index("esfera")
    assert t.loc["municipal", "itens"] == 4
    assert t.loc["municipal", "pct_julgados"] == 0.5
    assert t.loc["municipal", "pct_revisar_dos_julgados"] == 0.5
    assert t.loc["municipal", "pct_sem_compra_parecida"] == 0.5
    assert t.loc["federal", "pct_julgados"] == 1.0
    assert t.loc["federal", "pct_revisar_dos_julgados"] == 0.25
