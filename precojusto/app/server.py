"""The demo app: a triage queue and a "try it yourself" item analysis.

Reads only files written by the batch job (``model.py queue``), so the queue
never depends on PNCP being up. Only ``POST /api/analisar`` runs live: one
LLM call (Jev) plus the trained model, a second or less.

Usage::

    python -m precojusto.app            # http://127.0.0.1:8000
"""

import json
import os
import pickle
import threading
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..features import api_key, cache_key, load_cache, save_cache, state_for, to_features
from ..model import BUNDLE, MARGIN, feature_matrix, verdict
from ..questions import QUESTIONS

ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).resolve().parent / "static"
QUEUE = ROOT / "data/items/queue.json"
LIVE_CACHE = ROOT / "data/items/live_cache.json"

# --------------------------------------------------------------- readings

CAT_LABEL = {
    "hardware_computador": "Computador, notebook ou servidor",
    "periferico_suprimento": "Periférico ou suprimento",
    "infraestrutura_rede": "Rede ou datacenter",
    "licenca_software": "Licença de software",
    "desenvolvimento_sistema": "Desenvolvimento de sistema",
    "conectividade": "Conectividade",
    "hospedagem_nuvem": "Hospedagem ou nuvem",
    "manutencao_suporte": "Manutenção ou suporte",
    "outro": "Outro",
}
SCALES = {
    "especificidade": ("Especificação", ["Genérica", "Só a categoria", "Alguma especificação",
                                         "Especificação completa", "Um único produto atende"]),
    "tier_tecnico": ("Nível técnico", ["Básico", "Escritório", "Intermediário",
                                       "Profissional", "Alto desempenho"]),
    "padronizacao": ("Padronização", ["Commodity", "Padrão de mercado", "Configurável",
                                      "Sob medida"]),
    "escala_quantidade": ("Volume", ["Uma unidade", "2 a 10", "Dezenas", "Centenas",
                                     "Milhares"]),
}
FLAGS = {
    "e_hardware__p": "É um equipamento físico",
    "tem_marca__p": "Cita marca ou modelo",
    "exige_certificacao__p": "Exige garantia ou certificação",
    "descricao_truncada__p": "Descrição cortada ou incompleta",
}


def readings(f):
    """The LLM's typed answers, as a person would read them."""
    cats = {k[5:]: v for k, v in f.items() if k.startswith("cat__") and k != "cat__conf"}
    top = max(cats, key=cats.get)
    out = [{"rotulo": "Categoria", "valor": CAT_LABEL.get(top, top), "nivel": cats[top],
            "tipo": "prob"}]
    for key, (label, levels) in SCALES.items():
        lvl = f[f"{key}__nivel"]
        out.append({"rotulo": label, "valor": levels[min(len(levels) - 1, round(lvl))],
                    "nivel": lvl / (len(levels) - 1), "tipo": "escala",
                    "incerteza": f[f"{key}__disp"]})
    for key, label in FLAGS.items():
        out.append({"rotulo": label, "valor": f"{100 * f[key]:.0f}%", "nivel": f[key],
                    "tipo": "prob"})
    return out


def present(item):
    """Full sheet: the stored item plus readable LLM answers."""
    return {**{k: v for k, v in item.items() if k != "jev"}, "leituras": readings(item["jev"])}


def light(item):
    keep = ("id", "descricao", "orgao", "uf", "esfera", "estado", "estimado", "previsto",
            "faixa_min", "faixa_max", "razao", "quantidade", "unidade", "motivos")
    return {k: item.get(k) for k in keep}


# ------------------------------------------------------------------ state

class State:
    def __init__(self):
        with open(BUNDLE if os.path.isabs(BUNDLE) else ROOT / BUNDLE, "rb") as fh:
            b = pickle.load(fh)
        self.model, self.index = b["model"], b["index"]
        self.dev_report, self.holdout_report = b.get("dev_report"), b.get("holdout_report")
        with open(QUEUE, encoding="utf-8") as fh:
            q = json.load(fh)
        self.items = {it["id"]: it for it in q["items"]}
        self.order = [it["id"] for it in q["items"]]
        self.cache = load_cache(str(LIVE_CACHE))
        self.lock = threading.Lock()
        self._client = None

    def client(self):
        if self._client is None:
            from typesafe_sdk import TypeSafeClient
            self._client = TypeSafeClient(api_key=api_key())
        return self._client

    def jev(self, row):
        key = cache_key(row)
        if key in self.cache:
            return self.cache[key]
        resp = self.client().system_one(state=state_for(row), questions=QUESTIONS,
                                        model=os.environ.get("JEV_MODEL", "jev-latest"))
        feats = to_features(resp.answers)
        with self.lock:
            self.cache[key] = feats
            save_cache(self.cache, str(LIVE_CACHE))
        return feats


app = FastAPI(title="Preço Justo")
S = None


def state():
    global S
    if S is None:
        S = State()
    return S


class Pedido(BaseModel):
    descricao: str = Field(min_length=8, max_length=700)
    preco: float | None = Field(default=None, gt=0)
    quantidade: float | None = Field(default=None, gt=0)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/fila")
def fila():
    s = state()
    items = [light(s.items[i]) for i in s.order]
    counts = pd.Series([it["estado"] for it in items]).value_counts().to_dict()
    tenders = len({s.items[i].get("tender") for i in s.order})
    return {"itens": items, "contagem": counts, "licitacoes": tenders,
            "dev": s.dev_report, "holdout": s.holdout_report, "margem": MARGIN}


@app.get("/api/item/{item_id}")
def item(item_id: int):
    s = state()
    if item_id not in s.items:
        raise HTTPException(404, "item não encontrado")
    return present(s.items[item_id])


@app.post("/api/analisar")
def analisar(p: Pedido):
    s = state()
    desc = " ".join(p.descricao.split())
    row = {"item_desc": desc, "unidade": "UN", "quantidade": p.quantidade or 1.0,
           "tipo": "Material", "objeto": ""}
    try:
        f = s.jev(row)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"o LLM não respondeu ({type(e).__name__}); tente de novo") from e
    jev = pd.DataFrame([f])
    X = feature_matrix([desc], jev, s.model.columns, s.index)
    mid, lo, hi = (v[0] for v in s.model.predict(X))
    v = verdict(p.preco, mid, lo, hi, hi - lo, s.model.width_cut, f, top_sim=float(X[0, -1]))
    it = {"id": None, "descricao": desc, "estimado": p.preco, "quantidade": p.quantidade,
          "unidade": "UN", **v, "jev": f, "llm_direto": 10 ** f["direct__log10"],
          "parecidas": s.index.similar(desc)}
    return present(it)
