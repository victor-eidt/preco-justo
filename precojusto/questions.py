"""The typed question set: what we ask the LLM about every item description.

This is the schema of the whole project. Each question returns a *typed*
answer rather than free text, so the answers become columns directly:

- ``Score``: an ordered scale. Becomes TWO columns: the probability-weighted
  level, and the spread of the distribution. The spread is a feature in its
  own right: when the model cannot tell how specific a description is, the
  description is usually bad.
- ``Noul``: one probability in [0, 1]. One column.
- ``Choice``: a categorical. One column per option, plus a confidence column.

The question texts are in Portuguese on purpose: they are read together
with the item description, which is Portuguese, and translating the rubric
would move it away from the vocabulary of the text it judges.

``faixa_preco`` is special. It asks for the price outright and exists only
so the ladder can measure "what the LLM already knows without training".
It is excluded from the feature matrix of every learned rung, otherwise
those rungs would silently contain it. See ``ladder.DIRECT_COLUMNS``.

Changing any question changes ``QUESTIONS_FINGERPRINT``, which invalidates
cached answers for that question set automatically.
"""

import hashlib
import json

from typesafe_sdk import Choice, Noul, Score

ESPECIFICIDADE = [
    "Genérica: nomeia só a família do produto, sem nenhuma característica "
    "(ex.: 'material de informática', 'software')",
    "Categoria clara, mas sem especificação técnica (ex.: 'notebook', "
    "'licença de sistema web')",
    "Traz alguma especificação técnica relevante (ex.: 'notebook i5, 8GB')",
    "Especificação técnica completa e detalhada, com várias características",
    "Tão específica que só um produto ou fornecedor atende (marca, modelo "
    "e configuração exata)",
]

TIER = [
    "Básico / de entrada, o mais barato da categoria",
    "Padrão de escritório, uso administrativo comum",
    "Intermediário, acima do básico mas não profissional",
    "Profissional / corporativo, exigências de desempenho explícitas",
    "Alto desempenho / datacenter / missão crítica",
]

ESCOPO = [
    "Uma peça ou item avulso, isolado",
    "Um equipamento completo, ou uma licença única",
    "Um conjunto de itens, ou um serviço delimitado",
    "Um sistema integrado, com implantação e vários componentes",
    "Uma solução corporativa completa, abrangendo toda a organização",
]

RECORRENCIA = [
    "Entrega única, acaba quando é entregue",
    "Entrega única com garantia ou suporte por um período",
    "Serviço por prazo determinado, com começo e fim",
    "Serviço continuado, cobrado por período (mensalidade, assinatura)",
]

ESFORCO = [
    "Nenhum: é só entregar o produto",
    "Baixo: instalação simples",
    "Médio: configuração, parametrização ou treinamento",
    "Alto: migração de dados, integração com outros sistemas",
    "Muito alto: desenvolvimento sob medida",
]

PADRONIZACAO = [
    "Commodity: produto de prateleira, muitos fornecedores equivalentes",
    "Padrão de mercado, com pequenas variações entre fornecedores",
    "Configurável: o mesmo produto ajustado ao cliente",
    "Sob medida: feito especificamente para este órgão",
]

CATEGORIES = {
    "licenca_software": "Licença, assinatura ou locação de software pronto, incluindo SaaS",
    "desenvolvimento_sistema": "Desenvolvimento, customização, implantação ou migração de sistema",
    "hardware_computador": "Computador, notebook, servidor, tablet ou celular",
    "periferico_suprimento": "Periférico ou suprimento: toner, cartucho, teclado, mouse, monitor, cabo",
    "infraestrutura_rede": "Equipamento de rede ou datacenter: switch, roteador, firewall, rack, nobreak, storage",
    "conectividade": "Serviço de conectividade: link de internet, telefonia, transmissão de dados",
    "hospedagem_nuvem": "Hospedagem, nuvem, datacenter como serviço, backup externo",
    "manutencao_suporte": "Manutenção, assistência técnica ou suporte a equipamentos existentes",
    "outro": "Qualquer outra coisa que não caiba nas opções acima",
}

QUESTIONS = {
    "especificidade": Score(
        instructions="Quão específica e detalhada é a descrição deste item de "
                     "licitação pública? Julgue o texto da descrição, não o "
                     "produto em si.",
        criteria=ESPECIFICIDADE),
    "tier_tecnico": Score(
        instructions="Que nível técnico o item exige, pelo que a descrição "
                     "deixa entender?",
        criteria=TIER),
    "escopo": Score(
        instructions="Qual é a abrangência do que está sendo comprado neste item?",
        criteria=ESCOPO),
    "recorrencia": Score(
        instructions="Este item é uma entrega única ou um serviço que se repete "
                     "ao longo do tempo?",
        criteria=RECORRENCIA),
    "esforco": Score(
        instructions="Quanto trabalho de implantação o fornecedor precisa fazer "
                     "depois de entregar?",
        criteria=ESFORCO),
    "padronizacao": Score(
        instructions="O quanto este item é um produto padronizado de mercado, em "
                     "oposição a algo feito sob medida?",
        criteria=PADRONIZACAO),

    "tem_marca": Noul(
        instructions="A descrição cita uma marca, fabricante ou modelo específico."),
    "e_licenca": Noul(
        instructions="O item é uma licença, assinatura ou direito de uso de software."),
    "e_hardware": Noul(
        instructions="O item é um equipamento físico que será entregue."),
    "inclui_mao_de_obra": Noul(
        instructions="O preço deste item inclui mão de obra ou trabalho humano, "
                     "não apenas um produto."),
    "descricao_truncada": Noul(
        instructions="A descrição está claramente incompleta, cortada no meio, "
                     "ou termina anunciando um conteúdo que não aparece."),
    "menciona_prazo": Noul(
        instructions="A descrição menciona um prazo, duração, período de vigência "
                     "ou quantidade de meses."),
    "exige_certificacao": Noul(
        instructions="A descrição exige certificação, homologação, garantia "
                     "estendida ou conformidade com alguma norma."),

    "categoria": Choice(
        instructions="Em que categoria de gasto público de TI este item se "
                     "encaixa melhor?",
        criteria=CATEGORIES),

    # Scale. Descriptions often state quantity or duration ("113 linhas",
    # "24 meses"), and nothing else in the set was reading that.
    "escala_quantidade": Score(
        instructions="Que volume de unidades esta contratação abrange, pelo "
                     "que a descrição deixa entender?",
        criteria=[
            "Uma única unidade, ou um serviço indivisível",
            "Poucas unidades (2 a 10)",
            "Dezenas de unidades",
            "Centenas de unidades",
            "Milhares de unidades, ou toda uma rede/órgão",
        ]),
    "duracao": Score(
        instructions="Por quanto tempo esta contratação vale, pelo que a "
                     "descrição deixa entender?",
        criteria=[
            "Sem duração: entrega única, acaba na entrega",
            "Alguns meses (até 6)",
            "Cerca de 12 meses",
            "24 meses ou mais",
        ]),
    "abrangencia": Score(
        instructions="Qual a abrangência organizacional desta contratação?",
        criteria=[
            "Um setor ou unidade específica",
            "Um órgão inteiro (uma prefeitura, uma secretaria)",
            "Vários órgãos, ou um município inteiro",
            "Um estado inteiro, ou abrangência nacional",
        ]),
    "quantidade_explicita": Noul(
        instructions="A descrição informa explicitamente um número de "
                     "unidades, de meses, ou de postos de trabalho."),

    # The direct-price question. Ladder rung "LLM direct" only. Bands are
    # for the UNIT price of one item.
    "faixa_preco": Score(
        instructions="Qual o preço UNITÁRIO provável deste item numa compra "
                     "pública brasileira, em reais? O preço é por UMA unidade "
                     "da unidade de fornecimento indicada, não pelo lote todo.",
        criteria=[
            "Até R$ 50",
            "Entre R$ 50 e R$ 200",
            "Entre R$ 200 e R$ 800",
            "Entre R$ 800 e R$ 3.000",
            "Entre R$ 3.000 e R$ 12.000",
            "Entre R$ 12.000 e R$ 50.000",
            "Acima de R$ 50.000",
        ]),
}

#: log10 of the geometric midpoint of each ``faixa_preco`` band.
BAND_LOG10 = [1.35, 2.00, 2.60, 3.19, 3.79, 4.39, 5.00]

SCORES = [k for k, v in QUESTIONS.items() if isinstance(v, Score)]
NOULS = [k for k, v in QUESTIONS.items() if isinstance(v, Noul)]


def questions_fingerprint(questions=QUESTIONS):
    """Short hash of the question set. Part of every cache key."""
    blob = json.dumps({k: repr(v) for k, v in sorted(questions.items())},
                      ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


QUESTIONS_FINGERPRINT = questions_fingerprint()
