import os

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from atributos import PROC

MULT_CAPITAO = 1.5
PATRIMONIO_INICIAL = 100.0
ANO_RESERVA_LUXO = 2025
PEN_DEF = 2.0
PEN_OF3 = 2.0
PEN_OF4 = 5.0
POS = ["GOL", "LAT", "ZAG", "MEI", "ATA"]
FORMACOES = {
    "3-4-3": (1, 0, 3, 4, 3), "3-5-2": (1, 0, 3, 5, 2), "4-3-3": (1, 2, 2, 3, 3), "4-4-2": (1, 2, 2, 4, 2),
    "4-5-1": (1, 2, 2, 5, 1), "5-3-2": (1, 2, 3, 3, 2), "5-4-1": (1, 2, 3, 4, 1),
}
ESTRATEGIAS = {"modelo": "pred", "baseline (média do Cartola)": "prev_media"}


def preparar():
    """Junta os candidatos com a previsão do modelo. Para os técnicos, os pontos esperados são a média
    das 5 rodadas anteriores (ou a média do ano), e só entram técnicos de clubes que jogam na rodada.
    """
    b = pd.read_pickle(os.path.join(PROC, "atributos_final.pkl"))
    prev = pd.read_pickle(os.path.join(PROC, "previsoes_final.pkl"))
    cand = b[b["candidato"]].merge(prev, on=["ano", "rodada", "atleta_id"])
    cand = cand[cand["preco_pre"].notna()]

    tec = pd.read_csv(os.path.join(PROC, "tecnicos_2018_2026.csv.gz")).sort_values(["ano", "atleta_id", "rodada"])
    media5 = tec.groupby(["ano", "atleta_id"])["pontos"].transform(lambda s: s.shift().rolling(5, min_periods=1).mean())
    tec["e_pts"] = media5.fillna(tec.groupby("ano")["pontos"].transform("mean"))
    jogam = cand[["ano", "rodada", "clube_id"]].drop_duplicates()
    tec = tec.merge(jogam, on=["ano", "rodada", "clube_id"])
    return cand, tec


def escalar(c, t, pat, col):
    """Monta o melhor time da rodada por programação inteira (cada escolha é uma variável 0 ou 1).
    Objetivo: soma dos pontos previstos (coluna `col`), com o capitão valendo 1,5x, mais o técnico,
    menos as penalidades por concentrar atletas do mesmo clube.
    Regras: formação válida, 1 técnico, 1 capitão que esteja escalado e custo <= patrimônio.
    Por clube: até 2 defensores (GOL/LAT/ZAG) livres, o 3º paga PEN_DEF e o 4º é proibido;
    até 2 meias/atacantes livres, o 3º paga PEN_OF3, o 4º paga também PEN_OF4 e o 5º é proibido.
    """
    n, m, k = len(c), len(t), len(FORMACOES)
    clubes = sorted(c["clube_id"].unique())
    nc = len(clubes)
    nv = 2 * n + m + k + 3 * nc
    ix, icap, itec, iform = 0, n, 2 * n, 2 * n + m
    idef, iof3, iof4 = iform + k, iform + k + nc, iform + k + 2 * nc

    obj = np.zeros(nv)
    obj[ix:ix + n] = c[col].values
    obj[icap:icap + n] = (MULT_CAPITAO - 1) * c[col].values
    obj[itec:itec + m] = t["e_pts"].values
    obj[idef:idef + nc] = -PEN_DEF
    obj[iof3:iof3 + nc] = -PEN_OF3
    obj[iof4:iof4 + nc] = -PEN_OF4

    linhas, lo, hi = [], [], []

    def regra(lin, minimo, maximo):
        linhas.append(lin)
        lo.append(minimo)
        hi.append(maximo)

    pos = c["posicao"].values
    for p_i, p in enumerate(POS):
        lin = np.zeros(nv)
        lin[ix + np.where(pos == p)[0]] = 1
        for f_i, qtd in enumerate(FORMACOES.values()):
            lin[iform + f_i] = -qtd[p_i]
        regra(lin, 0, 0)
    for ini, tam in [(iform, k), (itec, m), (icap, n)]:
        lin = np.zeros(nv)
        lin[ini:ini + tam] = 1
        regra(lin, 1, 1)
    for i in range(n):
        lin = np.zeros(nv)
        lin[icap + i] = 1
        lin[ix + i] = -1
        regra(lin, -np.inf, 0)
    lin = np.zeros(nv)
    lin[ix:ix + n] = c["preco_pre"].values
    lin[itec:itec + m] = t["preco_pre"].values
    regra(lin, -np.inf, pat)

    clube = c["clube_id"].values
    defesa = np.isin(pos, ["GOL", "ZAG", "LAT"])
    ataque = np.isin(pos, ["MEI", "ATA"])
    for j, cl in enumerate(clubes):
        d = ix + np.where((clube == cl) & defesa)[0]
        a = ix + np.where((clube == cl) & ataque)[0]
        lin = np.zeros(nv); lin[d] = 1; lin[idef + j] = -1
        regra(lin, -np.inf, 2)
        lin = np.zeros(nv); lin[d] = 1
        regra(lin, -np.inf, 3)
        lin = np.zeros(nv); lin[a] = 1; lin[iof3 + j] = -1; lin[iof4 + j] = -1
        regra(lin, -np.inf, 2)
        lin = np.zeros(nv); lin[a] = 1
        regra(lin, -np.inf, 4)

    res = milp(-obj, constraints=LinearConstraint(np.array(linhas), lo, hi), integrality=np.ones(nv),
               bounds=Bounds(0, 1), options={"time_limit": 20})
    x = res.x.round().astype(bool)
    return x[ix:ix + n], x[icap:icap + n], x[itec:itec + m]


def escolher_reservas(c, tit, luxo, col):
    """Um reserva por posição: o de maior previsão entre os não escalados que custam no máximo o titular
    mais barato da posição. A partir de 2025, o reserva com maior vantagem prevista sobre o titular mais
    fraco da sua posição vira o Reserva de Luxo.
    """
    res = {}
    livres = c.drop(index=tit.index)
    for p, g in tit.groupby("posicao"):
        pode = livres[(livres["posicao"] == p) & (livres["preco_pre"] <= g["preco_pre"].min())]
        if len(pode):
            res[p] = {"idx": pode[col].idxmax(), "luxo": False}
    if luxo and res:
        vantagem = {}
        for p, r in res.items():
            vantagem[p] = c.loc[r["idx"], col] - tit.loc[tit["posicao"] == p, col].min()
        res[max(vantagem, key=vantagem.get)]["luxo"] = True
    return res


def aplicar_reservas(c, tit, capitao, res):
    """Pontos e variação reais do time depois das substituições automáticas.
    Titular que não jogou sai e entra o reserva da posição (se o reserva jogou); capitão ausente perde
    o bônus. O Reserva de Luxo troca com o titular mais fraco da posição (menos o capitão) se pontuou mais.
    Só conta a variação de preço de quem pontuou para o time.
    """
    pts = tit["pontos_esc"].sum() + (MULT_CAPITAO - 1) * c.loc[capitao, "pontos_esc"]
    var = tit["variacao"].sum()
    ausentes = int((~tit["em_campo"]).sum())
    subs = luxo = 0
    for p, r in res.items():
        reserva = c.loc[r["idx"]]
        if not reserva["em_campo"]:
            continue
        da_pos = tit[tit["posicao"] == p]
        faltou = da_pos[~da_pos["em_campo"]]
        if len(faltou):
            sai = faltou.index[0]
            pts += reserva["pontos_esc"]
            var += reserva["variacao"] - tit.loc[sai, "variacao"]
            subs += 1
        elif r["luxo"]:
            outros = da_pos.drop(index=[capitao], errors="ignore")
            if len(outros) and reserva["pontos_esc"] > outros["pontos_esc"].min():
                sai = outros["pontos_esc"].idxmin()
                pts += reserva["pontos_esc"] - tit.loc[sai, "pontos_esc"]
                var += reserva["variacao"] - tit.loc[sai, "variacao"]
                luxo += 1
    return pts, var, ausentes, subs, luxo


def simular(cand, tec, ano, col):
    """Joga uma temporada inteira: a cada rodada escala o time com o patrimônio atual, escolhe os reservas,
    aplica os pontos e as variações reais e atualiza o patrimônio. Devolve o histórico rodada a rodada.
    """
    pat = PATRIMONIO_INICIAL
    hist = []
    for rodada in sorted(cand.loc[cand["ano"] == ano, "rodada"].unique()):
        c = cand[(cand["ano"] == ano) & (cand["rodada"] == rodada)].reset_index(drop=True)
        t = tec[(tec["ano"] == ano) & (tec["rodada"] == rodada)].reset_index(drop=True)
        if len(t) == 0 or len(c) < 20:
            continue
        x, cap, tj = escalar(c, t, pat, col)
        tit = c[x]
        capitao = c.index[cap][0]
        res = escolher_reservas(c, tit, ano >= ANO_RESERVA_LUXO, col)
        pts, var, ausentes, subs, luxo = aplicar_reservas(c, tit, capitao, res)
        pts = pts + t.loc[tj, "pontos"].sum()
        var = var + t.loc[tj, "variacao"].sum()
        pat += var
        hist.append({"ano": ano, "rodada": rodada, "pontos": pts, "patrimonio": pat,
                     "custo": tit["preco_pre"].sum() + t.loc[tj, "preco_pre"].sum(),
                     "titulares_ausentes": ausentes, "substituicoes": subs, "zerados": ausentes - subs,
                     "trocas_luxo": luxo, "pontos_previstos": tit[col].sum()})
    return pd.DataFrame(hist)


def resumo(h):
    """Resume uma temporada simulada: pontos, risco, patrimônio e substituições."""
    p = h["pontos"].sort_values()
    piores = p.head(max(1, round(0.1 * len(p))))
    return {"rodadas simuladas": len(h), "pontos totais": h["pontos"].sum(),
            "pontos médios/rodada": h["pontos"].mean(), "desvio/rodada": h["pontos"].std(),
            "média 10% piores rodadas": piores.mean(), "melhor rodada": h["pontos"].max(),
            "patrimônio final (C$)": h["patrimonio"].iloc[-1],
            "valorização (C$)": h["patrimonio"].iloc[-1] - PATRIMONIO_INICIAL,
            "maior patrimônio (C$)": h["patrimonio"].max(),
            "titulares ausentes": h["titulares_ausentes"].sum(), "cobertos por reserva": h["substituicoes"].sum(),
            "titulares zerados": h["zerados"].sum(), "trocas de luxo": h["trocas_luxo"].sum()}


def main():
    """Simula 2024, 2025 e 2026 com o modelo e com o baseline (média do Cartola), nas mesmas regras,
    e salva o resumo por temporada e o histórico rodada a rodada.
    """
    cand, tec = preparar()
    linhas, hists = [], []
    for nome, col in ESTRATEGIAS.items():
        for ano in (2024, 2025, 2026):
            h = simular(cand, tec, ano, col)
            hists.append(h.assign(estrategia=nome))
            linhas.append({"estratégia": nome, "temporada": ano, **resumo(h)})
    res = pd.DataFrame(linhas)
    pd.concat(hists).to_csv(os.path.join(PROC, "simulacao_final.csv"), index=False)
    res.to_csv(os.path.join(PROC, "resultado_final.csv"), index=False)
    pd.set_option("display.width", 250)
    print(res.set_index(["estratégia", "temporada"]).round(2).T.to_string())


if __name__ == "__main__":
    main()
