import math
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from atributos import CH, PROC, SCOUTS, historico_atleta, media_posicao
from penalti import PRIOR, Cobrador
from ponderado import media_ponderada, soma_ponderada

SEED = 42
ANOS_TREINO = list(range(2018, 2025))
OF = ["FF", "FD", "FT", "G", "A", "FS"]
DE = ["DS_h", "FC"]
ORDEM = ["ano", "rodada"]


def titularidade(vdg):
    """Descobre quem foi titular: em cada clube e rodada, o goleiro com mais ações e os 10 jogadores
    de linha com mais ações. Quem não jogou fica com 0.
    """
    jog = vdg[vdg["em_campo"] & vdg["scouts_confiaveis"]].copy()
    acoes = [s for s in SCOUTS if s not in ("SG", "GS")] + ["DE"]
    jog["acoes"] = jog[acoes].sum(axis=1)
    goleiro = jog["posicao"] == "GOL"
    jog["rank"] = jog.groupby(["ano", "rodada", "clube_id", goleiro])["acoes"].rank(ascending=False, method="first")
    jog["titular_inf"] = np.where(goleiro, jog["rank"] <= 1, jog["rank"] <= 10).astype(float)

    vdg = vdg.merge(jog[["ano", "rodada", "atleta_id", "titular_inf"]], on=["ano", "rodada", "atleta_id"], how="left")
    vdg.loc[~vdg["em_campo"], "titular_inf"] = 0.0
    return vdg.sort_values(["ano", "atleta_id", "rodada"])


def papel_tatico(vdg):
    """Estilo de jogo de cada atleta. Soma scouts ofensivos e defensivos dos 5 jogos anteriores e
    calcula o índice ofensivo. Depois separa os atletas de cada posição em grupos com K-means, pelo
    perfil de scouts da temporada (2, 3 ou 4 grupos, escolhidos pela silhueta, só com anos de treino).
    """
    vdg = vdg.sort_values(["ano", "atleta_id", "rodada"]).copy()
    valeu = vdg["em_campo"] & vdg["scouts_confiaveis"]
    for nome, cols in [("of", OF), ("de", DE)]:
        vdg["_tmp"] = vdg[cols].sum(axis=1).where(valeu)
        vdg[f"{nome}_m5"] = vdg.groupby(CH)["_tmp"].transform(lambda s: s.shift().rolling(5, min_periods=1).sum())
    vdg = vdg.drop(columns="_tmp")
    vdg["ind_ofensivo"] = vdg["of_m5"] / (vdg["of_m5"] + vdg["de_m5"]).replace(0, np.nan)

    cols = ["G", "A", "FT", "FD", "FF", "FS", "I", "DS_h", "FC", "CA"]
    acum = vdg[cols].fillna(0).mul(valeu.astype(float), axis=0)
    acum = acum.groupby([vdg["ano"], vdg["atleta_id"]]).cumsum()
    acum = acum.groupby([vdg["ano"], vdg["atleta_id"]]).shift()
    perfil = acum.div(acum.sum(axis=1).replace(0, np.nan), axis=0)
    ok = perfil.notna().all(axis=1) & (vdg["n_jogos_antes"] >= 3)

    vdg["cluster_papel"] = "sem_perfil"
    for pos in ["LAT", "ZAG", "MEI", "ATA"]:
        da_pos = (vdg["posicao"] == pos) & ok
        treino = da_pos & vdg["ano"].isin(ANOS_TREINO)
        linhas = vdg[treino].groupby(CH)["rodada"].idxmax()
        X = perfil.loc[linhas].values
        melhor = None
        for k in (2, 3, 4):
            km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
            nota = silhouette_score(X, km.labels_, sample_size=min(3000, len(X)), random_state=SEED)
            if melhor is None or nota > melhor[0]:
                melhor = (nota, k, km)
        nota, k, km = melhor
        grupos = km.predict(perfil.loc[da_pos].values)
        vdg.loc[da_pos, "cluster_papel"] = [f"{pos}_{g}" for g in grupos]
        print(f"  {pos}: {k} grupos (silhueta {nota:.3f})")
    return vdg


def elo_clubes(vdg, K=20.0, mando=60.0, volta=0.75):
    """Força (Elo) de cada clube rodada a rodada. Primeiro guarda o Elo antes dos jogos da rodada,
    depois atualiza com os placares: quem vence ganha pontos de acordo com a surpresa do resultado,
    com bônus para o mandante e peso maior para goleadas. Na virada de temporada o Elo volta um pouco
    para 1500, e quem subiu da Série B começa no nível dos 4 piores do ano anterior.
    """
    jog = vdg[vdg["em_campo"] & vdg["scouts_confiaveis"] & vdg["adversario_id"].notna()]
    t = jog.groupby(["ano", "rodada", "clube_id"]).agg(G=("G", "sum"), GC=("GC", "sum"),
                                                       adv=("adversario_id", "first"), mando=("mando", "first"))
    t = t.reset_index()
    contra = t[["ano", "rodada", "clube_id", "GC"]].rename(columns={"clube_id": "adv", "GC": "GC_adv"})
    t = t.merge(contra, on=["ano", "rodada", "adv"])
    t["gols"] = t["G"] + t["GC_adv"]
    gols = t.set_index(["ano", "rodada", "clube_id"])["gols"]

    elo = defaultdict(lambda: 1450.0)
    linhas = []
    clubes_ano = vdg.groupby("ano")["clube_id"].apply(set).to_dict()
    for ano in sorted(vdg["ano"].unique()):
        anteriores = clubes_ano.get(ano - 1, set())
        if anteriores:
            piores = sorted(elo[c] for c in anteriores)[:4]
        else:
            piores = [1450.0]
        nivel_subiu = float(np.mean(piores))
        for c in list(elo):
            elo[c] = 1500 + volta * (elo[c] - 1500)
        for c in clubes_ano[ano]:
            if anteriores and c not in anteriores:
                elo[c] = 1500 + volta * (nivel_subiu - 1500)

        casa = t[(t["ano"] == ano) & (t["mando"] == 1)]
        for rodada in sorted(vdg.loc[vdg["ano"] == ano, "rodada"].unique()):
            for c in vdg.loc[(vdg["ano"] == ano) & (vdg["rodada"] == rodada), "clube_id"].unique():
                linhas.append((ano, rodada, c, elo[c]))
            for _, jogo in casa[casa["rodada"] == rodada].iterrows():
                h, a = jogo["clube_id"], int(jogo["adv"])
                gh, ga = jogo["gols"], gols.get((ano, rodada, a), np.nan)
                if np.isnan(ga):
                    continue
                esperado = 1 / (1 + 10 ** (-(elo[h] + mando - elo[a]) / 400))
                if gh > ga:
                    res = 1.0
                elif gh == ga:
                    res = 0.5
                else:
                    res = 0.0
                peso = math.log(abs(gh - ga) + 1) + 1
                elo[h] += K * peso * (res - esperado)
                elo[a] -= K * peso * (res - esperado)
    return pd.DataFrame(linhas, columns=["ano", "rodada", "clube_id", "elo"])


def penalti(vdg):
    """Atributos de pênalti: média de pênaltis a favor do time e cometidos pelo adversário nos
    10 jogos anteriores, e a chance de cada atleta ser o cobrador (modelo Cobrador). A chance é
    calculada antes da rodada; só depois os pênaltis dela entram no histórico do modelo.
    """
    jog = vdg[vdg["em_campo"] & vdg["scouts_confiaveis"]]
    t = jog.groupby(["ano", "rodada", "clube_id"]).agg(PS=("PS", "sum"), PC=("PC", "sum"), PP=("PP", "sum"),
                                                       adv=("adversario_id", "first")).reset_index()
    contra = t[["ano", "rodada", "clube_id", "PC"]].rename(columns={"clube_id": "adv", "PC": "PC_adv"})
    t = t.merge(contra, on=["ano", "rodada", "adv"], how="left")
    t["batidos"] = np.maximum(t["PS"], t["PC_adv"].fillna(0))
    t = t.sort_values(["ano", "clube_id", "rodada"])
    g = t.groupby(["ano", "clube_id"])
    t["taxa_pen_time"] = g["batidos"].transform(lambda s: s.shift().rolling(10, min_periods=1).mean())
    t["taxa_pen_cometido"] = g["PC"].transform(lambda s: s.shift().rolling(10, min_periods=1).mean())

    times = {k: g for k, g in jog.groupby(["ano", "rodada", "clube_id"])}
    eventos = defaultdict(list)
    for r in t.itertuples():
        time = times.get((r.ano, r.rodada, r.clube_id))
        if time is None:
            continue
        perdidos = {a: int(p) for a, p in zip(time["atleta_id"], time["PP"]) if p > 0}
        convertidos = int(max(r.batidos - sum(perdidos.values()), 0))
        gols = {a: int(x) for a, x in zip(time["atleta_id"], time["G"]) if x > 0}
        if convertidos and not gols:
            convertidos = 0
        if convertidos or perdidos:
            eventos[(r.ano, r.rodada)].append({"campo": dict(zip(time["atleta_id"], time["posicao"])),
                                               "convertidos": convertidos, "gols": gols, "perdidos": perdidos})

    cand = vdg[vdg["status_pre"].isin(["Provável", "Dúvida"])]
    cand_rodada = defaultdict(list)
    for (a, r, c), time in cand.groupby(["ano", "rodada", "clube_id"]):
        cand_rodada[(a, r)].append(time)

    modelo = Cobrador()
    hist = []
    chance = {}
    for ano in sorted(vdg["ano"].unique()):
        for j in modelo.forca:
            prior = PRIOR.get(modelo.pos[j], 0.0)
            modelo.forca[j] = prior + 0.7 * (modelo.forca[j] - prior)
        for rodada in sorted(vdg.loc[vdg["ano"] == ano, "rodada"].unique()):
            for time in cand_rodada.get((ano, rodada), []):
                prov = time[time["status_pre"] == "Provável"]
                if len(prov):
                    campo = dict(zip(prov["atleta_id"], prov["posicao"]))
                else:
                    campo = dict(zip(time["atleta_id"], time["posicao"]))
                ch = modelo.chances(campo)
                for jid in time["atleta_id"]:
                    chance[(ano, rodada, jid)] = ch.get(jid, 0.0)
            hist += eventos.get((ano, rodada), [])
            hist = hist[-400:]
            modelo.ajustar(hist[-150:], epocas=2)

    vdg["p_cobrar_penalti"] = [chance.get(k, np.nan) for k in zip(vdg["ano"], vdg["rodada"], vdg["atleta_id"])]
    vdg = vdg.merge(t[["ano", "rodada", "clube_id", "taxa_pen_time"]], on=["ano", "rodada", "clube_id"], how="left")
    adv = t[["ano", "rodada", "clube_id", "taxa_pen_cometido"]]
    adv = adv.rename(columns={"clube_id": "adversario_id", "taxa_pen_cometido": "adv_taxa_pen_cometido"})
    vdg = vdg.merge(adv, on=["ano", "rodada", "adversario_id"], how="left")
    vdg["pen_esperado"] = vdg["p_cobrar_penalti"] * (vdg["taxa_pen_time"].fillna(0) + vdg["adv_taxa_pen_cometido"].fillna(0)) / 2
    return vdg


def regra_preco(vdg):
    """Ajusta no treino a regra de preço (variação = a·pontos + b·preço + c·média5 + d) e calcula
    quantos pontos cada atleta precisa fazer para não perder valor.
    """
    e = vdg[vdg["em_campo"] & vdg["ano"].isin(ANOS_TREINO) & vdg["media_5"].notna()]
    X = np.column_stack([e["pontos"], e["preco_pre"], e["media_5"], np.ones(len(e))])
    coef = np.linalg.lstsq(X, e["variacao"].values, rcond=None)[0]
    a, bp, c, d = coef
    print(f"  variação = {a:.3f}·pontos {bp:+.3f}·preço {c:+.3f}·média5 {d:+.3f}")
    media = vdg["media_5"].fillna(vdg["media_posicao"])
    vdg["minimo_valorizar"] = -(bp * vdg["preco_pre"] + c * media + d) / a
    return vdg


def ponderados(vdg):
    """Médias ponderadas por blocos de 5 jogos, atravessando temporadas: pontos e scouts do atleta,
    quanto ele joga e é titular, desempenho do clube e do adversário, e pontos cedidos pelo
    adversário (puxados para a média da liga quando há poucos jogos).
    """
    vdg = vdg.sort_values(["atleta_id"] + ORDEM).copy()

    jog = vdg[vdg["em_campo"]]
    cols = ["pontos", "pontos_z"] + SCOUTS
    atl = media_ponderada(jog, ["atleta_id"], cols, ORDEM, desvio=["pontos"], com_atual=True)
    novas = list(atl.columns)
    vdg = vdg.join(atl)
    vdg[novas] = vdg.groupby("atleta_id")[novas].shift()
    vdg[novas] = vdg.groupby("atleta_id")[novas].ffill()

    vdg["_em"] = vdg["em_campo"].astype(float)
    disp = media_ponderada(vdg, ["atleta_id"], ["_em", "titular_inf"], ORDEM)
    vdg["em_campo_pond"] = disp["_em_pond"]
    vdg["titular_pond"] = disp["titular_inf_pond"]
    vdg = vdg.drop(columns="_em")

    jog = vdg[vdg["em_campo"]]
    tc = jog.groupby(["ano", "rodada", "clube_id"]).agg(G=("G", "sum"), GC=("GC", "sum"), pts=("pontos", "sum"),
                                                        adv=("adversario_id", "first"),
                                                        conf=("scouts_confiaveis", "all")).reset_index()
    gs = jog[jog["posicao"] == "GOL"].groupby(["ano", "rodada", "clube_id"])["GS"].sum().rename("GS")
    tc = tc.merge(gs.reset_index(), on=["ano", "rodada", "clube_id"], how="left")
    contra = tc[["ano", "rodada", "clube_id", "GC"]].rename(columns={"clube_id": "adv", "GC": "GC_adv"})
    tc = tc.merge(contra, on=["ano", "rodada", "adv"], how="left")
    tc["gf"] = tc["G"] + tc["GC_adv"]
    tc["ga"] = tc["GS"]
    tc["sg"] = (tc["ga"] == 0).astype(float).where(tc["ga"].notna())
    tc.loc[~tc["conf"], ["gf", "ga", "sg"]] = np.nan
    tc = tc.join(media_ponderada(tc, ["clube_id"], ["gf", "ga", "sg", "pts"], ORDEM))
    clu = tc[["ano", "rodada", "clube_id", "gf_pond", "ga_pond", "sg_pond", "pts_pond"]]
    vdg = vdg.merge(clu, on=["ano", "rodada", "clube_id"], how="left")
    clu_adv = clu.rename(columns={"clube_id": "adversario_id", "gf_pond": "adv_gf_pond", "ga_pond": "adv_ga_pond",
                                  "sg_pond": "adv_sg_pond", "pts_pond": "adv_pts_pond"})
    vdg = vdg.merge(clu_adv, on=["ano", "rodada", "adversario_id"], how="left")

    jog = vdg[vdg["em_campo"] & vdg["adversario_id"].notna()]
    treino = jog[jog["ano"].isin(ANOS_TREINO)]
    media_pos = treino.groupby("posicao")["pontos"].mean()
    media_geral = treino["pontos"].mean()
    K = 10
    pos = jog.groupby(["ano", "rodada", "adversario_id", "posicao"])["pontos"].agg(soma="sum", n="size").reset_index()
    s = soma_ponderada(pos, ["adversario_id", "posicao"], ["soma", "n"], ORDEM)
    pos["cedido_pos_pond"] = (s["soma"] + K * pos["posicao"].map(media_pos)) / (s["n"] + K)
    ger = jog.groupby(["ano", "rodada", "adversario_id"])["pontos"].agg(soma="sum", n="size").reset_index()
    s = soma_ponderada(ger, ["adversario_id"], ["soma", "n"], ORDEM)
    ger["cedido_geral_pond"] = (s["soma"] + K * media_geral) / (s["n"] + K)
    vdg = vdg.merge(pos[["ano", "rodada", "adversario_id", "posicao", "cedido_pos_pond"]],
                on=["ano", "rodada", "adversario_id", "posicao"], how="left")
    vdg = vdg.merge(ger[["ano", "rodada", "adversario_id", "cedido_geral_pond"]], on=["ano", "rodada", "adversario_id"], how="left")
    return vdg


def promovidos(vdg, ate_rodada=8):
    """Marca os clubes que subiram da Série B. Até a rodada 8 apaga as estatísticas antigas desses
    clubes, para o modelo ver só a marca de promovido e o Elo inicial fraco.
    """
    clubes_ano = vdg.groupby("ano")["clube_id"].apply(set).to_dict()
    subiram = set()
    for a in clubes_ano:
        if (a - 1) in clubes_ano:
            for c in clubes_ano[a]:
                if c not in clubes_ano[a - 1]:
                    subiram.add((a, c))

    vdg["promovido"] = [float((a, c) in subiram) if (a - 1) in clubes_ano else np.nan
                      for a, c in zip(vdg["ano"], vdg["clube_id"])]
    vdg["adv_promovido"] = [float((a, c) in subiram) if (a - 1) in clubes_ano and not pd.isna(c) else np.nan
                          for a, c in zip(vdg["ano"], vdg["adversario_id"])]
    inicio = vdg["rodada"] <= ate_rodada
    vdg.loc[inicio & (vdg["promovido"] == 1), ["gf_pond", "ga_pond", "sg_pond", "pts_pond"]] = np.nan
    vdg.loc[inicio & (vdg["adv_promovido"] == 1),
          ["adv_gf_pond", "adv_ga_pond", "adv_sg_pond", "adv_pts_pond", "cedido_pos_pond", "cedido_geral_pond"]] = np.nan
    print(f"  clubes promovidos: {len(subiram)}")
    return vdg


def media_baseline(vdg):
    """Previsão do baseline: a média do atleta no Cartola antes da rodada. Na 1ª rodada (sem média)
    usa a média dele na temporada anterior; sem histórico, a média da posição na temporada anterior.
    """
    jog = vdg[vdg["em_campo"]]
    atl = jog.groupby(CH)["pontos"].mean().rename("media_ano_ant").reset_index()
    atl["ano"] += 1
    pos = jog.groupby(["ano", "posicao"])["pontos"].mean().rename("media_pos_ant").reset_index()
    pos["ano"] += 1
    vdg = vdg.merge(atl, on=CH, how="left").merge(pos, on=["ano", "posicao"], how="left")
    vdg["prev_media"] = vdg["media_cartola_pre"].fillna(vdg["media_ano_ant"]).fillna(vdg["media_pos_ant"]).fillna(0)
    return vdg


def construir():
    """Monta a tabela de atributos na ordem: histórico, média da posição, titularidade, papel tático,
    Elo, pênalti, médias ponderadas, promovidos, regra de preço e baseline. Define o alvo (pontos se
    jogou, 0 se não jogou) e os candidatos (Provável ou Dúvida, com jogo na rodada).
    """
    base = pd.read_csv(os.path.join(PROC, "base_2018_2026.csv.gz"), low_memory=False)
    print("histórico do atleta…")
    vdg = historico_atleta(base)
    vdg = media_posicao(vdg)
    print("titularidade…")
    vdg = titularidade(vdg)
    print("papel tático…")
    vdg = papel_tatico(vdg)
    print("Elo…")
    elo = elo_clubes(vdg)
    vdg = vdg.merge(elo, on=["ano", "rodada", "clube_id"], how="left")
    elo_adv = elo.rename(columns={"clube_id": "adversario_id", "elo": "adv_elo"})
    vdg = vdg.merge(elo_adv, on=["ano", "rodada", "adversario_id"], how="left")
    vdg["elo_dif"] = vdg["elo"] - vdg["adv_elo"] + 60 * (2 * vdg["mando"] - 1)
    print("pênalti…")
    vdg = penalti(vdg)
    print("médias ponderadas…")
    vdg = ponderados(vdg)
    print("promovidos…")
    vdg = promovidos(vdg)
    print("regra de preço…")
    vdg = regra_preco(vdg)
    vdg = media_baseline(vdg)

    vdg["pontos_esc"] = np.where(vdg["em_campo"], vdg["pontos"], 0.0)
    primeira = vdg.groupby("ano")["rodada"].transform("min")
    sem_status = (vdg["status_pre"] == "Desconhecido") & (vdg["rodada"] == primeira)
    vdg["candidato"] = (vdg["status_pre"].isin(["Provável", "Dúvida"]) | sem_status) & vdg["adversario_id"].notna()
    return vdg


if __name__ == "__main__":
    vdg = construir()
    vdg.to_pickle(os.path.join(PROC, "atributos_final.pkl"))
    c = vdg[vdg["candidato"]]
    print(f"linhas: {len(vdg):,} | candidatos: {len(c):,} | candidatos que jogaram: {100 * c['em_campo'].mean():.1f}%")
