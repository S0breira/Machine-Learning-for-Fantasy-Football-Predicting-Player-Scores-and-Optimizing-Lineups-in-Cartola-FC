"""Base unificada do Cartola 2018–2026: um esquema único, sem vazamentos, com adversário e mando.

Etapas:
 1. Leitura de todos os formatos (CSV 2018–2020 e 2022–2026; JSON em 2021, onde o arquivo N traz a rodada N−1).
 2. Padronização: posição/status em texto (2018–2020) viram códigos; clube_id em sigla (2018) vira número.
 3. Arquivos-fotografia (todos os pontos zerados, capturados antes dos jogos) são descartados. Se a primeira
    rodada mantida já acumula 2 jogos, seus scouts da rodada ficam como não confiáveis.
 4. Scouts acumulados → scouts da rodada (diferença entre aparições consecutivas do atleta).
 5. "Entrou em campo": campo original quando existe; senão, inferido (pontos ≠ 0, scout na rodada,
    jogos aumentou ou média mudou).
 6. Preço, status, média e jogos pré-rodada (o arquivo publica valores pós-rodada).
 7. Harmonização dos scouts entre eras e pontuação padronizada por temporada.
 8. Adversário inferido só com scouts; mando e placar oficial via football-data (validação pelo placar).
Uso: python src/base_unificada.py  → data/processed/base_2018_2026.csv.gz
"""
import json
import os

import numpy as np
import pandas as pd

AQUI = os.path.dirname(__file__)
RAW = os.path.join(AQUI, "..", "data", "raw")
SAIDA = os.path.join(AQUI, "..", "data", "processed")
ANOS = list(range(2018, 2027))

SCOUTS = ["G", "A", "FT", "FD", "FF", "FS", "PS", "I", "PP", "DS", "SG", "DE", "DP", "GS",
          "FC", "PC", "CA", "CV", "GC", "RB", "PE", "DD", "PI"]
# Scouts com o mesmo significado em todas as temporadas (DS harmonizado com RB em 2018–2019)
SCOUTS_COMUNS = ["G", "A", "FT", "FD", "FF", "FS", "I", "PP", "DS_h", "SG", "DP", "GS", "FC", "CA", "CV", "GC"]

POS_TXT = {"gol": 1, "lat": 2, "zag": 3, "mei": 4, "ata": 5, "tec": 6}
POSICOES = {1: "GOL", 2: "LAT", 3: "ZAG", 4: "MEI", 5: "ATA", 6: "TEC"}
STATUS_TXT = {"Dúvida": 2, "Suspenso": 3, "Contundido": 5, "Nulo": 6, "Provável": 7}
STATUS = {2: "Dúvida", 3: "Suspenso", 5: "Contundido", 6: "Nulo", 7: "Provável"}

CLUBES = {
    262: "Flamengo", 263: "Botafogo", 264: "Corinthians", 265: "Bahia", 266: "Fluminense",
    267: "Vasco", 275: "Palmeiras", 276: "São Paulo", 277: "Santos", 280: "Bragantino",
    282: "Atlético-MG", 283: "Cruzeiro", 284: "Grêmio", 285: "Internacional", 286: "Juventude",
    287: "Vitória", 288: "Criciúma", 290: "Goiás", 292: "Sport", 293: "Athletico-PR", 294: "Coritiba",
    314: "Avaí", 315: "Chapecoense", 327: "América-MG", 341: "CSA", 354: "Ceará", 356: "Fortaleza",
    364: "Remo", 373: "Atlético-GO", 1371: "Cuiabá", 2305: "Mirassol", 9001: "Paraná",
}
NOME_FOOTBALL_DATA = {
    262: "Flamengo RJ", 263: "Botafogo RJ", 264: "Corinthians", 265: "Bahia", 266: "Fluminense",
    267: "Vasco", 275: "Palmeiras", 276: "Sao Paulo", 277: "Santos", 280: "Bragantino",
    282: "Atletico-MG", 283: "Cruzeiro", 284: "Gremio", 285: "Internacional", 286: "Juventude",
    287: "Vitoria", 288: "Criciuma", 290: "Goias", 292: "Sport Recife", 293: "Athletico-PR", 294: "Coritiba",
    314: "Avai", 315: "Chapecoense-SC", 327: "America MG", 341: "CSA", 354: "Ceara", 356: "Fortaleza",
    364: "Remo", 373: "Atletico GO", 1371: "Cuiaba", 2305: "Mirassol", 9001: "Parana",
}
# 2018: clube_id vem como sigla (e 'ATL' é usado por dois clubes) → usa o nome completo
ID_POR_NOME_2018 = {
    "América-MG": 327, "Atlético-MG": 282, "Atlético-PR": 293, "Bahia": 265, "Botafogo": 263,
    "Ceará": 354, "Chapecoense": 315, "Corinthians": 264, "Cruzeiro": 283, "Flamengo": 262,
    "Fluminense": 266, "Grêmio": 284, "Internacional": 285, "Palmeiras": 275, "Paraná": 9001,
    "Santos": 277, "São Paulo": 276, "Sport": 292, "Vasco": 267, "Vitória": 287,
}


# ----------------------------------------------------------------------------------------- leitura
def _ler_csv(ano, arq):
    caminho = os.path.join(RAW, "cartola", str(ano), f"rodada-{arq}.csv")
    if not os.path.exists(caminho):
        return None
    d = pd.read_csv(caminho).rename(columns=lambda c: c.replace("atletas.", ""))
    if ano == 2018:
        d["clube_id"] = d["clube.id.full.name"].map(ID_POR_NOME_2018)
    # em algumas temporadas/arquivos posição e status vêm como texto ('gol', 'Provável')
    if not pd.api.types.is_numeric_dtype(d["posicao_id"]):
        d["posicao_id"] = d["posicao_id"].map(lambda v: POS_TXT.get(str(v).strip().lower(), v)).astype(float)
    if not pd.api.types.is_numeric_dtype(d["status_id"]):
        d["status_id"] = d["status_id"].map(lambda v: STATUS_TXT.get(str(v).strip(), v)).astype(float)
    d["rodada"] = arq
    return d


def _ler_json(ano, arq):
    caminho = os.path.join(RAW, "cartola", str(ano), f"Mercado_{arq}.txt")
    if not os.path.exists(caminho):
        return None
    with open(caminho, "rb") as fh:
        j = json.loads(fh.read().decode("latin-1"))
    linhas = []
    for a in j["atletas"]:
        reg = {k: a.get(k) for k in ["atleta_id", "rodada_id", "clube_id", "posicao_id", "status_id",
                                    "pontos_num", "preco_num", "variacao_num", "media_num", "jogos_num",
                                    "apelido", "nome"]}
        reg.update(a.get("scout") or {})
        linhas.append(reg)
    d = pd.DataFrame(linhas)
    d["rodada"] = d["rodada_id"]  # no JSON de 2021, o arquivo N traz a rodada N−1
    return d


def carregar_bruto():
    partes = []
    for ano in ANOS:
        for arq in range(1, 39):
            d = _ler_json(ano, arq) if ano == 2021 else _ler_csv(ano, arq)
            if d is None:
                continue
            out = pd.DataFrame({
                "ano": ano, "arquivo": arq, "rodada": d["rodada"].astype(int),
                "atleta_id": d["atleta_id"].astype(int), "apelido": d.get("apelido"), "nome": d.get("nome"),
                "clube_id": pd.to_numeric(d["clube_id"], errors="coerce"), "posicao_id": d["posicao_id"],
                "status_id": d.get("status_id"), "pontos": d["pontos_num"].astype(float),
                "preco_pos": d["preco_num"].astype(float), "variacao": d["variacao_num"].astype(float),
                "media_pos": d.get("media_num"), "jogos_pos": d.get("jogos_num", pd.Series(np.nan, index=d.index)),
                "em_campo_original": d.get("entrou_em_campo", pd.Series(np.nan, index=d.index)),
            })
            for s in SCOUTS:
                # vazio dentro de uma coluna existente = zero; coluna ausente no arquivo = desconhecido (NaN)
                out[s] = pd.to_numeric(d[s], errors="coerce").fillna(0).values if s in d else np.nan
            partes.append(out)
    bruto = pd.concat(partes, ignore_index=True)
    assert bruto["clube_id"].notna().all(), "clube sem identificador"
    assert bruto["posicao_id"].isin(POSICOES).all(), "posição não reconhecida"
    bruto["clube_id"] = bruto["clube_id"].astype(int)
    bruto["status_pre_foto"] = np.nan

    # 2021: Mercado_1 (fotografia pré-jogo) e Mercado_2 (resultado) têm a mesma rodada_id = 1. Fica o resultado,
    # e o status da fotografia vira o status pré-rodada verdadeiro.
    bruto = bruto.sort_values(["ano", "rodada", "atleta_id", "arquivo"])
    dup = bruto.duplicated(["ano", "rodada", "atleta_id"], keep="last")
    foto = bruto[dup].set_index(["ano", "rodada", "atleta_id"])["status_id"]
    chave = pd.MultiIndex.from_frame(bruto[["ano", "rodada", "atleta_id"]])
    bruto["status_pre_foto"] = np.where(~dup, foto.reindex(chave).values, np.nan)
    bruto = bruto[~dup].reset_index(drop=True)

    # 2019: inicio-0.csv é a fotografia de pré-temporada → status pré-rodada 1
    ini = os.path.join(RAW, "cartola", "2019", "inicio-0.csv")
    if os.path.exists(ini):
        d0 = pd.read_csv(ini).rename(columns=lambda c: c.replace("atletas.", ""))
        st = d0.set_index("atleta_id")["status_id"].map(lambda v: STATUS_TXT.get(str(v).strip(), v))
        m = (bruto["ano"] == 2019) & (bruto["rodada"] == 1)
        bruto.loc[m, "status_pre_foto"] = bruto.loc[m, "atleta_id"].map(st).astype(float).values
    return bruto


# ------------------------------------------------------------------------- reconstrução de rodadas
def reconstruir(df, log):
    """Arquivos com todos os pontos zerados têm dois casos:
    B) pós-rodada com pontos zerados pela API (ex.: 2023/1, com 1 jogo e scouts): pontos = média·jogos − média·jogos anteriores.
    A) fotografia ANTES dos jogos (ex.: 2022/1, 2025/1, 2020/37): é o estado inicial (preço e status pré-rodada).
       O resultado da rodada é reconstruído pelo arquivo seguinte (validado em 2024: 100% de quem jogou, 99,5% dos pontos):
         jogou   = jogos aumentou 2, ou aumentou 1 e não entrou em campo na rodada seguinte
         pontos  = média·jogos (seguinte) − média·jogos (fotografia) − pontos da rodada seguinte
         preço   = preço pré-rodada do arquivo seguinte; variação = esse preço − preço da fotografia
       Scouts acumulados só são conhecidos para quem não jogou as duas rodadas (os demais ficam ausentes)."""
    df = df.copy()
    if "status_pre_foto" not in df:
        df["status_pre_foto"] = np.nan
    df["em_campo_original"] = df["em_campo_original"].astype(object)
    nz = df.groupby(["ano", "rodada"])["pontos"].apply(lambda s: (s != 0).sum())
    jmax = df.groupby(["ano", "rodada"])["jogos_pos"].max()
    caso_a, caso_b, novos = [], [], []
    for ano, r in nz[nz == 0].index:
        cur = df[(df["ano"] == ano) & (df["rodada"] == r)]
        prox = df[(df["ano"] == ano) & (df["rodada"] == r + 1)]
        prev = df[(df["ano"] == ano) & (df["rodada"] == r - 1)].set_index("atleta_id")
        if pd.notna(jmax.get((ano, r))) and jmax[(ano, r)] >= 1:
            j_ant = cur["atleta_id"].map(prev["jogos_pos"]).fillna(0)
            m_ant = cur["atleta_id"].map(prev["media_pos"]).fillna(0)
            df.loc[cur.index, "pontos"] = (cur["media_pos"] * cur["jogos_pos"] - m_ant * j_ant).round(2)
            df.loc[cur.index, "em_campo_original"] = (cur["jogos_pos"] - j_ant) > 0
            caso_b.append((ano, r))
        elif len(prox) and prox["jogos_pos"].notna().any():
            c, p = cur.set_index("atleta_id"), prox.set_index("atleta_id")
            ids = c.index.intersection(p.index)
            c, p = c.loc[ids].copy(), p.loc[ids]
            j0, m0 = c["jogos_pos"].fillna(0), c["media_pos"].fillna(0)
            e2 = p["em_campo_original"].eq(True) if p["em_campo_original"].notna().any() else p["pontos"].ne(0)
            delta = p["jogos_pos"] - j0
            jogou = (delta >= 2) | ((delta == 1) & ~e2)
            pts = (p["media_pos"] * p["jogos_pos"] - m0 * j0 - p["pontos"].where(e2, 0)).where(jogou, 0).round(2)
            novo = c.copy()
            novo["status_pre_foto"] = c["status_id"]
            novo["pontos"], novo["em_campo_original"] = pts, jogou
            novo["jogos_pos"] = j0 + jogou
            novo["media_pos"] = (m0 * j0 + pts) / novo["jogos_pos"].replace(0, np.nan)
            novo["preco_pos"] = p["preco_pos"] - p["variacao"]
            novo["variacao"] = novo["preco_pos"] - c["preco_pos"]
            acum = pd.DataFrame(np.nan, index=ids, columns=SCOUTS)
            so_r, nao_r = jogou & ~e2, ~jogou
            acum.loc[so_r] = p.loc[so_r, SCOUTS].values
            base_ant = prev.reindex(ids)[SCOUTS].fillna(0) if len(prev) else pd.DataFrame(0.0, index=ids, columns=SCOUTS)
            acum.loc[nao_r] = base_ant.loc[nao_r].values
            novo[SCOUTS] = acum
            df = df.drop(index=cur.index)
            novos.append(novo.reset_index())
            caso_a.append((ano, r))
    if novos:
        df = pd.concat([df] + novos, ignore_index=True)
    log["rodadas reconstruídas: pontos zerados pela API (ano, rodada)"] = caso_b
    log["rodadas reconstruídas a partir de fotografia pré-jogo (ano, rodada)"] = caso_a
    return df, caso_a


# ------------------------------------------------------------------------------------- preparação
def preparar(bruto):
    log = {}
    df = bruto[bruto["posicao_id"] != 6].copy()
    log["linhas de técnicos removidas"] = int((bruto["posicao_id"] == 6).sum())

    df, reconstruidas = reconstruir(df, log)
    # o que ainda tiver todos os pontos zerados e não puder ser reconstruído é descartado
    pontos_rodada = df.groupby(["ano", "rodada"])["pontos"].apply(lambda s: (s != 0).sum())
    fotos = pontos_rodada[pontos_rodada == 0].index.tolist()
    log["arquivos-fotografia descartados (ano, rodada)"] = fotos
    df = df[~df.set_index(["ano", "rodada"]).index.isin(fotos)].copy()
    # nas rodadas reconstruídas a partir da fotografia (e na seguinte), os scouts misturam duas rodadas
    fotos = fotos + [(a, r - 1) for a, r in reconstruidas] + [(a, r) for a, r in reconstruidas]

    df = df.sort_values(["ano", "atleta_id", "rodada"]).reset_index(drop=True)
    g = df.groupby(["ano", "atleta_id"])
    ant = g[["rodada", "preco_pos", "status_id", "media_pos", "jogos_pos"]].shift()
    consecutivo = ant["rodada"].eq(df["rodada"] - 1)

    # scouts da rodada. O acumulado real de um atleta nunca diminui, mas alguns arquivos trazem valores
    # corrompidos: picos isolados (2020/10: G total 1065 contra 188 no arquivo anterior) e arquivos
    # "atrasados", menores que o anterior (2022/6). Por atleta e scout:
    #   pico     = valor maior que os dois arquivos seguintes
    #   atrasado = valor menor que o maior valor válido anterior
    # Esses valores são anulados; o acumulado segue do último valor válido e a diferença cai no arquivo seguinte.
    chave = [df["ano"], df["atleta_id"]]
    acum = df[SCOUTS].copy()
    g_ac = acum.groupby(chave)
    prox1, prox2 = g_ac.shift(-1), g_ac.shift(-2)
    pico = (acum > prox1) & ((acum > prox2) | prox2.isna())
    sem_pico = acum.mask(pico)
    max_ant = sem_pico.groupby(chave).cummax().groupby(chave).shift(1)
    atrasado = sem_pico < max_ant
    invalido = pico | atrasado
    log["valores acumulados inválidos anulados"] = int(invalido.sum().sum())
    acum = acum.mask(invalido)
    # arquivos com muitos atletas afetados: os scouts da rodada (e da seguinte) misturam rodadas
    frac = invalido.any(axis=1).groupby([df["ano"], df["rodada"]]).mean()
    arquivos_ruins = set(frac[frac > 0.05].index)
    log["arquivos com acumulado inconsistente (>5% dos atletas)"] = sorted(arquivos_ruins)
    # coluna ausente ou valor anulado: repete o último acumulado válido
    acum = acum.groupby(chave).ffill().fillna(0)
    delta = acum.groupby(chave).diff().fillna(acum)
    log["linhas com diferença negativa (zeradas)"] = int((delta < 0).any(axis=1).sum())
    df[SCOUTS] = delta.clip(lower=0)

    # rodada logo após um arquivo-fotografia: o acumulado soma duas rodadas (a da fotografia se perdeu)
    apos_foto = df.set_index(["ano", "rodada"]).index.isin(
        [(a, r + 1) for a, r in fotos] + list(arquivos_ruins) + [(a, r + 1) for a, r in arquivos_ruins])
    # rodadas cujo volume total de scouts foge muito da mediana da temporada (ex.: arquivo que soma duas rodadas)
    vol = df[df["pontos"].ne(0)].groupby(["ano", "rodada"])[["G", "FS", "FC"]].sum().sum(axis=1)
    razao = vol / vol.groupby(level=0).transform("median")
    anomalas = razao[(razao > 1.8) | (razao < 0.4)].index.tolist()
    log["rodadas com volume de scouts anômalo"] = anomalas
    df["scouts_confiaveis"] = ~(apos_foto | df.set_index(["ano", "rodada"]).index.isin(anomalas))
    log["linhas com scouts não confiáveis"] = int((~df["scouts_confiaveis"]).sum())

    # entrou em campo: original ou inferido (nas rodadas misturadas, só os pontos da própria rodada valem)
    jogos_aum = (df["jogos_pos"] - ant["jogos_pos"]).gt(0) & consecutivo
    primeiro_jogo = df["jogos_pos"].gt(0) & ant["jogos_pos"].isna()
    media_mudou = (df["media_pos"] - ant["media_pos"]).abs().gt(1e-6) & consecutivo
    sinais = (df[SCOUTS] > 0).any(axis=1) | jogos_aum | primeiro_jogo | media_mudou
    proxy = df["pontos"].ne(0) | (sinais & df["scouts_confiaveis"])
    tem_original = df["em_campo_original"].notna()
    df["em_campo"] = np.where(tem_original, df["em_campo_original"].eq(True), proxy)
    df["em_campo_inferido"] = ~tem_original
    log["acerto do em_campo inferido onde há gabarito (%)"] = round(
        100 * (proxy[tem_original] == df.loc[tem_original, "em_campo_original"].eq(True)).mean(), 2)

    # valores pré-rodada
    df["preco_pre"] = df["preco_pos"] - df["variacao"]
    df["status_pre"] = ant["status_id"].where(consecutivo).map(STATUS).fillna("Desconhecido")
    # nas rodadas reconstruídas, o status da fotografia é o status pré-rodada verdadeiro
    foto = df["status_pre_foto"].notna()
    df.loc[foto, "status_pre"] = df.loc[foto, "status_pre_foto"].map(STATUS).fillna("Desconhecido")

    # VAZAMENTO DECLARADO: nas temporadas sem status pré-rodada 1 real (sem fotografia pré-jogo), o status da
    # rodada 1 é "emprestado" da rodada 2: Provável se o atleta entrou em campo na rodada 2, Nulo caso contrário.
    # É informação futura (rodada 2); fica marcado em `status_emprestado` e é discutido como limitação.
    ini = df["rodada"] == df.groupby("ano")["rodada"].transform("min")
    rodada_fonte = 2
    jogou_r2 = df[df["rodada"] == rodada_fonte].drop_duplicates(["ano", "atleta_id"]).set_index(["ano", "atleta_id"])["em_campo"]
    log["fonte do status emprestado da rodada 1"] = f"entrou em campo na rodada {rodada_fonte}"
    chave = pd.MultiIndex.from_frame(df.loc[ini, ["ano", "atleta_id"]])
    df["status_proxy_r2"] = pd.Series(np.nan, index=df.index, dtype=object)
    df.loc[ini, "status_proxy_r2"] = np.where(jogou_r2.reindex(chave).fillna(False).astype(bool).values, "Provável", "Nulo")
    sem = ini & (df["status_pre"] == "Desconhecido")
    df["status_emprestado"] = sem
    df.loc[sem, "status_pre"] = df.loc[sem, "status_proxy_r2"]
    log[f"rodada 1 com status emprestado da rodada {rodada_fonte} (atletas por ano)"] = df.loc[sem].groupby("ano").size().to_dict()
    log["rodada 1 com status pré-rodada real (atletas por ano)"] = df.loc[ini & foto].groupby("ano").size().to_dict()
    df["media_cartola_pre"] = ant["media_pos"].where(consecutivo)
    df["jogos_cartola_pre"] = ant["jogos_pos"].where(consecutivo)

    # rótulos e harmonização
    df["posicao"] = df["posicao_id"].map(POSICOES)
    df["clube"] = df["clube_id"].map(CLUBES)
    assert df["clube"].notna().all(), "clube_id sem nome padronizado"
    df["era_scouts"] = np.select([df["ano"] <= 2019, df["ano"] == 2020], ["antiga", "transicao"], "atual")
    df["DS_h"] = df["DS"] + df["RB"]
    df.loc[~df["scouts_confiaveis"], SCOUTS + ["DS_h"]] = np.nan

    em = df[df["em_campo"]]
    est = em.groupby("ano")["pontos"].agg(["mean", "std"])
    df["pontos_z"] = (df["pontos"] - df["ano"].map(est["mean"])) / df["ano"].map(est["std"])
    return df, log


# --------------------------------------------------------------------------- adversário e mando
def totais_clube(df):
    j = df[df["em_campo"] & df["scouts_confiaveis"]]
    t = j.groupby(["ano", "rodada", "clube_id"])[["G", "GC", "FC", "FS", "PC", "PS"]].sum()
    t["GS_goleiro"] = j[j["posicao"] == "GOL"].groupby(["ano", "rodada", "clube_id"])["GS"].sum()
    return t.fillna(0).reset_index()


def inferir_adversarios(totais):
    pares = []
    for (ano, rodada), grupo in totais.groupby(["ano", "rodada"]):
        t = grupo.set_index("clube_id")
        clubes = list(t.index)
        custos = []
        for i, a in enumerate(clubes):
            for b in clubes[i + 1:]:
                A, B = t.loc[a], t.loc[b]
                gols = abs(A.G + B.GC - B.GS_goleiro) + abs(B.G + A.GC - A.GS_goleiro)
                faltas = abs(A.FC - B.FS) + abs(B.FC - A.FS)
                pen = max(B.PS - A.PC, 0) + max(A.PS - B.PC, 0)
                custos.append((100 * gols + faltas + 50 * pen, a, b))
        custos.sort()
        livres = set(clubes)
        for custo, a, b in custos:
            if a in livres and b in livres:
                pares.append({"ano": ano, "rodada": rodada, "clube_a": a, "clube_b": b, "custo": custo})
                livres -= {a, b}
    return pd.DataFrame(pares)


def associar_mando(confrontos, totais):
    fd = pd.read_csv(os.path.join(RAW, "BRA.csv"), encoding="utf-8-sig")
    fd = fd[fd["Season"].isin(ANOS)].dropna(subset=["Res"]).copy()
    id_por_nome = {n: c for c, n in NOME_FOOTBALL_DATA.items()}
    fd["casa_id"], fd["fora_id"] = fd["Home"].map(id_por_nome), fd["Away"].map(id_por_nome)
    faltando = sorted(set(fd.loc[fd["casa_id"].isna(), "Home"]) | set(fd.loc[fd["fora_id"].isna(), "Away"]))
    assert not faltando, f"times do football-data sem correspondência: {faltando}"
    fd["data"] = pd.to_datetime(fd["Date"], dayfirst=True)
    fd["par"] = [tuple(sorted(x)) for x in zip(fd["casa_id"].astype(int), fd["fora_id"].astype(int))]
    confrontos = confrontos.copy()
    confrontos["par"] = [tuple(sorted(x)) for x in zip(confrontos["clube_a"], confrontos["clube_b"])]

    jogos_por_par = {k: g.sort_values("data") for k, g in fd.groupby(["Season", "par"])}
    linhas = []
    for (ano, par), grupo in confrontos.sort_values("rodada").groupby(["ano", "par"]):
        jogos = jogos_por_par.get((ano, par))
        if jogos is None:
            continue
        for rodada, (_, jogo) in zip(grupo["rodada"], jogos.iterrows()):
            linhas.append({"ano": ano, "rodada": rodada, "casa_id": int(jogo["casa_id"]), "fora_id": int(jogo["fora_id"]),
                           "gols_casa": int(jogo["HG"]), "gols_fora": int(jogo["AG"]), "data": jogo["data"]})
    m = pd.DataFrame(linhas)
    t = totais.set_index(["ano", "rodada", "clube_id"])
    gc = lambda a, r, c, o: t.loc[(a, r, c), "G"] + t.loc[(a, r, o), "GC"]  # noqa: E731
    m["placar_confere"] = [
        (gc(a, r, c, f) == hc) and (gc(a, r, f, c) == ac)
        for a, r, c, f, hc, ac in m[["ano", "rodada", "casa_id", "fora_id", "gols_casa", "gols_fora"]].values]
    return m


def calendario(df, por_clube, ate_rodada=3):
    """Nas rodadas iniciais em que os scouts não permitem inferir o adversário (rodadas reconstruídas),
    usa o calendário: a k-ésima partida do clube na temporada (por data, no football-data) é a rodada k."""
    fd = pd.read_csv(os.path.join(RAW, "BRA.csv"), encoding="utf-8-sig")
    fd = fd[fd["Season"].isin(ANOS)].dropna(subset=["Res"]).copy()
    id_por_nome = {n: c for c, n in NOME_FOOTBALL_DATA.items()}
    fd["casa_id"], fd["fora_id"] = fd["Home"].map(id_por_nome), fd["Away"].map(id_por_nome)
    fd["data"] = pd.to_datetime(fd["Date"], dayfirst=True)
    jogaram = df[df["em_campo"] & (df["rodada"] <= ate_rodada)][["ano", "rodada", "clube_id"]].drop_duplicates()
    tem = set(zip(por_clube["ano"], por_clube["rodada"], por_clube["clube_id"]))
    falta = jogaram[[k not in tem for k in zip(jogaram["ano"], jogaram["rodada"], jogaram["clube_id"])]]
    linhas = []
    for (ano, c), g in falta.groupby(["ano", "clube_id"]):
        jogos = fd[(fd["Season"] == ano) & ((fd["casa_id"] == c) | (fd["fora_id"] == c))].sort_values("data")
        for r in g["rodada"]:
            if r - 1 < len(jogos):
                j = jogos.iloc[r - 1]
                em_casa = j["casa_id"] == c
                linhas.append({"ano": ano, "rodada": r, "clube_id": c, "mando": int(em_casa), "placar_confere": np.nan,
                               "adversario_id": int(j["fora_id"] if em_casa else j["casa_id"]), "data": j["data"]})
    return pd.DataFrame(linhas, columns=["ano", "rodada", "clube_id", "adversario_id", "mando", "placar_confere", "data"])


def preparar_tecnicos(bruto, fotos):
    """Técnicos: preço pré-rodada, pontos e variação por rodada (escalados junto com o time)."""
    t = bruto[bruto["posicao_id"] == 6].copy()
    t, _ = reconstruir(t, {})
    t = t[~t.set_index(["ano", "rodada"]).index.isin(fotos)]
    t = t.drop_duplicates(["ano", "rodada", "atleta_id"], keep="last").sort_values(["ano", "atleta_id", "rodada"])
    ant = t.groupby(["ano", "atleta_id"])[["rodada", "status_id"]].shift()
    t["preco_pre"] = t["preco_pos"] - t["variacao"]
    t["status_pre"] = ant["status_id"].where(ant["rodada"].eq(t["rodada"] - 1)).map(STATUS).fillna("Desconhecido")
    foto = t["status_pre_foto"].notna()
    t.loc[foto, "status_pre"] = t.loc[foto, "status_pre_foto"].map(STATUS).fillna("Desconhecido")
    t["clube"] = t["clube_id"].map(CLUBES)
    return t[["ano", "rodada", "atleta_id", "apelido", "clube_id", "clube", "status_pre", "preco_pre",
              "pontos", "variacao", "preco_pos"]].reset_index(drop=True)


def montar():
    bruto = carregar_bruto()
    df, log = preparar(bruto)
    fotos = log["arquivos-fotografia descartados (ano, rodada)"]
    preparar_tecnicos(bruto, fotos).to_csv(os.path.join(SAIDA, "tecnicos_2018_2026.csv.gz"), index=False)
    totais = totais_clube(df)
    confrontos = inferir_adversarios(totais)
    mando = associar_mando(confrontos, totais)
    por_clube = pd.concat([
        mando.rename(columns={"casa_id": "clube_id", "fora_id": "adversario_id"}).assign(mando=1),
        mando.rename(columns={"fora_id": "clube_id", "casa_id": "adversario_id"}).assign(mando=0),
    ])[["ano", "rodada", "clube_id", "adversario_id", "mando", "placar_confere", "data"]]
    cal = calendario(df, por_clube)
    log["clube-rodadas com adversário pelo calendário (rodadas 1–3)"] = len(cal)
    por_clube = pd.concat([por_clube, cal], ignore_index=True)
    df = df.merge(por_clube, on=["ano", "rodada", "clube_id"], how="left")
    df["adversario"] = df["adversario_id"].map(CLUBES)

    colunas = (["ano", "rodada", "data", "atleta_id", "apelido", "nome", "clube_id", "clube", "posicao",
                "adversario_id", "adversario", "mando", "placar_confere", "status_pre", "status_emprestado",
                "status_proxy_r2", "preco_pre",
                "media_cartola_pre", "jogos_cartola_pre", "em_campo", "em_campo_inferido", "pontos", "pontos_z",
                "variacao", "preco_pos", "scouts_confiaveis", "era_scouts"] + SCOUTS + ["DS_h"])
    base = df[colunas].sort_values(["ano", "rodada", "clube", "atleta_id"]).reset_index(drop=True)
    return base, log, mando


if __name__ == "__main__":
    base, log, mando = montar()
    os.makedirs(SAIDA, exist_ok=True)
    caminho = os.path.join(SAIDA, "base_2018_2026.csv.gz")
    base.to_csv(caminho, index=False)
    for k, v in log.items():
        print(f"{k}: {v}")
    em = base[base["em_campo"]]
    resumo = base.groupby("ano").agg(linhas=("atleta_id", "size"), rodadas=("rodada", "nunique"),
                                     primeira=("rodada", "min"), ultima=("rodada", "max"),
                                     em_campo=("em_campo", "sum"))
    resumo["em_campo inferido (%)"] = em.groupby("ano")["em_campo_inferido"].mean().mul(100).round(0)
    resumo["média pts"] = em.groupby("ano")["pontos"].mean().round(2)
    resumo["% em campo c/ adversário"] = em.groupby("ano")["adversario_id"].apply(lambda x: round(100 * x.notna().mean(), 1))
    resumo["% placar confere"] = mando.groupby("ano")["placar_confere"].mean().mul(100).round(1)
    pd.set_option("display.width", 200)
    print(resumo.to_string())
    print(f"\nSalvo em {os.path.normpath(caminho)}: {base.shape[0]:,} linhas x {base.shape[1]} colunas")
