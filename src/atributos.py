import os

import pandas as pd

AQUI = os.path.dirname(__file__)
PROC = os.path.join(AQUI, "..", "data", "processed")
SCOUTS = ["G", "A", "FT", "FD", "FF", "FS", "I", "PP", "DS_h", "SG", "DP", "GS", "FC", "CA", "CV", "GC"]
CH = ["ano", "atleta_id"]


def para_antes(df, cols):
    """Transforma uma estatística que inclui o jogo atual em uma que só usa jogos anteriores:
    pega o valor da linha anterior do atleta e repete nas rodadas em que ele não jogou.
    """
    df[cols] = df.groupby(CH)[cols].shift()
    df[cols] = df.groupby(CH)[cols].ffill()
    return df


def historico_atleta(base):
    """Histórico simples de cada atleta, só com rodadas anteriores: nº de jogos, variação média de preço
    nas 3 últimas aparições, última pontuação, média dos 5 últimos jogos e média padronizada e nº de
    jogos da temporada anterior.
    """
    base = base.sort_values(["ano", "atleta_id", "rodada"]).copy()
    g = base.groupby(CH)
    base["n_jogos_antes"] = g["em_campo"].transform(lambda s: s.astype(float).shift().fillna(0).cumsum())
    base["var_media_3"] = g["variacao"].transform(lambda s: s.shift().rolling(3, min_periods=1).mean())

    jogos = base[base["em_campo"]]
    depois = pd.DataFrame(index=jogos.index)
    depois["ultima"] = jogos["pontos"]
    depois["media_5"] = jogos.groupby(CH)["pontos"].transform(lambda s: s.rolling(5, min_periods=1).mean())
    base = base.join(depois)
    base = para_antes(base, ["ultima", "media_5"])

    ano_ant = jogos.groupby(CH).agg(media_z_temp_ant=("pontos_z", "mean"), jogos_temp_ant=("pontos", "size"))
    ano_ant = ano_ant.reset_index()
    ano_ant["ano"] += 1
    base = base.merge(ano_ant, on=CH, how="left")
    base["jogos_temp_ant"] = base["jogos_temp_ant"].fillna(0)
    return base


def media_posicao(base):
    """Média de pontos de cada posição na temporada até a rodada anterior."""
    jogos = base[base["em_campo"]]
    mp = jogos.groupby(["ano", "posicao", "rodada"])["pontos"].agg(["sum", "size"]).reset_index()
    mp = mp.sort_values(["ano", "posicao", "rodada"])
    g = mp.groupby(["ano", "posicao"])
    soma = g["sum"].transform(lambda s: s.shift().cumsum())
    qtd = g["size"].transform(lambda s: s.shift().cumsum())
    mp["media_posicao"] = soma / qtd
    return base.merge(mp[["ano", "posicao", "rodada", "media_posicao"]], on=["ano", "posicao", "rodada"], how="left")
