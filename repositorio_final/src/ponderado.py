import numpy as np
import pandas as pd

BLOCO = 5
QUEDA = 0.6
MAX_JOGOS = 30


def pesos():
    """Peso de cada jogo anterior: os 5 mais recentes pesam 1, os 5 seguintes 0,6,
    os próximos 0,36, e assim por diante até 30 jogos.
    """
    return np.array([QUEDA ** ((k - 1) // BLOCO) for k in range(1, MAX_JOGOS + 1)])


def media_ponderada(df, chave, cols, ordem, desvio=None, com_atual=False):
    """Média ponderada dos jogos anteriores de cada grupo (ex.: de cada atleta).
    Para k = 1..30 pega o valor de k jogos atrás, multiplica pelo peso e soma; no fim divide pela
    soma dos pesos dos valores que existem. Em `desvio` calcula também o desvio ponderado.
    Com com_atual=True a própria linha conta como o jogo mais recente.
    """
    w = pesos()
    d = df.sort_values(chave + ordem)
    grupo = d.groupby(chave)[cols]
    soma = pd.DataFrame(0.0, index=d.index, columns=cols)
    soma2 = soma.copy()
    peso = soma.copy()
    ini = 0 if com_atual else 1
    for k in range(ini, MAX_JOGOS + ini):
        if k == 0:
            val = d[cols]
        else:
            val = grupo.shift(k)
        wk = w[k - ini]
        soma += val.fillna(0) * wk
        soma2 += (val.fillna(0) ** 2) * wk
        peso += val.notna() * wk

    media = soma / peso.replace(0, np.nan)
    n = d.groupby(chave).cumcount() + ini
    media[n < 1] = np.nan
    saida = media.add_suffix("_pond")
    if desvio:
        var = soma2[desvio] / peso[desvio].replace(0, np.nan) - media[desvio] ** 2
        saida = saida.join(np.sqrt(var.clip(lower=0)).add_suffix("_dp_pond"))
    return saida.reindex(df.index)


def soma_ponderada(df, chave, cols, ordem):
    """Igual à média ponderada, mas devolve só a soma ponderada dos jogos anteriores."""
    w = pesos()
    d = df.sort_values(chave + ordem)
    grupo = d.groupby(chave)[cols]
    soma = pd.DataFrame(0.0, index=d.index, columns=cols)
    for k in range(1, MAX_JOGOS + 1):
        soma += grupo.shift(k).fillna(0) * w[k - 1]
    return soma.reindex(df.index)
