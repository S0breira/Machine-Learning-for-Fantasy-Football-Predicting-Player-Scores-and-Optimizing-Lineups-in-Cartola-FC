import os

import pandas as pd
import requests

AQUI = os.path.dirname(__file__)
RAW = os.path.join(AQUI, "..", "data", "raw")
COMMIT = "639fd0569f826f998449b9c94c05cd2691f088bc"
URL_CARTOLA = f"https://raw.githubusercontent.com/henriquepgomide/caRtola/{COMMIT}/data/01_raw"
URL_JOGOS = "https://www.football-data.co.uk/new/BRA.csv"
DATA_CORTE = "2026-11-14"


def arquivos():
    """Lista os arquivos de cada temporada no repositório caRtola: um por rodada, mais as fotografias
    de pré-temporada de 2019 (inicio-0) e 2022 (rodada-0). Em 2021 os arquivos são JSON (Mercado_N.txt).
    """
    lista = []
    for ano in range(2018, 2027):
        if ano == 2021:
            nomes = [f"Mercado_{r}.txt" for r in range(1, 39)]
        else:
            nomes = [f"rodada-{r}.csv" for r in range(1, 39)]
        if ano == 2019:
            nomes.append("inicio-0.csv")
        if ano == 2022:
            nomes.append("rodada-0.csv")
        lista += [(ano, n) for n in nomes]
    return lista


def baixar():
    """Baixa os dados brutos do Cartola (commit fixo do caRtola) e o BRA.csv do football-data para
    data/raw. Arquivos que já existem não são baixados de novo; rodadas que não existem (2026 em
    andamento) são ignoradas. O BRA.csv é atualizado pelo site a cada rodada, então só ficam os jogos
    até DATA_CORTE (data dos dados usados no trabalho), para os resultados serem sempre os mesmos.
    """
    for ano, nome in arquivos():
        destino = os.path.join(RAW, "cartola", str(ano), nome)
        if os.path.exists(destino):
            continue
        r = requests.get(f"{URL_CARTOLA}/{ano}/{nome}", timeout=60)
        if r.status_code == 404:
            continue
        r.raise_for_status()
        os.makedirs(os.path.dirname(destino), exist_ok=True)
        with open(destino, "wb") as f:
            f.write(r.content)
    destino = os.path.join(RAW, "BRA.csv")
    if not os.path.exists(destino):
        jogos = pd.read_csv(URL_JOGOS, encoding="utf-8-sig")
        data = pd.to_datetime(jogos["Date"], dayfirst=True)
        jogos = jogos[data <= DATA_CORTE]
        jogos.to_csv(destino, index=False, encoding="utf-8-sig")
    print(f"dados em {os.path.normpath(RAW)}")


if __name__ == "__main__":
    baixar()
