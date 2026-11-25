import os
import time

import atributos_final
import baixar_dados
import base_unificada
import modelo_final
import otimizador


def main():
    """Baixa os dados (se ainda não estiverem em data/raw) e roda as 4 etapas em ordem: base tratada,
    atributos, modelo de pontos e simulação das temporadas.
    """
    inicio = time.time()
    baixar_dados.baixar()
    os.makedirs(base_unificada.SAIDA, exist_ok=True)

    print("== 1/4 base tratada ==")
    base, log, _ = base_unificada.montar()
    base.to_csv(os.path.join(base_unificada.SAIDA, "base_2018_2026.csv.gz"), index=False)
    for k, v in log.items():
        print(f"  {k}: {v}")

    print("== 2/4 atributos ==")
    atributos_final.construir().to_pickle(os.path.join(atributos_final.PROC, "atributos_final.pkl"))

    print("== 3/4 modelo de pontos ==")
    modelo_final.main()

    print("== 4/4 simulação (modelo x baseline) ==")
    otimizador.main()
    print(f"\nconcluído em {(time.time() - inicio) / 60:.1f} min")


if __name__ == "__main__":
    main()
