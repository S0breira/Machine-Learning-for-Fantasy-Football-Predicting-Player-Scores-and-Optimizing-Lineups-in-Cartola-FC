import os
import warnings

import lightgbm as lgb
import pandas as pd

from atributos import PROC, SCOUTS

warnings.filterwarnings("ignore")
SEED = 42
RODADA_MIN = 6

NUM = (["preco_pre", "var_media_3", "minimo_valorizar",
        "pontos_pond", "pontos_dp_pond", "pontos_z_pond", "ultima", "n_jogos_antes", "media_cartola_pre",
        "media_z_temp_ant", "jogos_temp_ant",
        "em_campo_pond", "titular_pond",
        "mando", "rodada",
        "gf_pond", "ga_pond", "sg_pond", "pts_pond", "adv_gf_pond", "adv_ga_pond", "adv_sg_pond", "adv_pts_pond",
        "elo", "adv_elo", "elo_dif", "promovido", "adv_promovido",
        "cedido_pos_pond", "cedido_geral_pond",
        "of_m5", "de_m5", "ind_ofensivo",
        "p_cobrar_penalti", "taxa_pen_time", "adv_taxa_pen_cometido", "pen_esperado"]
       + [f"{s}_pond" for s in SCOUTS])
CAT = ["posicao", "status_pre", "era_scouts", "cluster_papel"]
COLS = NUM + CAT
GRUPOS = {
    "titularidade": ["titular_pond"],
    "papel tático": ["of_m5", "de_m5", "ind_ofensivo", "cluster_papel"],
    "Elo": ["elo", "adv_elo", "elo_dif"],
    "pontos cedidos": ["cedido_pos_pond", "cedido_geral_pond"],
    "pênalti": ["p_cobrar_penalti", "taxa_pen_time", "adv_taxa_pen_cometido", "pen_esperado"],
}


def X(d, cols):
    """Pega as colunas de atributos e marca as categóricas como category (o LightGBM usa direto)."""
    x = d[cols].copy()
    for c in cols:
        if c in CAT:
            x[c] = x[c].astype("category")
    return x


def avaliar(d, col):
    """Avalia uma previsão rodada a rodada (a partir da rodada 6): Spearman entre previsto e real,
    Top-11 (pontos reais médios dos 11 com maior previsão) e erro absoluto médio.
    """
    d = d[d["rodada"] >= RODADA_MIN]
    sp, top = [], []
    for _, rod in d.groupby(["ano", "rodada"]):
        sp.append(rod[col].rank().corr(rod["pontos_esc"].rank()))
        top.append(rod.nlargest(11, col)["pontos_esc"].mean())
    erro = (d[col] - d["pontos_esc"]).abs().mean()
    return {"Spearman": pd.Series(sp).mean(), "Top-11": pd.Series(top).mean(), "MAE": erro}


def treinar(d, params, cols, queda, arvores, val=None):
    """Treina o LightGBM. Com `queda`, anos antigos pesam menos (queda^anos de distância).
    Com dados de validação, para quando 100 árvores seguidas não melhoram.
    """
    m = lgb.LGBMRegressor(**params, n_estimators=arvores)
    peso = None
    if queda:
        peso = queda ** (d["ano"].max() - d["ano"])
    extra = {}
    if val is not None:
        extra = dict(eval_set=[(X(val, cols), val["pontos_esc"])], callbacks=[lgb.early_stopping(100, verbose=False)])
    m.fit(X(d, cols), d["pontos_esc"], sample_weight=peso, **extra)
    return m


def main():
    """Treina o modelo de pontos:
    1. testa combinações de hiperparâmetros e fica com a de melhor Top-11 na validação;
    2. ablação: treina de novo sem cada grupo de atributos;
    3. faz as previsões fora da amostra (cada período previsto por um modelo que só viu dados anteriores);
    4. compara o modelo com o baseline (média do Cartola) e mostra a importância dos atributos.
    """
    b = pd.read_pickle(os.path.join(PROC, "atributos_final.pkl"))
    for c in CAT:
        b[c] = b[c].astype(str)
    c = b[b["candidato"]].copy()
    tr = c[c["ano"] <= 2024]
    va = c[(c["ano"] == 2025) & (c["rodada"] <= 19)]
    print(f"candidatos: treino {len(tr):,} | validação {len(va):,}")

    melhor = None
    for folhas in [15, 31, 63]:
        for minimo in [20, 50, 100]:
            for queda in [None, 0.85]:
                params = dict(objective="regression", learning_rate=0.03, num_leaves=folhas, min_child_samples=minimo,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1)
                m = treinar(tr, params, COLS, queda, 3000, val=va)
                r = avaliar(va.assign(p=m.predict(X(va, COLS))), "p")
                print(f"folhas={folhas:<3} min={minimo:<4} queda={str(queda):<5} árvores={m.best_iteration_:<5} "
                      f"Top-11={r['Top-11']:.3f} Spearman={r['Spearman']:.4f}")
                nota = (r["Top-11"], r["Spearman"])
                if melhor is None or nota > melhor[0]:
                    melhor = (nota, params, queda, m.best_iteration_)
    _, params, queda, arvores = melhor
    print(f"\nescolhido: folhas={params['num_leaves']} min={params['min_child_samples']} queda={queda} árvores={arvores}")

    print("\nablação (validação, diferença sem o grupo):")
    m = treinar(tr, params, COLS, queda, arvores)
    ref = avaliar(va.assign(p=m.predict(X(va, COLS))), "p")
    print(f"  completo: Top-11={ref['Top-11']:.3f} Spearman={ref['Spearman']:.4f}")
    for nome, grupo in GRUPOS.items():
        cols = [x for x in COLS if x not in grupo]
        m = treinar(tr, params, cols, queda, arvores)
        r = avaliar(va.assign(p=m.predict(X(va, cols))), "p")
        print(f"  sem {nome:<15} Top-11 {r['Top-11'] - ref['Top-11']:+.3f}  Spearman {r['Spearman'] - ref['Spearman']:+.4f}")

    periodos = [
        ("2024", c["ano"] <= 2023, c["ano"] == 2024),
        ("2025 r1-19 (validação)", c["ano"] <= 2024, (c["ano"] == 2025) & (c["rodada"] <= 19)),
        ("2025 r20-38 (teste)", (c["ano"] <= 2024) | ((c["ano"] == 2025) & (c["rodada"] <= 19)),
         (c["ano"] == 2025) & (c["rodada"] >= 20)),
        ("2026 (holdout)", c["ano"] <= 2025, c["ano"] == 2026),
    ]
    partes = []
    print("\nmodelo x baseline (média do Cartola):")
    for nome, treina, preve in periodos:
        m = treinar(c[treina], params, COLS, queda, int(arvores * 1.1))
        d = c[preve].copy()
        d["pred"] = m.predict(X(d, COLS))
        partes.append(d[["ano", "rodada", "atleta_id", "pred"]])
        rm, rb = avaliar(d, "pred"), avaliar(d, "prev_media")
        print(f"  {nome:<24} modelo: Top-11={rm['Top-11']:.2f} Spearman={rm['Spearman']:.3f} MAE={rm['MAE']:.2f} | "
              f"baseline: Top-11={rb['Top-11']:.2f} Spearman={rb['Spearman']:.3f} MAE={rb['MAE']:.2f}")
    pd.concat(partes, ignore_index=True).to_pickle(os.path.join(PROC, "previsoes_final.pkl"))

    imp = pd.Series(m.booster_.feature_importance("gain"), index=COLS).sort_values(ascending=False)
    print("\nimportância (% do ganho), 20 maiores:")
    print((100 * imp / imp.sum()).head(20).round(1).to_string())


if __name__ == "__main__":
    main()
