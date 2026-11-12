"""
Hapi 3 – Analiza e rezultateve.

Lexon results.csv, përjashton ekzekutimet warm-up dhe prodhon:
  tabela/ permbledhje.csv      – mediana dhe devijimi standard për çdo konfigurim
  tabela/ speedup.csv          – përshpejtimi S(c) = T(1) / T(c) dhe efikasiteti E(c) = S(c) / c
  tabela/ hipotezat.csv        – testi Spearman për H1 (bërthamat) dhe H2 (madhësia)
  tabela/ krahasimi_motoreve.csv – Spark kundrejt scikit-learn për çdo fraksion
  tabela/ hipoteza_H3.csv      – raporti i kohëve dhe pika e kalimit për H3
  tabela/ spark_t_init.csv     – koha e nisjes së Spark-ut për çdo numër bërthamash
  figura/ *.png                – grafikët për Kapitullin 5

Përdorimi:
  python 03_analiza.py --results rezultatet/results.csv --out rezultatet/analiza
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

ALPHA = 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tab, fig = os.path.join(args.out, "tabela"), os.path.join(args.out, "figura")
    os.makedirs(tab, exist_ok=True)
    os.makedirs(fig, exist_ok=True)

    df = pd.read_csv(args.results)
    df = df[df["warmup"] == 0].copy()
    keys = ["engine", "algo", "cores", "fraction"]
    metrics = ["t_load", "t_prep", "t_train", "t_eval", "t_total", "auc", "accuracy", "f1"]

    # 1) Përmbledhja
    summ = df.groupby(keys).agg(
        n_train=("n_train", "first"), runs=("rep", "count"),
        **{f"{m}_med": (m, "median") for m in metrics},
        **{f"{m}_std": (m, "std") for m in ["t_train", "t_total", "auc"]},
        pos_rate_train=("pos_rate_train", "mean")).reset_index()
    # variabiliteti i matjeve: koeficienti i variacionit (std / mesatare) në %
    means = df.groupby(keys)[["t_train", "t_total"]].mean().reset_index()
    summ = summ.merge(means.rename(columns={"t_train": "t_train_mean", "t_total": "t_total_mean"}), on=keys)
    summ["cv_t_train_pct"] = 100 * summ["t_train_std"] / summ["t_train_mean"]
    summ["cv_t_total_pct"] = 100 * summ["t_total_std"] / summ["t_total_mean"]
    summ.round(4).to_csv(os.path.join(tab, "permbledhje.csv"), index=False)

    # 2) Speedup dhe efikasiteti (kundrejt 1 bërthame, për të njëjtin motor/algoritëm/fraksion)
    base = summ[summ["cores"] == 1][["engine", "algo", "fraction", "t_train_med", "t_total_med"]]
    sp = summ.merge(base, on=["engine", "algo", "fraction"], suffixes=("", "_c1"))
    sp["speedup_train"] = sp["t_train_med_c1"] / sp["t_train_med"]
    sp["speedup_total"] = sp["t_total_med_c1"] / sp["t_total_med"]
    sp["efikasiteti_train"] = sp["speedup_train"] / sp["cores"]
    sp[keys + ["t_train_med", "speedup_train", "efikasiteti_train", "speedup_total"]] \
        .round(3).to_csv(os.path.join(tab, "speedup.csv"), index=False)

    # 3) Hipotezat (Spearman, mbi të gjitha ekzekutimet jo-warm-up)
    rows = []
    for (eng, algo), g in df.groupby(["engine", "algo"]):
        # H1: më shumë bërthama -> më pak kohë (pritet rho < 0), për çdo fraksion
        for frac, gf in g.groupby("fraction"):
            if gf["cores"].nunique() > 1:
                rho, p = spearmanr(gf["cores"], gf["t_total"])
                rows.append(dict(hipoteza="H1", engine=eng, algo=algo, kontrollohet=f"fraction={frac}",
                                 n=len(gf), rho=rho, p=p, mbeshtetet=bool(rho < 0 and p < ALPHA)))
        # H2: më shumë rreshta -> më shumë kohë trajnimi (pritet rho > 0), për çdo nr. bërthamash
        for cores, gc in g.groupby("cores"):
            if gc["n_train"].nunique() > 1:
                rho, p = spearmanr(gc["n_train"], gc["t_train"])
                rows.append(dict(hipoteza="H2", engine=eng, algo=algo, kontrollohet=f"cores={cores}",
                                 n=len(gc), rho=rho, p=p, mbeshtetet=bool(rho > 0 and p < ALPHA)))
    pd.DataFrame(rows).round(4).to_csv(os.path.join(tab, "hipotezat.csv"), index=False)

    # 4) Spark kundrejt scikit-learn (me numrin maksimal të bërthamave)
    if df["engine"].nunique() > 1:
        cmax = int(summ["cores"].max())
        s = summ[summ["cores"] == cmax].pivot_table(index=["algo", "fraction", "n_train"],
                                                    columns="engine",
                                                    values=["t_total_med", "auc_med"])
        s.columns = [f"{a}_{b}" for a, b in s.columns]
        if {"t_total_med_spark", "t_total_med_sklearn"} <= set(s.columns):
            s["raporti_koha_spark_mbi_sklearn"] = s["t_total_med_spark"] / s["t_total_med_sklearn"]
        s = s.round(4).reset_index()
        s.to_csv(os.path.join(tab, "krahasimi_motoreve.csv"), index=False)
        # H3: pika e kalimit = fraksioni më i vogël ku Spark nuk është më i ngadaltë (raporti <= 1)
        if "raporti_koha_spark_mbi_sklearn" in s.columns:
            h3 = []
            for algo, g in s.sort_values("fraction").groupby("algo"):
                r = g["raporti_koha_spark_mbi_sklearn"].tolist()
                cross = g[g["raporti_koha_spark_mbi_sklearn"] <= 1]
                h3.append(dict(hipoteza="H3", algo=algo, cores=cmax,
                               raporti_sipas_fraksionit=" | ".join(f"{f:g}: {x:.2f}" for f, x in zip(g["fraction"], r)),
                               raporti_zvogelohet=all(b <= a for a, b in zip(r, r[1:])),
                               raporti_mbi_1_te_fraksioni_me_i_vogel=r[0] > 1,
                               pika_e_kalimit=(f"fraction={cross['fraction'].iloc[0]:g}, n_train={int(cross['n_train'].iloc[0])}"
                                               if len(cross) else "nuk u arrit brenda fraksioneve të testuara")))
            pd.DataFrame(h3).to_csv(os.path.join(tab, "hipoteza_H3.csv"), index=False)

    # Koha e nisjes së Spark-ut (raportohet veçmas, nuk përfshihet në t_total)
    if "t_init" in df.columns and (df["engine"] == "spark").any():
        (df[df["engine"] == "spark"].groupby("cores")["t_init"].first()
         .to_csv(os.path.join(tab, "spark_t_init.csv")))

    # ------------------------------------------------------------ Figurat
    for (eng, algo), g in summ.groupby(["engine", "algo"]):
        # Figura A: koha e trajnimit sipas bërthamave, një vijë për çdo fraksion
        plt.figure(figsize=(6, 4))
        for frac, gf in g.groupby("fraction"):
            gf = gf.sort_values("cores")
            plt.errorbar(gf["cores"], gf["t_train_med"], yerr=gf["t_train_std"],
                         marker="o", capsize=3, label=f"{int(frac * 100)}%")
        plt.xlabel("Numri i bërthamave")
        plt.ylabel("Koha e trajnimit (s), mediana")
        plt.xticks(sorted(g["cores"].unique()))
        plt.legend(title="Fraksioni")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(fig, f"koha_sipas_berthamave_{eng}_{algo}.png"), dpi=200)
        plt.close()

        # Figura B: koha e trajnimit sipas madhësisë, një vijë për çdo numër bërthamash
        plt.figure(figsize=(6, 4))
        for cores, gc in g.groupby("cores"):
            gc = gc.sort_values("n_train")
            plt.plot(gc["n_train"] / 1e6, gc["t_train_med"], marker="o", label=f"{cores}")
        plt.xlabel("Rreshta trajnimi (milionë)")
        plt.ylabel("Koha e trajnimit (s), mediana")
        plt.legend(title="Bërthama")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(fig, f"koha_sipas_madhesise_{eng}_{algo}.png"), dpi=200)
        plt.close()

    # Figura C: speedup kundrejt idealit
    plt.figure(figsize=(6, 4))
    fmax = sp["fraction"].max()
    for (eng, algo), g in sp[sp["fraction"] == fmax].groupby(["engine", "algo"]):
        g = g.sort_values("cores")
        plt.plot(g["cores"], g["speedup_train"], marker="o", label=f"{eng} – {algo}")
    cs = sorted(sp["cores"].unique())
    plt.plot(cs, cs, "k--", label="Ideal (linear)")
    plt.xlabel("Numri i bërthamave")
    plt.ylabel("Speedup i trajnimit")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig, "speedup.png"), dpi=200)
    plt.close()

    print("Analiza përfundoi:", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
