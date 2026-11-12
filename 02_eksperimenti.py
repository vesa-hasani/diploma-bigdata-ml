"""
Hapi 2 – Eksperimenti.

Mat kohën e ngarkimit, të parapërpunimit, të trajnimit dhe të vlerësimit për:
  - motorin:      Spark MLlib (local[c])  dhe  scikit-learn (një makinë)
  - algoritmin:   Logistic Regression (lr)  dhe  Random Forest (rf)
  - bërthamat:    p.sh. 1, 2, 4, 8, 10 (të njëjtat për të dy motorët)
  - fraksionin:   p.sh. 0.10, 0.25, 0.50, 1.00 të bashkësisë train
Çdo konfigurim përsëritet (--reps) herë; ekzekutimi shtesë i parë shënohet si
"warm-up" dhe përjashtohet nga analiza. Koha e nisjes së Spark-ut (t_init) matet
veçmas dhe NUK përfshihet në t_total. Çdo rresht shkruhet menjëherë në CSV, që rezultatet
të mos humbin nëse ekzekutimi ndërpritet.

Përdorimi (shembull):
  python 02_eksperimenti.py --data data/higgs_parquet --out rezultatet/results.csv \
      --engines spark,sklearn --algos lr,rf --cores 1,2,4,8,10 \
      --fractions 0.1,0.25,0.5,1.0 --reps 5

Prova pilot (rreth 10–20 minuta):
  python 02_eksperimenti.py --data data/higgs_parquet --out rezultatet/pilot.csv \
      --cores 1,4 --fractions 0.1 --reps 1
"""
import argparse
import csv
import datetime as dt
import json
import os
import platform
import time

SEED = 42
FEATURES = [f"f{i}" for i in range(1, 29)]

# Hiperparametrat – të njëjtë për të dy motorët, sa është e mundur
LR_MAX_ITER = 100
RF_NUM_TREES = 20
RF_MAX_DEPTH = 8

FIELDS = ["timestamp", "engine", "engine_version", "algo", "cores", "fraction",
          "n_train", "n_test", "rep", "warmup", "t_load", "t_prep", "t_train",
          "t_eval", "t_total", "t_init", "auc", "accuracy", "f1",
          "pos_rate_train", "threads_used"]


def write_row(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def hardware_info():
    info = {"platform": platform.platform(), "python": platform.python_version(),
            "processor": platform.processor(), "logical_cpus": os.cpu_count()}
    try:
        info["ram_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    except (ValueError, OSError, AttributeError):
        info["ram_gb"] = None
    # macOS: modeli i çipit dhe ndarja e bërthamave (performance / efficiency)
    import subprocess
    for key, name in [("machdep.cpu.brand_string", "cpu"), ("hw.physicalcpu", "physical_cores"),
                      ("hw.perflevel0.physicalcpu", "performance_cores"),
                      ("hw.perflevel1.physicalcpu", "efficiency_cores")]:
        try:
            info[name] = subprocess.run(["sysctl", "-n", key], capture_output=True,
                                        text=True, timeout=5).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            info[name] = None
    return info


# --------------------------------------------------------------------- Spark
def run_spark(args, out):
    from pyspark.sql import SparkSession
    from pyspark.ml.feature import VectorAssembler, StandardScaler
    from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
    from pyspark.ml.evaluation import (BinaryClassificationEvaluator,
                                       MulticlassClassificationEvaluator)

    for cores in args.cores:
        ti = time.perf_counter()
        spark = (SparkSession.builder
                 .master(f"local[{cores}]")
                 .appName(f"higgs-bench-{cores}")
                 .config("spark.driver.memory", args.driver_memory)
                 .config("spark.sql.shuffle.partitions", str(max(2, 2 * cores)))
                 .config("spark.ui.enabled", "false")
                 .getOrCreate())
        spark.sparkContext.setLogLevel("ERROR")
        spark.range(1).count()          # detyron nisjen e plotë të JVM-së dhe të ekzekutuesve
        t_init = round(time.perf_counter() - ti, 3)
        version = spark.version
        auc_ev = BinaryClassificationEvaluator(labelCol="label",
                                               rawPredictionCol="rawPrediction",
                                               metricName="areaUnderROC")
        acc_ev = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")
        f1_ev = MulticlassClassificationEvaluator(labelCol="label", metricName="f1")

        for frac in args.fractions:
            for algo in args.algos:
                for rep in range(args.reps + 1):          # rep 0 = warm-up
                    t0 = time.perf_counter()
                    # 1) Ngarkimi (lazy evaluation -> cache() + count() e detyron ekzekutimin)
                    train = (spark.read.parquet(f"{args.data}/train")
                             .filter(f"u < {frac}").drop("u").cache())
                    test = spark.read.parquet(f"{args.data}/test").cache()
                    n_train, n_test = train.count(), test.count()
                    t1 = time.perf_counter()

                    # 2) Parapërpunimi – shkallëzuesi mësohet vetëm nga train (pa data leakage)
                    assembler = VectorAssembler(inputCols=FEATURES, outputCol="raw")
                    tr = assembler.transform(train)
                    te = assembler.transform(test)
                    if algo == "lr":
                        scaler = StandardScaler(inputCol="raw", outputCol="features",
                                                withMean=True, withStd=True).fit(tr)
                        tr, te = scaler.transform(tr), scaler.transform(te)
                    else:
                        tr = tr.withColumnRenamed("raw", "features")
                        te = te.withColumnRenamed("raw", "features")
                    tr = tr.select("features", "label").cache()
                    te = te.select("features", "label").cache()
                    tr.count(); te.count()
                    # të dhënat bruto nuk duhen më: lirohet memoria para trajnimit
                    train.unpersist(blocking=True); test.unpersist(blocking=True)
                    t2 = time.perf_counter()

                    # 3) Trajnimi
                    if algo == "lr":
                        est = LogisticRegression(maxIter=LR_MAX_ITER, regParam=0.0,
                                                 featuresCol="features", labelCol="label")
                    else:
                        est = RandomForestClassifier(numTrees=RF_NUM_TREES,
                                                     maxDepth=RF_MAX_DEPTH, seed=SEED,
                                                     featuresCol="features", labelCol="label")
                    model = est.fit(tr)
                    t3 = time.perf_counter()

                    # 4) Vlerësimi në bashkësinë test
                    pred = model.transform(te).cache()
                    auc = auc_ev.evaluate(pred)
                    acc = acc_ev.evaluate(pred)
                    f1 = f1_ev.evaluate(pred)
                    t4 = time.perf_counter()

                    # jashtë matjes së kohës: përqindja e klasës pozitive në train
                    pos_rate = tr.agg({"label": "avg"}).first()[0]
                    for d in (pred, tr, te):
                        d.unpersist(blocking=True)
                    del model, pred, tr, te, train, test
                    row = dict(timestamp=dt.datetime.now().isoformat(timespec="seconds"),
                               engine="spark", engine_version=version, algo=algo,
                               cores=cores, fraction=frac, n_train=n_train, n_test=n_test,
                               rep=rep, warmup=int(rep == 0),
                               t_load=round(t1 - t0, 3), t_prep=round(t2 - t1, 3),
                               t_train=round(t3 - t2, 3), t_eval=round(t4 - t3, 3),
                               t_total=round(t4 - t0, 3), t_init=t_init, auc=round(auc, 5),
                               accuracy=round(acc, 5), f1=round(f1, 5),
                               pos_rate_train=round(pos_rate, 5),
                               threads_used=spark.sparkContext.defaultParallelism)
                    write_row(out, row)
                    print(row, flush=True)
        spark.stop()


# ------------------------------------------------------------- scikit-learn
def run_sklearn(args, out):
    import pandas as pd
    import sklearn
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
    from threadpoolctl import threadpool_limits, threadpool_info

    for cores in args.cores:
        for frac in args.fractions:
            for algo in args.algos:
                for rep in range(args.reps + 1):
                    with threadpool_limits(limits=cores):
                        # verifikim: sa threads lejohen realisht nga bibliotekat numerike
                        blas_threads = max([p["num_threads"] for p in threadpool_info()] or [cores])
                        t0 = time.perf_counter()
                        train = pd.read_parquet(f"{args.data}/train")
                        train = train[train["u"] < frac]
                        test = pd.read_parquet(f"{args.data}/test")
                        Xtr, ytr = train[FEATURES].to_numpy(), train["label"].to_numpy()
                        Xte, yte = test[FEATURES].to_numpy(), test["label"].to_numpy()
                        del train, test
                        t1 = time.perf_counter()

                        if algo == "lr":
                            sc = StandardScaler().fit(Xtr)        # vetëm nga train
                            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
                        t2 = time.perf_counter()

                        if algo == "lr":
                            # C shumë i madh ≈ pa rregullim, si regParam=0.0 në Spark
                            est = LogisticRegression(C=1e6, max_iter=LR_MAX_ITER)
                        else:
                            est = RandomForestClassifier(n_estimators=RF_NUM_TREES,
                                                         max_depth=RF_MAX_DEPTH,
                                                         random_state=SEED, n_jobs=cores)
                        est.fit(Xtr, ytr)
                        t3 = time.perf_counter()

                        proba = est.predict_proba(Xte)[:, 1]
                        pred = (proba >= 0.5).astype(float)
                        auc = roc_auc_score(yte, proba)
                        acc = accuracy_score(yte, pred)
                        # f1 i ponderuar, për krahasim me metricName="f1" të Spark
                        f1 = f1_score(yte, pred, average="weighted")
                        t4 = time.perf_counter()

                    row = dict(timestamp=dt.datetime.now().isoformat(timespec="seconds"),
                               engine="sklearn", engine_version=sklearn.__version__,
                               algo=algo, cores=cores, fraction=frac,
                               n_train=len(ytr), n_test=len(yte), rep=rep,
                               warmup=int(rep == 0),
                               t_load=round(t1 - t0, 3), t_prep=round(t2 - t1, 3),
                               t_train=round(t3 - t2, 3), t_eval=round(t4 - t3, 3),
                               t_total=round(t4 - t0, 3), t_init="", auc=round(auc, 5),
                               accuracy=round(acc, 5), f1=round(f1, 5),
                               pos_rate_train=round(float(ytr.mean()), 5),
                               threads_used=(cores if algo == "rf" else blas_threads))
                    write_row(out, row)
                    print(row, flush=True)
                    del Xtr, Xte, ytr, yte, est


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--engines", default="spark,sklearn")
    ap.add_argument("--algos", default="lr,rf")
    ap.add_argument("--cores", default="1,2,4,8,10")
    ap.add_argument("--fractions", default="0.1,0.25,0.5,1.0")
    ap.add_argument("--reps", type=int, default=5, help="Përsëritje pa llogaritur warm-up")
    ap.add_argument("--driver-memory", default="10g")
    args = ap.parse_args()
    args.cores = [int(c) for c in args.cores.split(",")]
    args.fractions = [float(f) for f in args.fractions.split(",")]
    args.algos = args.algos.split(",")
    engines = args.engines.split(",")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    hw = hardware_info()
    hw.update(dict(seed=SEED, lr_max_iter=LR_MAX_ITER, rf_num_trees=RF_NUM_TREES,
                   rf_max_depth=RF_MAX_DEPTH, args=vars(args)))
    with open(os.path.splitext(args.out)[0] + "_mjedisi.json", "w", encoding="utf-8") as fh:
        json.dump(hw, fh, indent=2, ensure_ascii=False, default=str)

    if "spark" in engines:
        run_spark(args, args.out)
    if "sklearn" in engines:
        run_sklearn(args, args.out)


if __name__ == "__main__":
    main()
