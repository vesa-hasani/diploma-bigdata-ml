"""
Hapi 1 – Përgatitja e të dhënave (ekzekutohet vetëm një herë).

Lexon HIGGS.csv.gz (UCI, 11 milionë rreshta, 28 tipare + etiketa),
i jep emra kolonave, heq rreshtat me vlera që mungojnë dhe e ndan
dataset-in në:
  - test  (20%, i fiksuar, i njëjtë për të gjitha eksperimentet)
  - train (80%, nga i cili merren fraksionet 10/25/50/100%)
Rezultati ruhet në format Parquet, që ngarkohet shumë më shpejt se CSV.

Përdorimi:
  python 01_pergatit_te_dhenat.py --input data/HIGGS.csv.gz --output data/higgs_parquet
"""
import argparse
import json
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

SEED = 42
FEATURES = [f"f{i}" for i in range(1, 29)]
COLUMNS = ["label"] + FEATURES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Shtegu te HIGGS.csv.gz")
    ap.add_argument("--output", required=True, help="Dosja ku ruhen train/test në Parquet")
    ap.add_argument("--driver-memory", default="6g")
    args = ap.parse_args()

    spark = (SparkSession.builder
             .master("local[*]")
             .appName("higgs-prepare")
             .config("spark.driver.memory", args.driver_memory)
             .config("spark.sql.shuffle.partitions", "16")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    t0 = time.perf_counter()
    schema = ", ".join(f"{c} DOUBLE" for c in COLUMNS)
    df = spark.read.csv(args.input, header=False, schema=schema)

    n_raw = df.count()
    df = df.dropna()
    n_clean = df.count()

    train, test = df.randomSplit([0.8, 0.2], seed=SEED)
    # Kolona "u" (numër i rastësishëm 0–1, i fiksuar me seed) përdoret për të marrë
    # fraksionet: filter(u < 0.25) jep saktësisht të njëjtat rreshta në Spark dhe në
    # scikit-learn, dhe fraksionet janë të ndërthurura (10% ⊂ 25% ⊂ 50% ⊂ 100%).
    train = train.withColumn("u", F.rand(seed=SEED + 1))
    # 16 particione: mjaftueshëm për 1–8 bërthama pa krijuar skedarë të vegjël tepër
    train.repartition(16).write.mode("overwrite").parquet(f"{args.output}/train")
    test.repartition(4).write.mode("overwrite").parquet(f"{args.output}/test")

    n_train = spark.read.parquet(f"{args.output}/train").count()
    n_test = spark.read.parquet(f"{args.output}/test").count()
    label_dist = (spark.read.parquet(f"{args.output}/train")
                  .groupBy("label").count().orderBy("label").collect())

    info = {
        "rreshta_fillestare": n_raw,
        "rreshta_pas_pastrimit": n_clean,
        "rreshta_train": n_train,
        "rreshta_test": n_test,
        "shperndarja_e_etiketave_train": {str(r["label"]): r["count"] for r in label_dist},
        "koha_e_pergatitjes_s": round(time.perf_counter() - t0, 2),
        "spark_version": spark.version,
        "seed": SEED,
    }
    with open(f"{args.output}/info_dataset.json", "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2, ensure_ascii=False)
    print(json.dumps(info, indent=2, ensure_ascii=False))
    spark.stop()


if __name__ == "__main__":
    main()
