#!/bin/bash
# Ekzekuton eksperimentin. Përdorimi:
#   bash ~/diploma/diploma-bigdata-ml/run_all.sh pilot   # provë e shkurtër (rreth 10–20 min)
#   bash ~/diploma/diploma-bigdata-ml/run_all.sh full    # eksperimenti i plotë + analiza (orë të tëra)
# Mbaje Mac-in në karikues dhe mbyll programet e tjera gjatë matjeve.
set -eo pipefail
cd "$(dirname "$0")"
eval "$(/opt/homebrew/bin/brew shellenv)"
export JAVA_HOME="$(brew --prefix openjdk@17)"; export PATH="$JAVA_HOME/bin:$PATH"
source ~/venv-diploma/bin/activate
mkdir -p rezultatet

if [ ! -d data/higgs_parquet/train ]; then
  echo "== Përgatitja e të dhënave (vetëm herën e parë) =="
  python 01_pergatit_te_dhenat.py --input data/HIGGS.csv.gz --output data/higgs_parquet 2>&1 | tee rezultatet/log_pergatitja.txt
  cp data/higgs_parquet/info_dataset.json rezultatet/
fi

case "${1:-pilot}" in
  pilot)
    caffeinate -i python 02_eksperimenti.py --data data/higgs_parquet --out rezultatet/pilot.csv \
      --cores 1,4 --fractions 0.1 --reps 1 2>&1 | tee rezultatet/log_pilot.txt ;;
  full)
    if [ -f rezultatet/results.csv ]; then
      echo ">> rezultatet/results.csv ekziston tashmë. Riemëroje ose më pyet para se ta nisësh sërish."; exit 1
    fi
    # Çdo motor dhe çdo numër bërthamash ekzekutohet si proces i veçantë:
    # nisje e re e Spark-ut (t_init i krahasueshëm) dhe memorie e ndarë mes motorëve.
    for ENGINE in spark sklearn; do
      for C in 1 2 4 8 10; do
        echo "== $ENGINE, $C bërthama ==" | tee -a rezultatet/log_full.txt
        caffeinate -i python 02_eksperimenti.py --data data/higgs_parquet --out rezultatet/results.csv \
          --engines $ENGINE --cores $C --fractions 0.1,0.25,0.5,1.0 --reps 5 --driver-memory 10g 2>&1 | tee -a rezultatet/log_full.txt
        cp rezultatet/results_mjedisi.json "rezultatet/mjedisi_${ENGINE}_${C}.json"
      done
    done
    python 03_analiza.py --results rezultatet/results.csv --out rezultatet/analiza 2>&1 | tee rezultatet/log_analiza.txt ;;
  *) echo "Përdorimi: bash run_all.sh pilot|full"; exit 1 ;;
esac
echo "== Përfundoi. Rezultatet janë te: $(pwd)/rezultatet =="
