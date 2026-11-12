# Integrimi i Big Data dhe Machine Learning – eksperimenti

Kodi i pjesës praktike të punimit të diplomës **"Integrimi i Big Data dhe Machine Learning në sistemet e procesimit të të dhënave"** (UBT, Shkenca Kompjuterike dhe Inxhinieri).

Autore: Vesë Hasani · Mentor: Prof. Ass. Dr. Sc. Liridon Hoti

## Çfarë bën projekti

Krahason dy rrjedha pune për të njëjtën detyrë klasifikimi binar mbi dataset-in publik **HIGGS** (UCI, 11 milionë rreshta, 28 tipare):

| Rruga | Teknologjia | Përfaqëson |
|---|---|---|
| E shpërndarë | Apache Spark MLlib, `local[K]` | Përpunimin paralel me Spark në një makinë |
| Në një makinë | pandas + scikit-learn | Qasjen tradicionale |

Për secilin motor, algoritëm (Logistic Regression, Random Forest), numër bërthamash dhe madhësi të të dhënave maten:

- kohët e fazave: `t_load`, `t_prep`, `t_train`, `t_eval`, `t_total`, si dhe `t_init` (nisja e Spark-ut, raportohet veçmas);
- performanca parashikuese: AUC, accuracy, F1;
- kontrollet: përqindja e klasës pozitive në train dhe numri i threads të përdorura.

**Kufizim i rëndësishëm:** Spark ekzekutohet në një kompjuter të vetëm (`local[K]`). Rezultatet tregojnë paralelizimin brenda një makine dhe nuk duhen interpretuar si shkallëzim horizontal në një klaster me disa makina.

## Struktura

```
01_pergatit_te_dhenat.py   # CSV → Parquet, pastrim, ndarje 80/20, kolona u për fraksionet
02_eksperimenti.py         # matjet për të gjitha konfigurimet → rezultatet/results.csv
03_analiza.py              # mediana, variabiliteti, speedup, Spearman (H1, H2), raporti (H3), figura
requirements.txt
setup_mac.sh               # instalimi në macOS
run_all.sh                 # ekzekutimi i pilotit / eksperimentit të plotë
rezultatet/                # rezultatet e ekzekutimeve reale (ruhen në GitHub)
```

## Mjedisi i testuar

- MacBook Air, Apple M4 (10 bërthama: 4 performance + 6 efficiency), 16 GB RAM
- macOS, Python 3.12, OpenJDK 17
- Versionet e sakta të paketave: `rezultatet/versionet.txt`

## Instalimi (macOS)

```bash
xcode-select --install
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
# ekzekuto rreshtat që Homebrew shfaq te "Next steps", pastaj rihap Terminal-in
brew install openjdk@17 python@3.12
echo 'export JAVA_HOME="$(brew --prefix openjdk@17)"' >> ~/.zshrc
echo 'export PATH="$JAVA_HOME/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

cd ~/diploma/diploma-bigdata-ml
python3.12 -m venv ~/venv-diploma
source ~/venv-diploma/bin/activate
pip install -r requirements.txt
mkdir -p rezultatet && pip freeze > rezultatet/versionet.txt
```

## Të dhënat

Shkarko HIGGS nga https://archive.ics.uci.edu/dataset/280/higgs dhe vendos `HIGGS.csv.gz` te `data/`.
Citimi: Whiteson, D. (2014). *HIGGS* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5V312

## Ekzekutimi

Në macOS, gjithçka mund të bëhet me dy skripte:

```bash
bash setup_mac.sh          # instalimi i mjedisit dhe shkarkimi i dataset-it
bash run_all.sh pilot      # provë e shkurtër
bash run_all.sh full       # eksperimenti i plotë: çdo motor × bërthama si proces i veçantë, pastaj analiza
```

Ose hap pas hapi:

```bash
# 1) Përgatitja (vetëm një herë)
python 01_pergatit_te_dhenat.py --input data/HIGGS.csv.gz --output data/higgs_parquet

# 2) Prova pilot
caffeinate -i python 02_eksperimenti.py --data data/higgs_parquet --out rezultatet/pilot.csv \
    --cores 1,4 --fractions 0.1 --reps 1

# 3) Eksperimenti i plotë (orë të tëra; Mac në karikues, programet e tjera të mbyllura)
caffeinate -i python 02_eksperimenti.py --data data/higgs_parquet --out rezultatet/results.csv \
    --cores 1,2,4,8,10 --fractions 0.1,0.25,0.5,1.0 --reps 5

# 4) Analiza
python 03_analiza.py --results rezultatet/results.csv --out rezultatet/analiza
```

## Vendimet metodologjike në kod

- **Të njëjtat të dhëna për të dy motorët:** fraksionet merren me `u < p` nga e njëjta kolonë e rastësishme (seed 43), prandaj janë identike dhe të ndërthurura (10% ⊂ 25% ⊂ 50% ⊂ 100%). Bashkësia e testimit (20%, seed 42) është e njëjtë për të gjitha konfigurimet.
- **Pa data leakage:** `StandardScaler` mësohet vetëm nga train dhe përdoret vetëm për Logistic Regression.
- **Të njëjtat bërthama:** Spark `local[K]`; scikit-learn me `threadpoolctl` (K threads) dhe `n_jobs=K` për Random Forest.
- **Fazat e ndara:** Spark përdor *lazy evaluation*, prandaj pas çdo faze bëhet `cache()` + `count()`. Nisja e Spark-ut (`t_init`) matet veçmas.
- **Përsëritjet:** për çdo konfigurim kryhet 1 ekzekutim *warm-up* (përjashtohet) + `--reps` matje; raportohen mediana, devijimi standard dhe koeficienti i variacionit.
- **Hiperparametrat:** LR `maxIter=100`, pa rregullim (Spark `regParam=0`, scikit-learn `C=1e6`); RF 20 pemë, thellësia 8, seed 42. Implementimet nuk janë identike (p.sh. MLlib diskretizon tiparet në *bins*), prandaj krahasohet edhe AUC.

## Deklarim

Kodi fillestar është zhvilluar me ndihmën e një asistenti AI. Ekzekutimi i eksperimenteve, kontrolli, përshtatjet dhe interpretimi i rezultateve janë kryer nga autorja.
