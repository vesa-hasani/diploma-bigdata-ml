#!/bin/bash
# Instalon gjithçka që duhet për eksperimentin në macOS (Apple Silicon).
# Përdorimi:  bash ~/diploma/diploma-bigdata-ml/setup_mac.sh
set -eo pipefail
cd "$(dirname "$0")"

echo "== 1/5 Command Line Tools (Git) =="
if ! xcode-select -p >/dev/null 2>&1; then
  xcode-select --install || true
  echo ">> Përfundo instalimin në dritaren që u hap, pastaj ekzekuto sërish këtë skript."
  exit 1
fi

echo "== 2/5 Homebrew =="
if [ ! -x /opt/homebrew/bin/brew ]; then
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi
eval "$(/opt/homebrew/bin/brew shellenv)"
grep -q 'brew shellenv' ~/.zprofile 2>/dev/null || echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile

echo "== 3/5 Java 17 dhe Python 3.12 =="
brew install openjdk@17 python@3.12
JH="$(brew --prefix openjdk@17)"
if ! grep -q 'openjdk@17' ~/.zshrc 2>/dev/null; then
  echo "export JAVA_HOME=\"$JH\"" >> ~/.zshrc
  echo 'export PATH="$JAVA_HOME/bin:$PATH"' >> ~/.zshrc
fi
export JAVA_HOME="$JH"; export PATH="$JAVA_HOME/bin:$PATH"

echo "== 4/5 Mjedisi i Python-it dhe paketat =="
[ -d ~/venv-diploma ] || "$(brew --prefix python@3.12)/bin/python3.12" -m venv ~/venv-diploma
source ~/venv-diploma/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
mkdir -p rezultatet data
pip freeze > rezultatet/versionet.txt
java -version 2> rezultatet/java_version.txt
python -c "import pyspark, sklearn; print('PySpark', pyspark.__version__, '| scikit-learn', sklearn.__version__)"

echo "== 5/5 Dataset-i HIGGS (~2.6 GB) =="
if [ ! -f data/HIGGS.csv.gz ]; then
  if curl -L --fail -o data/higgs.zip https://archive.ics.uci.edu/static/public/280/higgs.zip; then
    (cd data && unzip -o higgs.zip && rm -f higgs.zip)
  else
    echo ">> Shkarkimi automatik dështoi. Shkarko manualisht nga https://archive.ics.uci.edu/dataset/280/higgs"
    echo ">> dhe vendos HIGGS.csv.gz te: $(pwd)/data/"
  fi
fi
ls -lh data
echo "== GATI. Hapi tjetër:  bash $(pwd)/run_all.sh pilot =="
