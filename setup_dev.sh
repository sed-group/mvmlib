# Setup python virtual environment

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Flit local installation in virtual environment

flit build
flit install --deps=develop

# build the documentation

sphinx-autobuild docs/source docs/build/html