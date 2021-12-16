# Welcome to dmLib: the design margins library

## Installation for development

**MacOS/Linux**

Create a virtual environment to develop this app

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install ``pandoc``, needed to build documentation, see [https://pandoc.org/installing.html](https://pandoc.org/installing.html) 

Use ``flit`` to install this package locally

```
flit build
```

On windows use.
```
flit install --deps=develop --pth-file
```
On MacOS/Linux use
```
flit install --deps=develop --symlink
```

Use ``sphinx`` to build the documentation

```
sphinx-autobuild docs/source docs/build/html
```

Navigate to <http://127.0.0.1:8000> to view the documentation.

Alternatively, use the supplied bash script to execute all of the above [setup_dev.sh](setup_dev.sh) using the command 

```
source setup_dev.sh
```