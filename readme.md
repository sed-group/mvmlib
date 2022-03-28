# Welcome to mvmlib: the margin value method library

## Quickstart guide

Create a virtual environment for your project

**MacOS/Linux**

```
python -m venv .venv
source .venv/bin/activate
```

**Windows**
```
python -m venv .env
.env\Scripts\activate
```
### Prerequisites

``mvmlib`` requires ``numpy`` make sure you have it installed

### Installing
To install the latest released version of ``mvmlib``
```
pip install mvmlib
```

### Uninstalling
to uninstall
```
pip uninstall mvmlib
```

## Installation for development

First clone this repository and in the ```mvmlib`` directory do the following:

**MacOS/Linux**

Create a virtual environment to develop this library

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install ``pandoc``, needed to build documentation, see [https://pandoc.org/installing.html](https://pandoc.org/installing.html) 
Use ``flit`` to install this package locally

```
flit build
flit install --deps=develop --symlink
```

Use ``sphinx`` to build the documentation

```
sphinx-autobuild docs docs/build/html
```

Navigate to <http://127.0.0.1:8000> to view the documentation.

Alternatively, use the supplied bash script to execute all of the above [setup_dev.sh](setup_dev.sh) using the command 

```
source setup_dev.sh
```


**Windows**
```
python -m venv .env
.env\Scripts\activate
pip install -r requirements.txt
```

Install ``pandoc``, needed to build documentation, see [https://pandoc.org/installing.html](https://pandoc.org/installing.html) 
Use ``flit`` to install this package locally

```
flit build
flit install --deps=develop --pth-file
```

Use ``sphinx`` to build the documentation

```
sphinx-autobuild docs docs/build/html
```

Navigate to <http://127.0.0.1:8000> to view the documentation.