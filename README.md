<div align="center">
<h1>Probability & Statistics Documentation</a></h1>
by Hongnan Gao
Sep, 2022
<br>
</div>


<h4 align="center">
  <a href="https://gao-hongnan.github.io/gaohn-probability-stats/intro.html">Documentation</a>
</h4>

This is the documentation for the course [Introduction to Probability for Data Science](https://probability4datascience.com/).

## Workflow

### Installation

```bash
~/gaohn        $ git clone https://github.com/gao-hongnan/gaohn-probability-stats.git gaohn_probability_stats
~/gaohn        $ cd gaohn-probability-stats
~/gaohn        $ python -m venv <venv_name> && <venv_name>\Scripts\activate 
~/gaohn (venv) $ python -m pip install --upgrade pip setuptools wheel
~/gaohn (venv) $ pip install -r requirements.txt
~/gaohn (venv) $ pip install myst-nb==0.16.0 
```

The reason for manual install of `myst-nb==0.16.0` is because it is not in sync with the current jupyterbook
version, I updated this feature to be able to show line numbers in code cells.

### Building the book

After cloning, you can edit the books source files located in the `content/` directory. 

You run

```bash
~/gaohn (venv) $ jupyter-book build content/
```

to build the book, and

```bash
~/gaohn (venv) $ jupyter-book clean content/
```

to clean the build files.

A fully-rendered HTML version of the book will be built in `content/_build/html/`.