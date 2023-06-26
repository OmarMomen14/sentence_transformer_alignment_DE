# DEplain Alignment via Sentence Transformers

As a part of the paper ["DEplain: A German Parallel Corpus with Intralingual Translations into Plain Language for Sentence and Document Simplification."](https://arxiv.org/abs/2305.18939), we developed and evaluated a simple method utilizing sentence transformers to align German text datasets automatically.

## Usage

After cloning the repository

1. Setup the environment
```
python3 -m venv env
source env/bin/activate
pip install -U pip setuptools
pip install -r requirements.txt
```

2. Go through the `procedure.ipynb` notebook for aligning your documents