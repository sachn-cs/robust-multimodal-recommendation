# GRE-MC: Robust Multimodal Recommendation via Graph Retrieval-Enhanced Modality Completion

Reproduction of the paper by Li et al. (NUS, 2026).

## Setup

```bash
pip install -r requirements.txt
```

## Data

```bash
python -m data.download
python -m data.features
python -m data.graph_builder
python -m data.masking
```

## Training

```bash
python scripts/train_completion.py
python scripts/train_recommender.py
```

## Evaluation

```bash
python scripts/evaluate.py
```

## Demo

```bash
python demo.py
```
