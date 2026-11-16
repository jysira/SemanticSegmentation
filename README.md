# Semantic Segmentation

Use the industrial_vision Conda environment.

```cmd
python scripts/train_semantic_clahe.py
python scripts/manual_error_review.py apply --preserve-holes
```

Data: data/pcb_semantic_clahe/dataset.yaml. Review workspace: reports/manual_semantic_review_pcb4. Compression guide: docs/semantic_compression.md.
