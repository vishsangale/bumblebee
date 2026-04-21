# Experiments

Keep experiment entry points organized by research track:

- `experiments/memory_state/`
- `experiments/adaptive_inference/`
- `experiments/hierarchical_programs/`

Use `experiments/train.py` as the minimal Hydra + PyTorch + TensorBoard smoke runner for the repo scaffold. Promote reusable utilities into `src/<track>/` or `src/shared/` once they stop being one-off experiment code.
