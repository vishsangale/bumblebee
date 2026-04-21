# Experiments

Keep experiment code thin and reproducible. As the repo grows, organize entry points by track:

- `experiments/memory_state/`
- `experiments/adaptive_inference/`
- `experiments/hierarchical_programs/`
- `experiments/shared/`

Shared evaluation harnesses should live in `src/bumblebee/` once they become reusable across tracks.
