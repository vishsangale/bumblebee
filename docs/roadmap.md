# Roadmap

## Phase 0: Foundation
- lock the three research tracks and seed paper list
- build a minimal code and evaluation scaffold
- fix the first `memory_state` evaluation suite before adding architecture variants
- define one tractable benchmark family per track

## Phase 1: Reproduction
- run the `memory_state` core suite on at least one pretrained long-context baseline
- implement a narrow baseline from each track
- verify small-scale reproduction before introducing new ideas
- record failures and ambiguities in `notes/`

## Phase 2: Track Papers
- `memory_state`: write-capable memory plus long-context evaluation
- `adaptive_inference`: latent refinement plus compute-budget analysis
- `hierarchical_programs`: dynamic abstraction plus compositional generalization study

## Phase 3: Synthesis
- combine the strongest ideas across tracks
- test whether the interaction terms matter more than isolated improvements
- target a flagship paper on a hybrid post-transformer architecture
