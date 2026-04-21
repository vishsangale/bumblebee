# Research Program

## Objective
Build a publishable research program around a successor architecture to the standard Transformer. The repo is intentionally organized around three axes that look both empirically live and long-term important: memory, compute allocation, and abstraction.

## Selection Criteria
Each direction should satisfy four filters:
- it changes model capability, not just FLOPs
- it has at least one recent primary-source result worth building on
- it supports a staged path from small experiments to scale-up work
- it exposes measurable internal structure, not only benchmark gains

## Direction 1: Memory-State Architectures
Hypothesis: the KV cache is a strong short-term retrieval mechanism but a weak model of persistent memory. We should study architectures with explicit write operations, multi-timescale state, and test-time adaptation.

What grounds this track:
- `Transformers are SSMs / Mamba-2` shows the recurrent-attention boundary is more fluid than it first appears
- `Learning to (Learn at Test Time)` and `Titans` argue for expressive hidden state and neural memory at inference time
- `ATLAS` pushes this further by optimizing memory with respect to current and past context

First paper targets:
- a write-gated associative memory block
- memory consolidation across chunks
- long-context evaluation beyond naive context extension

## Direction 2: Adaptive Inference
Hypothesis: fixed one-pass decoding is too rigid. A better architecture should spend extra compute only when the problem requires it, ideally in latent space with learned halting or refinement.

What grounds this track:
- `Adaptive Computation Time` and `Universal Transformer` established the variable-compute idea early
- `COCONUT` shows continuous latent reasoning can outperform explicit chain-of-thought on some backtracking-heavy tasks
- `LLaDA` and `Mercury` show diffusion-style language modeling is now plausible at useful scale

First paper targets:
- latent scratchpad refinement with a halting controller
- difficulty-conditioned compute budgets
- comparisons between autoregressive, latent-recurrent, and diffusion-like decoding

## Direction 3: Hierarchical Programs
Hypothesis: flat token streams are the wrong abstraction for compositional reasoning. We should explore learned compression hierarchies, modular state, and explicit intermediate workspaces.

What grounds this track:
- `Byte Latent Transformer` shows dynamic patches can beat fixed tokenization assumptions
- `Neural Turing Machines`, `Recurrent Independent Mechanisms`, and shared-workspace models provide a path toward modular computation
- these systems better match the idea that reasoning should create and manipulate intermediate state, not only emit tokens

First paper targets:
- dynamic segment or patch formation beyond fixed tokenization
- modular latent workspaces with sparse communication
- synthetic compositional tasks that reveal causal structure, not only language loss

## Cross-Cutting Constraint
We will use white-box pressure as a design rule across all three tracks. Inspired by CRATE, each proposal should expose interpretable state, clear ablations, and a concrete claim about what computation a block is performing.
