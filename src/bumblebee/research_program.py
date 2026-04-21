from dataclasses import dataclass


@dataclass(frozen=True)
class ResearchTrack:
    slug: str
    name: str
    thesis: str
    primary_question: str
    first_paper_targets: tuple[str, ...]
    seed_papers: tuple[str, ...]


RESEARCH_TRACKS: dict[str, ResearchTrack] = {
    "memory_state": ResearchTrack(
        slug="memory_state",
        name="Memory-State Architectures",
        thesis=(
            "Explicit write-capable memory and multi-timescale state are a stronger "
            "path to long-horizon "
            "reasoning than continuing to stretch the KV cache."
        ),
        primary_question=(
            "How should a language model learn, compress, and retrieve state at "
            "test time?"
        ),
        first_paper_targets=(
            "Write-gated associative memory",
            "Chunk-level memory consolidation",
            "Long-context evaluation harness",
        ),
        seed_papers=(
            "https://arxiv.org/abs/2405.21060",
            "https://arxiv.org/abs/2407.04620",
            "https://arxiv.org/abs/2501.00663",
            "https://arxiv.org/abs/2505.23735",
        ),
    ),
    "adaptive_inference": ResearchTrack(
        slug="adaptive_inference",
        name="Adaptive Inference",
        thesis=(
            "A strong successor architecture should spend inference compute "
            "selectively through latent "
            "refinement, learned halting, or iterative generation."
        ),
        primary_question=(
            "When should the model think longer, and in what internal space "
            "should it do so?"
        ),
        first_paper_targets=(
            "Latent scratchpad with halting",
            "Difficulty-aware compute allocation",
            "Decode-mode comparison study",
        ),
        seed_papers=(
            "https://arxiv.org/abs/1603.08983",
            "https://arxiv.org/abs/1807.03819",
            "https://arxiv.org/abs/2412.06769",
            "https://arxiv.org/abs/2502.09992",
            "https://arxiv.org/abs/2506.17298",
        ),
    ),
    "hierarchical_programs": ResearchTrack(
        slug="hierarchical_programs",
        name="Hierarchical Programs",
        thesis=(
            "Flat token streams are the wrong abstraction for compositional "
            "reasoning; models need learned "
            "hierarchies, modular workspaces, and explicit intermediate state."
        ),
        primary_question="What structure should a model build internally before it emits text?",
        first_paper_targets=(
            "Dynamic segmentation or patching",
            "Modular latent workspace",
            "Compositional generalization benchmark suite",
        ),
        seed_papers=(
            "https://arxiv.org/abs/1410.5401",
            "https://arxiv.org/abs/1909.10893",
            "https://arxiv.org/abs/2103.01197",
            "https://arxiv.org/abs/2412.09871",
        ),
    ),
}


def list_tracks() -> list[ResearchTrack]:
    """Return research tracks in stable planning order."""
    return [
        RESEARCH_TRACKS[slug]
        for slug in ("memory_state", "adaptive_inference", "hierarchical_programs")
    ]
