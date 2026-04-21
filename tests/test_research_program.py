from bumblebee import RESEARCH_TRACKS, list_tracks


def test_track_registry_has_expected_slugs() -> None:
    assert tuple(RESEARCH_TRACKS) == (
        "memory_state",
        "adaptive_inference",
        "hierarchical_programs",
    )


def test_tracks_have_seed_papers_and_targets() -> None:
    for track in list_tracks():
        assert track.seed_papers
        assert track.first_paper_targets
        assert track.primary_question.endswith("?")
