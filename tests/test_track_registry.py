from shared.track_registry import get_track_spec, list_track_specs


def test_track_registry_reads_all_three_tracks() -> None:
    slugs = [track["slug"] for track in list_track_specs()]
    assert slugs == ["adaptive_inference", "hierarchical_programs", "memory_state"]


def test_track_lookup_returns_expected_metadata() -> None:
    spec = get_track_spec("memory_state")
    assert spec["name"] == "Memory-State Architectures"
    assert spec["seed_papers"]
