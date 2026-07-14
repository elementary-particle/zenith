from pathlib import Path


def test_executable_surface_uses_env_and_action_vocabulary():
    roots = (
        Path("training/src"),
        Path("riichi/core/src"),
        Path("riichi/src/batch_env"),
        Path("riichi/python"),
    )
    forbidden = (
        "zenith_ppo.simulator",
        "selectionbatch",
        "statebuffers",
        "eventbuffers",
        "data_contract_version",
        "api_schema_version",
    )
    offenders = []
    for root in roots:
        for path in root.rglob("*"):
            if path.suffix not in {".py", ".rs"}:
                continue
            text = path.read_text(encoding="utf-8").lower()
            for term in forbidden:
                if term in text:
                    offenders.append(f"{path}: {term}")
    assert not offenders, "legacy executable vocabulary remains:\n" + "\n".join(offenders)
