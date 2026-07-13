import pytest

from llmgen import SkillRecord


def test_skill_metadata_is_immutable_after_validation() -> None:
    original = {"owner": "calendar", "tags": ["read"]}
    record = SkillRecord("calendar.read", metadata=original)
    original["owner"] = "mutated-outside"
    original["tags"].append("mutated-outside")

    assert record.metadata["owner"] == "calendar"
    assert record.metadata["tags"] == ("read",)
    with pytest.raises(TypeError):
        record.metadata["owner"] = "mutated-inside"
    assert SkillRecord.from_dict(record.to_dict()) == record
