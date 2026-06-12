"""Tests for the stem domain model."""

import unittest

from pydantic import ValidationError

from music_data_science.stems.models import (
    EditPlan,
    SongBlueprint,
    StemClass,
    StemEdit,
    StemOperation,
    StemSpec,
)

# Mirror of acestep.constants.TRACK_NAMES; asserted here so drift is caught
# without importing the heavy acestep package.
ACESTEP_TRACK_NAMES = [
    "woodwinds", "brass", "fx", "synth", "strings", "percussion",
    "keyboard", "guitar", "bass", "drums", "backing_vocals", "vocals",
]


class StemClassTest(unittest.TestCase):
    """StemClass must stay aligned with ACE-Step's track vocabulary."""

    def test_values_match_acestep_track_names(self):
        self.assertEqual(sorted(item.value for item in StemClass), sorted(ACESTEP_TRACK_NAMES))

    def test_is_string_enum(self):
        self.assertEqual(StemClass.DRUMS.value, "drums")
        self.assertEqual(StemClass("vocals"), StemClass.VOCALS)


class StemOperationTest(unittest.TestCase):
    """Operations map 1:1 onto ACE-Step task types."""

    def test_task_type_mapping(self):
        expected = {
            StemOperation.REGENERATE: "repaint",
            StemOperation.REPLACE_STYLE: "cover",
            StemOperation.ADD_LAYER: "lego",
            StemOperation.SEPARATE: "extract",
            StemOperation.EXTEND: "complete",
        }
        for operation, task_type in expected.items():
            self.assertEqual(operation.task_type, task_type)

    def test_every_operation_has_a_task_type(self):
        for operation in StemOperation:
            self.assertTrue(operation.task_type)


class StemSpecTest(unittest.TestCase):
    """StemSpec defaults and validation."""

    def test_minimal_spec(self):
        spec = StemSpec(stem_class=StemClass.BASS)
        self.assertIsNone(spec.bpm)
        self.assertIsNone(spec.seed)
        self.assertEqual(spec.caption, "")

    def test_rejects_unknown_stem_class(self):
        with self.assertRaises(ValidationError):
            StemSpec(stem_class="theremin")


class SongBlueprintTest(unittest.TestCase):
    """Blueprint lookup helpers."""

    def test_stem_lookup(self):
        blueprint = SongBlueprint(stems=[
            StemSpec(stem_class=StemClass.DRUMS, caption="boom bap"),
            StemSpec(stem_class=StemClass.BASS),
        ])
        found = blueprint.stem(StemClass.DRUMS)
        self.assertIsNotNone(found)
        self.assertEqual(found.caption, "boom bap")
        self.assertIsNone(blueprint.stem(StemClass.SYNTH))


class StemEditTest(unittest.TestCase):
    """StemEdit defaults and bounds."""

    def test_defaults(self):
        edit = StemEdit(target=StemClass.GUITAR)
        self.assertIs(edit.operation, StemOperation.REGENERATE)
        self.assertEqual(edit.repaint_mode, "balanced")
        self.assertEqual(edit.window_start, 0.0)
        self.assertIsNone(edit.window_end)

    def test_strength_bounds(self):
        with self.assertRaises(ValidationError):
            StemEdit(target=StemClass.GUITAR, repaint_strength=1.5)


class EditPlanTest(unittest.TestCase):
    """EditPlan preserves edit order."""

    def test_preserves_order(self):
        plan = EditPlan(edits=[
            StemEdit(target=StemClass.DRUMS),
            StemEdit(target=StemClass.BASS),
        ])
        self.assertEqual([edit.target for edit in plan.edits], [StemClass.DRUMS, StemClass.BASS])
        self.assertEqual(plan.acceptance_criteria, [])


if __name__ == "__main__":
    unittest.main()
