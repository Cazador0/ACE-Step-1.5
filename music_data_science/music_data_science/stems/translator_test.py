"""Tests for deterministic prompt -> StemEdit / request translation."""

import unittest

from music_data_science.stems.models import SongBlueprint, StemClass, StemEdit, StemOperation, StemSpec
from music_data_science.stems.translator import PromptTranslator, blueprint_to_request, to_request


class TranslateEditTest(unittest.TestCase):
    """Keyword/regex grammar coverage."""

    def setUp(self):
        self.translator = PromptTranslator()

    def test_repaint_with_clock_window(self):
        edit = self.translator.translate_edit("make the drums punchier from 0:45 to 1:10")
        self.assertEqual(edit.target, StemClass.DRUMS)
        self.assertIs(edit.operation, StemOperation.REGENERATE)
        self.assertEqual(edit.window_start, 45.0)
        self.assertEqual(edit.window_end, 70.0)
        self.assertEqual(edit.repaint_mode, "balanced")

    def test_seconds_window(self):
        edit = self.translator.translate_edit("rework the guitar from 12s to 30s")
        self.assertEqual(edit.window_start, 12.0)
        self.assertEqual(edit.window_end, 30.0)

    def test_replace_with_targets_the_replaced_stem(self):
        edit = self.translator.translate_edit("replace the synth line with strings in the chorus")
        self.assertEqual(edit.target, StemClass.SYNTH)
        self.assertIs(edit.operation, StemOperation.REPLACE_STYLE)

    def test_add_layer(self):
        edit = self.translator.translate_edit("add a shimmering strings layer")
        self.assertEqual(edit.target, StemClass.STRINGS)
        self.assertIs(edit.operation, StemOperation.ADD_LAYER)

    def test_separate(self):
        edit = self.translator.translate_edit("extract the vocals")
        self.assertEqual(edit.target, StemClass.VOCALS)
        self.assertIs(edit.operation, StemOperation.SEPARATE)

    def test_extend(self):
        edit = self.translator.translate_edit("extend the piano outro")
        self.assertEqual(edit.target, StemClass.KEYBOARD)
        self.assertIs(edit.operation, StemOperation.EXTEND)

    def test_subtle_maps_to_conservative(self):
        edit = self.translator.translate_edit("subtly tweak the bass groove")
        self.assertEqual(edit.repaint_mode, "conservative")
        self.assertLess(edit.repaint_strength, 0.5)

    def test_completely_different_maps_to_aggressive(self):
        edit = self.translator.translate_edit("make the guitar solo completely different")
        self.assertEqual(edit.repaint_mode, "aggressive")
        self.assertGreater(edit.repaint_strength, 0.5)

    def test_backing_vocals_beats_vocals(self):
        edit = self.translator.translate_edit("redo the backing vocals")
        self.assertEqual(edit.target, StemClass.BACKING_VOCALS)

    def test_default_target_when_no_stem_mentioned(self):
        edit = self.translator.translate_edit("brighten it up a bit", default_target=StemClass.FX)
        self.assertEqual(edit.target, StemClass.FX)


class TranslatePlanTest(unittest.TestCase):
    """EditPlan assembly and the llm_refine hook."""

    def test_plan_preserves_order_and_criteria(self):
        plan = PromptTranslator().translate(["mute-ish subtle bass tweak", "extract the drums"])
        self.assertEqual(len(plan.edits), 2)
        self.assertEqual(plan.edits[1].operation, StemOperation.SEPARATE)
        self.assertEqual(len(plan.acceptance_criteria), 2)

    def test_llm_refine_hook_is_called(self):
        calls = []

        def refine(plan):
            calls.append(plan)
            plan.recombine_instructions = "refined"
            return plan

        plan = PromptTranslator(llm_refine=refine).translate("extract the drums")
        self.assertEqual(len(calls), 1)
        self.assertEqual(plan.recombine_instructions, "refined")


class ToRequestTest(unittest.TestCase):
    """Emitted payloads must match the GenerateMusicRequest contract."""

    def setUp(self):
        self.blueprint = SongBlueprint(
            global_caption="warm lo-fi hip hop",
            lyrics="[verse] dust and rain",
            bpm=84,
            key_scale="F major",
            time_signature="4",
            duration=120.0,
            stems=[StemSpec(stem_class=StemClass.DRUMS, caption="dusty kit", seed=7)],
        )

    def test_repaint_payload(self):
        edit = StemEdit(
            target=StemClass.DRUMS, intent="punchier drums", window_start=45.0, window_end=70.0,
            repaint_mode="aggressive", repaint_strength=0.9,
        )
        payload = to_request(edit, self.blueprint, src_audio_path="/tmp/song.wav")
        self.assertEqual(payload["task_type"], "repaint")
        self.assertEqual(payload["repainting_start"], 45.0)
        self.assertEqual(payload["repainting_end"], 70.0)
        self.assertEqual(payload["repaint_mode"], "aggressive")
        self.assertEqual(payload["src_audio_path"], "/tmp/song.wav")
        self.assertEqual(payload["bpm"], 84)
        self.assertEqual(payload["key_scale"], "F major")
        # Seed pinned from the blueprint's stem spec.
        self.assertEqual(payload["seed"], 7)
        self.assertFalse(payload["use_random_seed"])

    def test_cover_payload(self):
        edit = StemEdit(target=StemClass.SYNTH, operation=StemOperation.REPLACE_STYLE, cover_strength=0.8)
        payload = to_request(edit, self.blueprint)
        self.assertEqual(payload["task_type"], "cover")
        self.assertEqual(payload["audio_cover_strength"], 0.8)
        self.assertNotIn("src_audio_path", payload)
        self.assertTrue(payload["use_random_seed"])
        self.assertEqual(payload["seed"], -1)

    def test_lego_and_extract_carry_track_fields(self):
        for operation, task_type in [(StemOperation.ADD_LAYER, "lego"), (StemOperation.SEPARATE, "extract")]:
            edit = StemEdit(target=StemClass.STRINGS, operation=operation)
            payload = to_request(edit, self.blueprint)
            self.assertEqual(payload["task_type"], task_type)
            self.assertEqual(payload["track_name"], "strings")
            self.assertEqual(payload["track_classes"], ["strings"])

    def test_complete_payload(self):
        edit = StemEdit(target=StemClass.BASS, operation=StemOperation.EXTEND)
        payload = to_request(edit, self.blueprint)
        self.assertEqual(payload["task_type"], "complete")
        self.assertEqual(payload["track_classes"], ["bass"])

    def test_overrides_win(self):
        edit = StemEdit(target=StemClass.DRUMS)
        payload = to_request(edit, self.blueprint, inference_steps=16)
        self.assertEqual(payload["inference_steps"], 16)

    def test_blueprint_to_request(self):
        payload = blueprint_to_request(self.blueprint)
        self.assertEqual(payload["task_type"], "text2music")
        self.assertEqual(payload["audio_duration"], 120.0)
        self.assertEqual(payload["lyrics"], "[verse] dust and rain")


if __name__ == "__main__":
    unittest.main()
