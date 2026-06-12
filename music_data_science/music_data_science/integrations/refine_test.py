"""Tests for the Claude refinement integration (stubbed client, no network)."""

import unittest
from dataclasses import dataclass
from typing import Any, Optional

from music_data_science.integrations.refine import DEFAULT_MODEL, BlueprintRefiner
from music_data_science.stems.models import EditPlan, SongBlueprint


@dataclass
class FakeResponse:
    parsed_output: Any
    stop_reason: Optional[str] = "end_turn"


class FakeMessages:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def parse(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(kwargs)
        return self.response


class FakeClient:
    def __init__(self, response: FakeResponse) -> None:
        self.messages = FakeMessages(response)


class RefineBlueprintTest(unittest.TestCase):
    """Blueprint refinement goes through structured outputs."""

    def test_returns_parsed_blueprint_and_sends_schema(self):
        refined = SongBlueprint(global_caption="dusty lofi hip hop", bpm=84, key_scale="C minor")
        client = FakeClient(FakeResponse(parsed_output=refined))
        result = BlueprintRefiner(client=client).refine_blueprint("a chill lofi beat")
        self.assertEqual(result.bpm, 84)
        call = client.messages.calls[0]
        self.assertEqual(call["model"], DEFAULT_MODEL)
        self.assertIs(call["output_format"], SongBlueprint)
        self.assertIn("a chill lofi beat", call["messages"][0]["content"])

    def test_base_draft_is_included_in_prompt(self):
        client = FakeClient(FakeResponse(parsed_output=SongBlueprint()))
        base = SongBlueprint(bpm=120)
        BlueprintRefiner(client=client).refine_blueprint("more energy", base=base)
        self.assertIn('"bpm":120', client.messages.calls[0]["messages"][0]["content"].replace(" ", ""))

    def test_refusal_raises(self):
        client = FakeClient(FakeResponse(parsed_output=None, stop_reason="refusal"))
        with self.assertRaises(RuntimeError):
            BlueprintRefiner(client=client).refine_blueprint("anything")


class RefinePlanTest(unittest.TestCase):
    """Plan refinement is safe to wire as the translator hook."""

    def test_refined_plan_returned(self):
        refined = EditPlan()
        client = FakeClient(FakeResponse(parsed_output=refined))
        result = BlueprintRefiner(client=client).refine_plan(EditPlan())
        self.assertIs(result, refined)

    def test_refusal_falls_back_to_input_plan(self):
        client = FakeClient(FakeResponse(parsed_output=None, stop_reason="refusal"))
        original = EditPlan()
        result = BlueprintRefiner(client=client).refine_plan(original)
        self.assertIs(result, original)

    def test_hook_adapter_is_callable(self):
        client = FakeClient(FakeResponse(parsed_output=EditPlan()))
        hook = BlueprintRefiner(client=client).as_plan_hook()
        self.assertIsInstance(hook(EditPlan()), EditPlan)


if __name__ == "__main__":
    unittest.main()
