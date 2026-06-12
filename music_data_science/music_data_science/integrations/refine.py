"""Claude-backed refinement for blueprints and edit plans.

The deterministic translator works without any LLM; this module is the
optional "real backend" behind its hooks.  It uses the official ``anthropic``
SDK (``pip install music-data-science[llm]``) with structured outputs
(``client.messages.parse``), so responses are validated against the pydantic
models — no free-text parsing.  The client resolves ``ANTHROPIC_API_KEY``
from the environment.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from music_data_science.stems.models import EditPlan, SongBlueprint

DEFAULT_MODEL = "claude-opus-4-8"

_BLUEPRINT_SYSTEM = (
    "You are a music producer turning a song idea into a structured production "
    "blueprint. Fill in global_caption (rich style/instrumentation description), "
    "bpm, key_scale (e.g. 'C major'), time_signature (e.g. '4/4'), duration in "
    "seconds, and one StemSpec per layer the arrangement needs. Stay faithful to "
    "every constraint the user states; choose professional, genre-appropriate "
    "values for anything unspecified. Captions must describe sound, not feelings."
)

_PLAN_SYSTEM = (
    "You review a deterministic stem-edit plan produced from a user's prompt. "
    "Keep the plan's structure and operations unless they misread the intent. "
    "You may tighten time windows, adjust repaint strength, reword captions to be "
    "more production-precise, and order edits so destructive ones run last. "
    "Never invent edits the user did not ask for."
)


class BlueprintRefiner:
    """Refine blueprints and edit plans with Claude structured outputs.

    Args:
        model: Claude model id; defaults to :data:`DEFAULT_MODEL`.
        client: Injectable Anthropic-compatible client (used by tests). When
            omitted, the ``anthropic`` package is imported lazily.
        max_tokens: Output ceiling for refinement calls.

    Raises:
        ImportError: If no client is given and ``anthropic`` is not installed.
    """

    def __init__(self, model: str = DEFAULT_MODEL, client: Optional[Any] = None, max_tokens: int = 16000) -> None:
        if client is None:
            import anthropic  # noqa: PLC0415 -- optional dependency, imported lazily

            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    def _parse(self, system: str, prompt: str, output_format: type) -> Any:
        """One structured-output call; raises on refusal instead of returning junk."""
        response = self.client.messages.parse(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_format=output_format,
        )
        if getattr(response, "stop_reason", None) == "refusal":
            raise RuntimeError("model declined the refinement request")
        return response.parsed_output

    def refine_blueprint(self, description: str, base: Optional[SongBlueprint] = None) -> SongBlueprint:
        """Turn a free-text song description (plus optional draft) into a blueprint."""
        prompt = f"Song description:\n{description}"
        if base is not None:
            prompt += f"\n\nCurrent draft blueprint (improve, don't discard):\n{base.model_dump_json()}"
        return self._parse(_BLUEPRINT_SYSTEM, prompt, SongBlueprint)

    def refine_plan(self, plan: EditPlan) -> EditPlan:
        """Polish a deterministic edit plan; falls back to the input on refusal."""
        try:
            return self._parse(_PLAN_SYSTEM, plan.model_dump_json(), EditPlan)
        except RuntimeError:
            return plan

    def as_plan_hook(self) -> Callable[[EditPlan], EditPlan]:
        """Adapter for ``PromptTranslator(llm_refine=...)``."""
        return self.refine_plan
