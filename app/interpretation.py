"""Gemini-backed extraction of GridWise operator directives.

This module deliberately does not optimize an energy schedule or calculate costs.
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

from .schemas import OptimizeEnergyRequest, OperatorNoteInterpretationResult


class LLMInterpretationError(RuntimeError):
    """Raised when a Gemini result cannot be used as structured directives."""


def build_interpretation_prompt(request: OptimizeEnergyRequest) -> str:
    """Build a fixed, compact extraction prompt for all operator notes."""
    notes = "\n".join(
        f'{index}: {note!r}' for index, note in enumerate(request.operator_notes)
    )
    return f"""You extract GridWise operator directives. Return JSON only.

Interpret every note independently. Return exactly one result per input note, in
note_index order. Do not optimize, schedule energy, calculate costs, change demand,
solar forecasts, tariffs, or battery parameters, and never invent directive types.

Allowed directive_type values and structured_adjustment shapes:
- solar_reduction: {{"hours": [integer], "factor": number}}
- minimum_battery_reserve: {{"hours": [integer], "minimum_energy_kwh": number}}
- no_charge_window: {{"hours": [integer]}}
- no_discharge_window: {{"hours": [integer]}}
- max_grid_window: {{"hours": [integer], "max_grid_kwh": number}}
- no_op: null

For no_op use applies=false and structured_adjustment=null. For every other type,
use applies=true. Hours are unique integer hours 0 through 23 in ascending order.
Use [start, end): include the start and exclude the end; 1 PM to 3 PM is [13, 14].
Solar factor is the usable fraction remaining: "reduced by 80%" and "80% reduction"
mean factor 0.2; "reduced to 80%" and "80% remains" mean factor 0.8.
Convert percentage battery reserves using battery capacity.

Scenario context: 24-hour horizon (hours 0-23); battery capacity is
{request.battery.capacity_kwh} kWh.
Notes:
{notes}
"""


class GeminiDirectiveInterpreter:
    """Makes one Gemini structured-output request for an entire note list."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        client: Any | None = None,
    ) -> None:
        load_dotenv()
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.model = model or os.getenv("LLM_MODEL")
        self.client = client

    def interpret(
        self, request: OptimizeEnergyRequest
    ) -> OperatorNoteInterpretationResult:
        """Interpret all request notes in one Gemini call."""
        if not self.api_key:
            raise LLMInterpretationError("LLM_API_KEY is not configured")
        if not self.model:
            raise LLMInterpretationError("LLM_MODEL is not configured")

        client = self.client or genai.Client(api_key=self.api_key)
        try:
            response = client.models.generate_content(
                model=self.model,
                contents=build_interpretation_prompt(request),
                config=types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                    response_schema=OperatorNoteInterpretationResult,
                ),
            )
            if not response.text:
                raise LLMInterpretationError("Gemini returned an empty response")
            result = OperatorNoteInterpretationResult.model_validate_json(response.text)
            expected_indices = list(range(len(request.operator_notes)))
            actual_indices = [
                directive.note_index for directive in result.directive_interpretation
            ]
            if actual_indices != expected_indices:
                raise LLMInterpretationError(
                    "Gemini response must contain one directive per note in note_index order"
                )
            return result
        except LLMInterpretationError:
            raise
        except Exception as exc:
            raise LLMInterpretationError(
                "Gemini directive interpretation failed"
            ) from exc
