"""Gemini-backed extraction of GridWise operator directives.

This module deliberately does not optimize an energy schedule or calculate costs.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

from .directive_validation import DirectiveValidationError, validate_llm_interpretation
from .schemas import OptimizeEnergyRequest, OperatorNoteInterpretationResult


logger = logging.getLogger(__name__)


class LLMInterpretationError(RuntimeError):
    """Raised when a Gemini result cannot be used as structured directives."""


# Gemini structured output does not accept Pydantic's generated
# ``additionalProperties`` fields. This provider-facing schema keeps the same
# response shape; Pydantic plus deterministic validation remains the strict
# authority after Gemini returns JSON.
GEMINI_INTERPRETATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "directive_interpretation": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "note_index": {"type": "integer"},
                    "applies": {"type": "boolean"},
                    "directive_type": {
                        "type": "string",
                        "enum": [
                            "solar_reduction",
                            "minimum_battery_reserve",
                            "no_charge_window",
                            "no_discharge_window",
                            "max_grid_window",
                            "no_op",
                        ],
                    },
                    "structured_adjustment": {
                        "anyOf": [
                            {
                                "type": "object",
                                "properties": {
                                    "hours": {
                                        "type": "array",
                                        "items": {"type": "integer"},
                                    },
                                    "factor": {"type": "number"},
                                    "minimum_energy_kwh": {"type": "number"},
                                    "max_grid_kwh": {"type": "number"},
                                },
                            },
                            {"type": "null"},
                        ]
                    },
                    "explanation": {"type": "string"},
                },
                "required": [
                    "note_index",
                    "applies",
                    "directive_type",
                    "structured_adjustment",
                    "explanation",
                ],
            },
        }
    },
    "required": ["directive_interpretation"],
}

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
Use [start, end) for every time-range wording: include the start and exclude the
end. This applies equally to "from X to Y", "from X until Y", and "between X and
Y"; never include the ending hour. For example, 1 PM to 3 PM is [13, 14].
Use standard clock conversion: 12 AM is hour 0 and 12 PM is hour 12. A one-hour
window contains only its starting hour. Normalize ordinary fractions exactly.
Solar factor is the usable fraction remaining: "reduced by 80%" and "80% reduction"
mean factor 0.2; "reduced to 80%" and "80% remains" mean factor 0.8.
The same distinction applies to fractions: "reduced by one-third" leaves factor
2/3, while "reduced to one-third" means factor 1/3. Numbers or times alone do not
make an unrelated note relevant to energy scheduling.
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
        load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")
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
            prompt = build_interpretation_prompt(request)
            config = types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
                response_schema=GEMINI_INTERPRETATION_SCHEMA,
            )

            def generate():
                return client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )

            try:
                response = generate()
            except Exception as first_error:
                if type(first_error).__name__ != "ServerError":
                    raise
                logger.warning(
                    "Gemini provider unavailable; retrying once with model=%s",
                    self.model,
                )
                try:
                    response = generate()
                except Exception as retry_error:
                    raise LLMInterpretationError(
                        "Gemini provider unavailable after one controlled retry"
                    ) from retry_error
            if not response.text:
                raise LLMInterpretationError("Gemini returned an empty response")
            result = OperatorNoteInterpretationResult.model_validate_json(response.text)
            return validate_llm_interpretation(result, request)
        except LLMInterpretationError:
            raise
        except DirectiveValidationError as exc:
            raise LLMInterpretationError(
                "LLM output failed deterministic directive validation"
            ) from exc
        except Exception as exc:
            raise LLMInterpretationError(
                "Gemini directive interpretation failed"
            ) from exc
