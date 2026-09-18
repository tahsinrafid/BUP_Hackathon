"""Interpretation-layer checks based on public cases and original paraphrases."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.interpretation import GeminiDirectiveInterpreter
from app.schemas import OptimizeEnergyRequest


SAMPLE_CASES_PATH = (
    Path(__file__).resolve().parents[1]
    / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json"
)
PUBLIC_CASES = json.loads(SAMPLE_CASES_PATH.read_text(encoding="utf-8"))["cases"]


class FakeModels:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return type("Response", (), {"text": self.response_text})()


class FakeClient:
    def __init__(self, response_text: str) -> None:
        self.models = FakeModels(response_text)


def make_interpreter(expected_directives: list[dict]) -> tuple[GeminiDirectiveInterpreter, FakeClient]:
    """Return a mock Gemini client with only machine-checkable output fields."""
    response = {
        "directive_interpretation": [
            {
                "note_index": directive["note_index"],
                "applies": directive["applies"],
                "directive_type": directive["directive_type"],
                "structured_adjustment": directive["structured_adjustment"],
                "explanation": "Test-only structured interpretation.",
            }
            for directive in expected_directives
        ]
    }
    client = FakeClient(json.dumps(response))
    return (
        GeminiDirectiveInterpreter(
            api_key="test-key", model="test-model", client=client
        ),
        client,
    )


def structured_meaning(result) -> list[dict]:
    """Discard free text and retain only organizer-machine-checkable fields."""
    return [
        {
            "note_index": directive.note_index,
            "applies": directive.applies,
            "directive_type": directive.directive_type,
            "structured_adjustment": (
                directive.structured_adjustment.model_dump()
                if directive.structured_adjustment is not None
                else None
            ),
        }
        for directive in result.directive_interpretation
    ]


@pytest.mark.parametrize("case", PUBLIC_CASES, ids=lambda case: case["id"])
def test_all_public_cases_preserve_required_structured_meaning(case: dict) -> None:
    request = OptimizeEnergyRequest.model_validate(case["input"])
    expected = case["expected_output"]["directive_interpretation"]
    interpreter, client = make_interpreter(expected)

    result = interpreter.interpret(request)

    assert len(client.models.calls) == 1
    assert structured_meaning(result) == [
        {
            "note_index": directive["note_index"],
            "applies": directive["applies"],
            "directive_type": directive["directive_type"],
            "structured_adjustment": directive["structured_adjustment"],
        }
        for directive in expected
    ]


@pytest.mark.parametrize(
    ("notes", "expected_directives"),
    [
        (
            ["PV availability will be cut by four-fifths from 09:00 to 11:00."],
            [
                {
                    "note_index": 0,
                    "applies": True,
                    "directive_type": "solar_reduction",
                    "structured_adjustment": {"hours": [9, 10], "factor": 0.2},
                }
            ],
        ),
        (
            ["Maintain 30 percent state of charge between 20:00 and 22:00."],
            [
                {
                    "note_index": 0,
                    "applies": True,
                    "directive_type": "minimum_battery_reserve",
                    "structured_adjustment": {
                        "hours": [20, 21],
                        "minimum_energy_kwh": 60,
                    },
                }
            ],
        ),
        (
            ["Do not replenish the battery during the 00:00-02:00 inspection."],
            [
                {
                    "note_index": 0,
                    "applies": True,
                    "directive_type": "no_charge_window",
                    "structured_adjustment": {"hours": [0, 1]},
                }
            ],
        ),
        (
            ["Battery output is prohibited from 15:00 through 17:00."],
            [
                {
                    "note_index": 0,
                    "applies": True,
                    "directive_type": "no_discharge_window",
                    "structured_adjustment": {"hours": [15, 16]},
                }
            ],
        ),
        (
            ["Limit utility draw to 125 kWh each hour from 04:00 to 06:00."],
            [
                {
                    "note_index": 0,
                    "applies": True,
                    "directive_type": "max_grid_window",
                    "structured_adjustment": {
                        "hours": [4, 5],
                        "max_grid_kwh": 125,
                    },
                }
            ],
        ),
        (
            ["The campus newsletter will be printed next Tuesday."],
            [
                {
                    "note_index": 0,
                    "applies": False,
                    "directive_type": "no_op",
                    "structured_adjustment": None,
                }
            ],
        ),
    ],
)
def test_original_paraphrases_preserve_structured_meaning(
    notes: list[str], expected_directives: list[dict]
) -> None:
    request = OptimizeEnergyRequest.model_validate(
        {
            "scenario_id": "PARAPHRASE-001",
            "operator_notes": notes,
            "hours": [
                {
                    "hour": hour,
                    "demand_kwh": 100,
                    "solar_kwh": 20,
                    "tariff_bdt_per_kwh": 8,
                }
                for hour in range(24)
            ],
            "battery": {
                "capacity_kwh": 200,
                "initial_energy_kwh": 100,
                "minimum_energy_kwh": 40,
                "max_charge_kwh_per_hour": 50,
                "max_discharge_kwh_per_hour": 50,
            },
        }
    )
    interpreter, client = make_interpreter(expected_directives)

    result = interpreter.interpret(request)

    assert len(client.models.calls) == 1
    assert structured_meaning(result) == expected_directives
