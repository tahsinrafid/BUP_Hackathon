import json

import pytest

from app.interpretation import (
    GeminiDirectiveInterpreter,
    LLMInterpretationError,
    build_interpretation_prompt,
)
from app.schemas import OptimizeEnergyRequest


def request_with_notes(notes: list[str]) -> OptimizeEnergyRequest:
    return OptimizeEnergyRequest(
        scenario_id="INTERPRET-001",
        operator_notes=notes,
        hours=[
            {
                "hour": hour,
                "demand_kwh": 100,
                "solar_kwh": 0,
                "tariff_bdt_per_kwh": 8,
            }
            for hour in range(24)
        ],
        battery={
            "capacity_kwh": 200,
            "initial_energy_kwh": 100,
            "minimum_energy_kwh": 40,
            "max_charge_kwh_per_hour": 50,
            "max_discharge_kwh_per_hour": 50,
        },
    )


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


def test_prompt_teaches_required_time_and_percentage_rules() -> None:
    prompt = build_interpretation_prompt(
        request_with_notes(["Keep 50% of the battery from 6 PM to 9 PM."])
    )
    assert "battery capacity is\n200.0 kWh" in prompt
    assert "1 PM to 3 PM is [13, 14]" in prompt
    assert '"reduced by 80%"' in prompt
    assert '"reduced to 80%"' in prompt


def test_interpreter_uses_one_structured_request_for_all_notes() -> None:
    response_text = json.dumps(
        {
            "directive_interpretation": [
                {
                    "note_index": 0,
                    "applies": True,
                    "directive_type": "solar_reduction",
                    "structured_adjustment": {"hours": [13, 14], "factor": 0.2},
                    "explanation": "Usable solar is 20% during the window.",
                },
                {
                    "note_index": 1,
                    "applies": False,
                    "directive_type": "no_op",
                    "structured_adjustment": None,
                    "explanation": "This does not affect energy operations.",
                },
            ]
        }
    )
    fake_client = FakeClient(response_text)
    interpreter = GeminiDirectiveInterpreter(
        api_key="test-key", model="test-model", client=fake_client
    )

    result = interpreter.interpret(
        request_with_notes(
            [
                "Solar is reduced by 80% from 1 PM to 3 PM.",
                "The cafeteria menu changes tomorrow.",
            ]
        )
    )

    assert len(fake_client.models.calls) == 1
    assert result.directive_interpretation[0].note_index == 0
    assert result.directive_interpretation[0].structured_adjustment.model_dump() == {
        "hours": [13, 14],
        "factor": 0.2,
    }
    assert result.directive_interpretation[1].directive_type == "no_op"
    assert result.directive_interpretation[1].applies is False


def test_missing_llm_configuration_is_a_controlled_error(monkeypatch) -> None:
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setattr("app.interpretation.load_dotenv", lambda: False)
    interpreter = GeminiDirectiveInterpreter(api_key="", model="")
    with pytest.raises(LLMInterpretationError, match="LLM_API_KEY"):
        interpreter.interpret(request_with_notes(["No energy changes today."]))


def test_invalid_no_op_semantics_are_rejected() -> None:
    fake_client = FakeClient(
        json.dumps(
            {
                "directive_interpretation": [
                    {
                        "note_index": 0,
                        "applies": True,
                        "directive_type": "no_op",
                        "structured_adjustment": None,
                        "explanation": "Incorrect no-op output.",
                    }
                ]
            }
        )
    )
    interpreter = GeminiDirectiveInterpreter(
        api_key="test-key", model="test-model", client=fake_client
    )

    with pytest.raises(LLMInterpretationError):
        interpreter.interpret(request_with_notes(["No energy changes today."]))
