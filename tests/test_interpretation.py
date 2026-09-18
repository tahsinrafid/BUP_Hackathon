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


def valid_directive() -> dict:
    return {
        "note_index": 0,
        "applies": True,
        "directive_type": "solar_reduction",
        "structured_adjustment": {"hours": [13, 14], "factor": 0.2},
        "explanation": "Solar availability is reduced.",
    }


def assert_invalid_llm_output(payload: dict, notes: list[str] | None = None) -> None:
    fake_client = FakeClient(json.dumps(payload, allow_nan=True))
    interpreter = GeminiDirectiveInterpreter(
        api_key="test-key", model="test-model", client=fake_client
    )
    with pytest.raises(LLMInterpretationError):
        interpreter.interpret(request_with_notes(notes or ["Solar maintenance."]))


def test_duplicate_note_index_is_rejected() -> None:
    first = valid_directive()
    second = valid_directive()
    second["explanation"] = "A second result with the same index."
    assert_invalid_llm_output(
        {"directive_interpretation": [first, second]},
        ["First note.", "Second note."],
    )


def test_missing_interpretation_is_rejected() -> None:
    assert_invalid_llm_output({"directive_interpretation": []})


def test_note_indexes_must_follow_original_note_order() -> None:
    first = valid_directive()
    first["note_index"] = 1
    second = valid_directive()
    second["note_index"] = 0
    assert_invalid_llm_output(
        {"directive_interpretation": [first, second]},
        ["First note.", "Second note."],
    )


@pytest.mark.parametrize(
    ("case", "hours"),
    [
        ("hour_24", [13, 24]),
        ("negative_hour", [-1, 13]),
        ("unsorted_hours", [14, 13]),
        ("duplicate_hours", [13, 13]),
        ("non_integer_hour", [13.0, 14]),
        ("empty_hours", []),
    ],
)
def test_invalid_hour_arrays_are_rejected(case: str, hours: list[int]) -> None:
    directive = valid_directive()
    directive["structured_adjustment"]["hours"] = hours
    assert_invalid_llm_output({"directive_interpretation": [directive]})


def test_solar_factor_above_one_is_rejected() -> None:
    directive = valid_directive()
    directive["structured_adjustment"]["factor"] = 1.5
    assert_invalid_llm_output({"directive_interpretation": [directive]})


def test_unsupported_directive_type_is_rejected() -> None:
    directive = valid_directive()
    directive["directive_type"] = "change_tariff"
    assert_invalid_llm_output({"directive_interpretation": [directive]})


def test_reserve_above_battery_capacity_is_rejected() -> None:
    directive = {
        "note_index": 0,
        "applies": True,
        "directive_type": "minimum_battery_reserve",
        "structured_adjustment": {"hours": [18, 19], "minimum_energy_kwh": 201},
        "explanation": "Reserve request.",
    }
    assert_invalid_llm_output({"directive_interpretation": [directive]})


def test_no_op_with_applies_true_is_rejected() -> None:
    directive = {
        "note_index": 0,
        "applies": True,
        "directive_type": "no_op",
        "structured_adjustment": None,
        "explanation": "Incorrect no-op.",
    }
    assert_invalid_llm_output({"directive_interpretation": [directive]})


def test_non_no_op_with_applies_false_is_rejected() -> None:
    directive = valid_directive()
    directive["applies"] = False
    assert_invalid_llm_output({"directive_interpretation": [directive]})


def test_missing_adjustment_is_rejected() -> None:
    directive = valid_directive()
    directive["structured_adjustment"] = None
    assert_invalid_llm_output({"directive_interpretation": [directive]})


@pytest.mark.parametrize("max_grid_kwh", [-1, float("inf")])
def test_invalid_max_grid_value_is_rejected(max_grid_kwh: float) -> None:
    directive = {
        "note_index": 0,
        "applies": True,
        "directive_type": "max_grid_window",
        "structured_adjustment": {"hours": [18, 19], "max_grid_kwh": max_grid_kwh},
        "explanation": "Grid cap.",
    }
    assert_invalid_llm_output({"directive_interpretation": [directive]})


@pytest.mark.parametrize(
    ("case", "field", "value"),
    [
        ("nan_factor", "factor", float("nan")),
        ("infinite_factor", "factor", float("inf")),
    ],
)
def test_nan_and_infinity_are_rejected(case: str, field: str, value: float) -> None:
    directive = valid_directive()
    directive["structured_adjustment"][field] = value
    assert_invalid_llm_output({"directive_interpretation": [directive]})


@pytest.mark.parametrize("case", ["missing_critical_field", "unexpected_critical_field"])
def test_missing_or_unexpected_critical_fields_are_rejected(case: str) -> None:
    directive = valid_directive()
    if case == "missing_critical_field":
        del directive["explanation"]
    else:
        directive["unexpected_critical_field"] = "not allowed"
    assert_invalid_llm_output({"directive_interpretation": [directive]})
