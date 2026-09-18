"""Adversarial cases likely to appear in hidden organizer evaluation."""

from __future__ import annotations

import json

import pytest

from app.interpretation import (
    GeminiDirectiveInterpreter,
    LLMInterpretationError,
    build_interpretation_prompt,
)
from app.optimizer import DirectiveOverlapError, EPSILON, _clean, optimize_energy_schedule
from app.replay_validator import validate_solved_plan
from app.schemas import (
    DirectiveInterpretation,
    OperatorNoteInterpretationResult,
    OptimizeEnergyRequest,
)


def make_request(
    *,
    notes: list[str] | None = None,
    demand: float = 10,
    solar: float = 0,
    tariffs: list[float] | None = None,
    capacity: float = 20,
    initial: float = 10,
    minimum: float = 0,
    max_charge: float = 10,
    max_discharge: float = 10,
) -> OptimizeEnergyRequest:
    tariffs = tariffs or [1.0] * 24
    return OptimizeEnergyRequest.model_validate(
        {
            "scenario_id": "ADVERSARIAL",
            "operator_notes": notes or ["No temporary energy restrictions today."],
            "hours": [
                {
                    "hour": hour,
                    "demand_kwh": demand,
                    "solar_kwh": solar,
                    "tariff_bdt_per_kwh": tariffs[hour],
                }
                for hour in range(24)
            ],
            "battery": {
                "capacity_kwh": capacity,
                "initial_energy_kwh": initial,
                "minimum_energy_kwh": minimum,
                "max_charge_kwh_per_hour": max_charge,
                "max_discharge_kwh_per_hour": max_discharge,
            },
        }
    )


def directive(kind: str, adjustment: dict, note_index: int = 0) -> DirectiveInterpretation:
    return DirectiveInterpretation.model_validate(
        {
            "note_index": note_index,
            "applies": True,
            "directive_type": kind,
            "structured_adjustment": adjustment,
            "explanation": "Adversarial test directive.",
        }
    )


class FakeModels:
    def __init__(self, text: str) -> None:
        self.text = text

    def generate_content(self, **_kwargs):
        return type("Response", (), {"text": self.text})()


class FakeClient:
    def __init__(self, text: str) -> None:
        self.models = FakeModels(text)


def interpret_fake(request: OptimizeEnergyRequest, payload: str):
    return GeminiDirectiveInterpreter(
        api_key="test", model="test", client=FakeClient(payload)
    ).interpret(request)


def replay(request, directives, result) -> None:
    validate_solved_plan(
        request,
        directives,
        result.hourly_plan,
        result.total_grid_kwh,
        result.total_cost_bdt,
        result.peak_grid_kwh,
    )


def test_prompt_distinguishes_reduced_by_from_reduced_to_and_fractions() -> None:
    prompt = build_interpretation_prompt(make_request())
    assert '"reduced by 80%"' in prompt and "factor 0.2" in prompt
    assert '"reduced to 80%"' in prompt and "factor 0.8" in prompt
    assert '"reduced by one-third" leaves factor\n2/3' in prompt
    assert '"reduced to one-third" means factor 1/3' in prompt


@pytest.mark.parametrize(
    ("note", "factor"),
    [
        ("Solar is reduced by 80% from 1 PM to 2 PM.", 0.2),
        ("Solar is reduced to 80% from 1 PM to 2 PM.", 0.8),
        ("Solar is reduced by one-third from 1 PM to 2 PM.", 2 / 3),
        ("One-third of solar remains from 1 PM to 2 PM.", 1 / 3),
        ("Solar is reduced by half from 1 PM to 2 PM.", 0.5),
        ("Three quarters of solar remains from 1 PM to 2 PM.", 0.75),
    ],
)
def test_fractional_solar_meanings_pass_structured_guardrails(note: str, factor: float) -> None:
    request = make_request(notes=[note])
    payload = json.dumps(
        {
            "directive_interpretation": [
                {
                    "note_index": 0,
                    "applies": True,
                    "directive_type": "solar_reduction",
                    "structured_adjustment": {"hours": [13], "factor": factor},
                    "explanation": "Normalized solar fraction.",
                }
            ]
        }
    )
    result = interpret_fake(request, payload)
    assert result.directive_interpretation[0].structured_adjustment.factor == pytest.approx(factor)


@pytest.mark.parametrize(
    ("note", "expected_hours"),
    [
        ("Do not charge from 12 AM to 1 AM.", [0]),
        ("Do not charge from 12 PM to 1 PM.", [12]),
        ("Do not charge from 11 AM to 1 PM.", [11, 12]),
        ("Do not charge from 3 PM to 4 PM.", [15]),
    ],
)
def test_clock_boundaries_and_one_hour_intervals(note: str, expected_hours: list[int]) -> None:
    request = make_request(notes=[note])
    payload = json.dumps(
        {
            "directive_interpretation": [
                {
                    "note_index": 0,
                    "applies": True,
                    "directive_type": "no_charge_window",
                    "structured_adjustment": {"hours": expected_hours},
                    "explanation": "End-exclusive time window.",
                }
            ]
        }
    )
    result = interpret_fake(request, payload)
    assert result.directive_interpretation[0].structured_adjustment.hours == expected_hours


@pytest.mark.skip(
    reason=(
        "SPEC AMBIGUITY: the official documents do not explicitly define how the "
        "word 'midnight' is represented as an end boundary in natural-language notes"
    )
)
def test_note_ending_at_midnight_requires_official_definition() -> None:
    """Intentionally not assigned invented hours such as [22, 23]."""


def test_zero_solar_is_valid_and_never_creates_solar_energy() -> None:
    request = make_request(solar=0)
    result = optimize_energy_schedule(request)
    assert all(entry.solar_used_kwh == 0 for entry in result.hourly_plan)
    replay(request, [], result)


@pytest.mark.parametrize(
    ("max_charge", "max_discharge"),
    [(0, 10), (10, 0), (0, 0)],
)
def test_zero_battery_rate_is_respected(max_charge: float, max_discharge: float) -> None:
    tariffs = [1.0] + [100.0] + [1.0] * 22
    request = make_request(
        tariffs=tariffs, max_charge=max_charge, max_discharge=max_discharge
    )
    result = optimize_energy_schedule(request)
    if max_charge == 0:
        assert all(entry.battery_action != "charge" for entry in result.hourly_plan)
    if max_discharge == 0:
        assert all(entry.battery_action != "discharge" for entry in result.hourly_plan)
    replay(request, [], result)


@pytest.mark.parametrize(("initial", "minimum"), [(0, 0), (20, 0)])
def test_initial_battery_exactly_at_a_bound(initial: float, minimum: float) -> None:
    request = make_request(initial=initial, minimum=minimum, capacity=20)
    result = optimize_energy_schedule(request)
    assert result.hourly_plan[-1].battery_energy_after_kwh == pytest.approx(initial)
    replay(request, [], result)


def test_initial_battery_exactly_at_nonzero_minimum() -> None:
    request = make_request(initial=5, minimum=5, capacity=20)
    result = optimize_energy_schedule(request)
    assert min(entry.battery_energy_after_kwh for entry in result.hourly_plan) >= 5
    replay(request, [], result)


def test_constant_tariff_accepts_any_optimal_tie_but_preserves_cost() -> None:
    request = make_request(tariffs=[7.5] * 24, solar=0)
    result = optimize_energy_schedule(request)
    assert result.total_grid_kwh == pytest.approx(240)
    assert result.total_cost_bdt == pytest.approx(1800)
    replay(request, [], result)


def test_huge_tariff_difference_drives_charge_then_discharge() -> None:
    tariffs = [1.0, 1_000_000.0] + [1.0] * 22
    request = make_request(
        tariffs=tariffs, initial=0, minimum=0, capacity=10,
        max_charge=10, max_discharge=10,
    )
    result = optimize_energy_schedule(request)
    assert result.hourly_plan[0].battery_action == "charge"
    assert result.hourly_plan[1].battery_action == "discharge"
    assert result.hourly_plan[1].grid_kwh == pytest.approx(0)
    replay(request, [], result)


def test_grid_cap_requires_precharging() -> None:
    request = make_request(initial=0, capacity=10)
    directives = [directive("max_grid_window", {"hours": [1], "max_grid_kwh": 0})]
    result = optimize_energy_schedule(request, directives)
    assert result.hourly_plan[0].battery_action == "charge"
    assert result.hourly_plan[1].battery_action == "discharge"
    assert result.hourly_plan[1].grid_kwh == pytest.approx(0)
    replay(request, directives, result)


def test_reserve_plus_no_charge_requires_precharging() -> None:
    request = make_request(initial=0, capacity=20)
    directives = [
        directive(
            "minimum_battery_reserve",
            {"hours": [1], "minimum_energy_kwh": 10},
            note_index=0,
        ),
        directive("no_charge_window", {"hours": [1]}, note_index=1),
    ]
    result = optimize_energy_schedule(request, directives)
    assert result.hourly_plan[0].battery_action == "charge"
    assert result.hourly_plan[1].battery_energy_after_kwh >= 10
    replay(request, directives, result)


def test_surplus_solar_is_curtailed_when_it_cannot_be_used_or_exported() -> None:
    request = make_request(
        demand=0, solar=100, initial=0, capacity=20, max_charge=0, max_discharge=0
    )
    result = optimize_energy_schedule(request)
    assert all(entry.solar_used_kwh == 0 for entry in result.hourly_plan)
    assert result.total_grid_kwh == 0
    replay(request, [], result)


def test_irrelevant_note_with_numbers_and_times_can_remain_no_op() -> None:
    request = make_request(
        notes=["The 12 PM seminar has 80 attendees in room 3."],
    )
    payload = json.dumps(
        {
            "directive_interpretation": [
                {
                    "note_index": 0,
                    "applies": False,
                    "directive_type": "no_op",
                    "structured_adjustment": None,
                    "explanation": "Unrelated to today's energy schedule.",
                }
            ]
        }
    )
    result = interpret_fake(request, payload)
    assert result.directive_interpretation[0].directive_type == "no_op"


@pytest.mark.parametrize("malformed", ["{", "not-json", '{"directive_interpretation":'])
def test_malformed_llm_json_is_controlled(malformed: str) -> None:
    with pytest.raises(LLMInterpretationError):
        interpret_fake(make_request(), malformed)


@pytest.mark.parametrize(
    "adjustment",
    [
        {"hours": [1], "factor": -0.01},
        {"hours": [1], "factor": 1.01},
        {"hours": [True], "factor": 0.5},
        {"hours": [1], "factor": float("nan")},
    ],
)
def test_valid_json_with_invalid_semantic_values_is_controlled(adjustment: dict) -> None:
    payload = json.dumps(
        {
            "directive_interpretation": [
                {
                    "note_index": 0,
                    "applies": True,
                    "directive_type": "solar_reduction",
                    "structured_adjustment": adjustment,
                    "explanation": "Invalid semantics.",
                }
            ]
        },
        allow_nan=True,
    )
    with pytest.raises(LLMInterpretationError):
        interpret_fake(make_request(), payload)


def test_repeated_non_solar_directives_combine_as_hard_constraints() -> None:
    request = make_request()
    directives = [
        directive("no_charge_window", {"hours": [1, 2]}, note_index=0),
        directive("no_charge_window", {"hours": [2, 3]}, note_index=1),
        directive(
            "minimum_battery_reserve",
            {"hours": [2], "minimum_energy_kwh": 8},
            note_index=2,
        ),
    ]
    result = optimize_energy_schedule(request, directives)
    assert all(result.hourly_plan[hour].battery_action != "charge" for hour in [1, 2, 3])
    assert result.hourly_plan[2].battery_energy_after_kwh >= 8
    replay(request, directives, result)


def test_multiple_constraints_on_same_hour_are_all_enforced() -> None:
    request = make_request(demand=10, solar=10, initial=10, minimum=0)
    directives = [
        directive("no_charge_window", {"hours": [5]}, note_index=0),
        directive("no_discharge_window", {"hours": [5]}, note_index=1),
        directive("max_grid_window", {"hours": [5], "max_grid_kwh": 0}, note_index=2),
    ]
    result = optimize_energy_schedule(request, directives)
    entry = result.hourly_plan[5]
    assert entry.battery_action == "idle"
    assert entry.grid_kwh == pytest.approx(0)
    assert entry.solar_used_kwh == pytest.approx(10)
    replay(request, directives, result)


def test_overlapping_solar_reductions_are_explicitly_undefined() -> None:
    request = make_request(solar=10)
    directives = [
        directive("solar_reduction", {"hours": [5], "factor": 0.5}, note_index=0),
        directive("solar_reduction", {"hours": [5], "factor": 0.8}, note_index=1),
    ]
    with pytest.raises(DirectiveOverlapError, match="not defined by the official specification"):
        optimize_energy_schedule(request, directives)


def test_values_inside_solver_epsilon_are_cleaned_to_idle_zero() -> None:
    assert _clean(EPSILON / 2) == 0
    assert _clean(-EPSILON / 2) == 0
    assert _clean(EPSILON * 2) != 0
