import math

import pytest

from app.optimizer import OptimizationResult, optimize_energy_schedule
from app.replay_validator import PlanValidationError, validate_solved_plan
from app.schemas import DirectiveInterpretation, OptimizeEnergyRequest


def make_request(
    *,
    demand_by_hour: dict[int, float] | None = None,
    solar_by_hour: dict[int, float] | None = None,
    tariff_by_hour: dict[int, float] | None = None,
    initial_energy: float = 5,
) -> OptimizeEnergyRequest:
    demand_by_hour = demand_by_hour or {}
    solar_by_hour = solar_by_hour or {}
    tariff_by_hour = tariff_by_hour or {}
    return OptimizeEnergyRequest.model_validate(
        {
            "scenario_id": "REPLAY-TEST",
            "operator_notes": ["No additional constraint."],
            "hours": [
                {
                    "hour": hour,
                    "demand_kwh": demand_by_hour.get(hour, 0),
                    "solar_kwh": solar_by_hour.get(hour, 0),
                    "tariff_bdt_per_kwh": tariff_by_hour.get(hour, 5),
                }
                for hour in range(24)
            ],
            "battery": {
                "capacity_kwh": 10,
                "initial_energy_kwh": initial_energy,
                "minimum_energy_kwh": 0,
                "max_charge_kwh_per_hour": 5,
                "max_discharge_kwh_per_hour": 5,
            },
        }
    )


def make_directive(
    directive_type: str, adjustment: dict | None, *, note_index: int = 0
) -> DirectiveInterpretation:
    return DirectiveInterpretation.model_validate(
        {
            "note_index": note_index,
            "applies": directive_type != "no_op",
            "directive_type": directive_type,
            "structured_adjustment": adjustment,
            "explanation": "Test directive.",
        }
    )


def replay(request, directives, result) -> None:
    validate_solved_plan(
        request,
        directives,
        result.hourly_plan,
        result.total_grid_kwh,
        result.total_cost_bdt,
        result.peak_grid_kwh,
    )


def changed_result(result, hour: int, **changes) -> OptimizationResult:
    plan = list(result.hourly_plan)
    plan[hour] = plan[hour].model_copy(update=changes)
    return OptimizationResult(
        hourly_plan=plan,
        total_grid_kwh=result.total_grid_kwh,
        total_cost_bdt=result.total_cost_bdt,
        peak_grid_kwh=result.peak_grid_kwh,
    )


def test_optimizer_result_passes_independent_replay() -> None:
    request = make_request(demand_by_hour={18: 5}, tariff_by_hour={18: 20})
    result = optimize_energy_schedule(request)

    replay(request, [], result)


def test_replay_recalculates_totals() -> None:
    request = make_request(demand_by_hour={18: 5}, tariff_by_hour={18: 20})
    result = optimize_energy_schedule(request)
    totals = validate_solved_plan(
        request,
        [],
        result.hourly_plan,
        result.total_grid_kwh,
        result.total_cost_bdt,
        result.peak_grid_kwh,
    )

    assert math.isclose(totals.total_grid_kwh, sum(x.grid_kwh for x in result.hourly_plan))
    assert math.isclose(
        totals.total_cost_bdt,
        sum(x.grid_kwh * request.hours[x.hour].tariff_bdt_per_kwh for x in result.hourly_plan),
    )


def test_replay_catches_wrong_battery_state() -> None:
    request = make_request(demand_by_hour={1: 5})
    result = optimize_energy_schedule(request)
    bad = changed_result(
        result,
        1,
        battery_energy_after_kwh=result.hourly_plan[1].battery_energy_after_kwh + 1,
    )

    with pytest.raises(PlanValidationError, match="battery state transition"):
        replay(request, [], bad)


def test_replay_catches_excess_solar_use() -> None:
    request = make_request(demand_by_hour={10: 5}, solar_by_hour={10: 5}, initial_energy=0)
    result = optimize_energy_schedule(request)
    bad = changed_result(result, 10, solar_used_kwh=6)

    with pytest.raises(PlanValidationError, match="exceeds effective solar"):
        replay(request, [], bad)


def test_replay_catches_wrong_total_cost() -> None:
    request = make_request(demand_by_hour={18: 5}, tariff_by_hour={18: 20})
    result = optimize_energy_schedule(request)
    bad = OptimizationResult(
        hourly_plan=result.hourly_plan,
        total_grid_kwh=result.total_grid_kwh,
        total_cost_bdt=result.total_cost_bdt + 1,
        peak_grid_kwh=result.peak_grid_kwh,
    )

    with pytest.raises(PlanValidationError, match="total_cost_bdt"):
        replay(request, [], bad)


def test_replay_catches_grid_cap_violation() -> None:
    request = make_request(demand_by_hour={18: 10}, tariff_by_hour={18: 20})
    directives = [make_directive("max_grid_window", {"hours": [18], "max_grid_kwh": 5})]
    result = optimize_energy_schedule(request, directives)
    bad = changed_result(result, 18, grid_kwh=6)

    with pytest.raises(PlanValidationError, match="max_grid_window"):
        replay(request, directives, bad)


def test_replay_catches_reserve_violation() -> None:
    request = make_request(demand_by_hour={18: 5}, tariff_by_hour={18: 20})
    directives = [
        make_directive(
            "minimum_battery_reserve",
            {"hours": [18], "minimum_energy_kwh": 5},
        )
    ]
    result = optimize_energy_schedule(request, directives)
    bad = changed_result(result, 18, battery_energy_after_kwh=4)

    with pytest.raises(PlanValidationError, match="minimum_battery_reserve"):
        replay(request, directives, bad)


def test_replay_catches_charging_during_no_charge_window() -> None:
    request = make_request(initial_energy=0)
    directives = [make_directive("no_charge_window", {"hours": [0]})]
    result = optimize_energy_schedule(request, directives)
    bad = changed_result(result, 0, battery_action="charge", battery_kwh=1, battery_energy_after_kwh=1)

    with pytest.raises(PlanValidationError, match="no_charge_window"):
        replay(request, directives, bad)


def test_replay_catches_discharging_during_no_discharge_window() -> None:
    request = make_request(initial_energy=5)
    directives = [make_directive("no_discharge_window", {"hours": [0]})]
    result = optimize_energy_schedule(request, directives)
    bad = changed_result(result, 0, battery_action="discharge", battery_kwh=1, battery_energy_after_kwh=4)

    with pytest.raises(PlanValidationError, match="no_discharge_window"):
        replay(request, directives, bad)


def test_replay_catches_final_battery_mismatch() -> None:
    request = make_request(demand_by_hour={23: 5}, initial_energy=5)
    result = optimize_energy_schedule(request)
    previous_energy = result.hourly_plan[22].battery_energy_after_kwh
    bad = changed_result(
        result,
        23,
        grid_kwh=4,
        battery_action="discharge",
        battery_kwh=1,
        battery_energy_after_kwh=previous_energy - 1,
    )

    with pytest.raises(PlanValidationError, match="end-of-day battery energy"):
        replay(request, [], bad)
