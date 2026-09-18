"""Deterministic linear-programming energy optimizer for GridWise."""

from __future__ import annotations

from dataclasses import dataclass

import pulp

from .schemas import (
    DirectiveInterpretation,
    HourlyPlanEntry,
    MaxGridWindowAdjustment,
    MinimumBatteryReserveAdjustment,
    OptimizeEnergyRequest,
    SolarReductionAdjustment,
)


EPSILON = 1e-7


class OptimizationError(RuntimeError):
    """The deterministic optimization problem could not be solved."""


class DirectiveOverlapError(OptimizationError):
    """An organizer-undefined directive combination was supplied."""


@dataclass(frozen=True)
class OptimizationResult:
    """The solved schedule and recalculated summary values."""

    hourly_plan: list[HourlyPlanEntry]
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float


def _effective_constraints(
    request: OptimizeEnergyRequest,
    directives: list[DirectiveInterpretation],
) -> tuple[list[float], list[float], list[float], set[int], set[int]]:
    """Compile already-validated directives into deterministic hourly limits."""
    effective_solar = [hour.solar_kwh for hour in request.hours]
    minimum_energy = [request.battery.minimum_energy_kwh] * 24
    grid_cap = [float("inf")] * 24
    no_charge_hours: set[int] = set()
    no_discharge_hours: set[int] = set()
    solar_reduction_hours: set[int] = set()

    for directive in directives:
        if not directive.applies or directive.structured_adjustment is None:
            continue
        adjustment = directive.structured_adjustment

        if directive.directive_type == "solar_reduction":
            assert isinstance(adjustment, SolarReductionAdjustment)
            for hour in adjustment.hours:
                if hour in solar_reduction_hours:
                    raise DirectiveOverlapError(
                        "overlapping solar_reduction directives are not defined "
                        "by the official specification"
                    )
                solar_reduction_hours.add(hour)
                effective_solar[hour] = request.hours[hour].solar_kwh * adjustment.factor
        elif directive.directive_type == "minimum_battery_reserve":
            assert isinstance(adjustment, MinimumBatteryReserveAdjustment)
            for hour in adjustment.hours:
                minimum_energy[hour] = max(
                    minimum_energy[hour], adjustment.minimum_energy_kwh
                )
        elif directive.directive_type == "no_charge_window":
            no_charge_hours.update(adjustment.hours)
        elif directive.directive_type == "no_discharge_window":
            no_discharge_hours.update(adjustment.hours)
        elif directive.directive_type == "max_grid_window":
            assert isinstance(adjustment, MaxGridWindowAdjustment)
            for hour in adjustment.hours:
                grid_cap[hour] = min(grid_cap[hour], adjustment.max_grid_kwh)

    return effective_solar, minimum_energy, grid_cap, no_charge_hours, no_discharge_hours


def _clean(value: float) -> float:
    """Remove solver noise while preserving values well inside official tolerance."""
    return 0.0 if abs(value) <= EPSILON else round(value, 9)


def optimize_energy_schedule(
    request: OptimizeEnergyRequest,
    directives: list[DirectiveInterpretation] | None = None,
) -> OptimizationResult:
    """Minimize grid cost using only deterministic data and constraints.

    ``battery_flow > 0`` charges the battery; ``battery_flow < 0`` discharges it.
    Directives are optional because the optimizer can solve a base scenario on its own.
    When supplied, they must already have passed the interpretation guardrails.
    """
    directives = directives or []
    battery = request.battery
    effective_solar, minimum_energy, grid_cap, no_charge, no_discharge = (
        _effective_constraints(request, directives)
    )

    model = pulp.LpProblem("gridwise_energy_schedule", pulp.LpMinimize)
    hours = range(24)
    grid = pulp.LpVariable.dicts("grid", hours, lowBound=0)
    solar_used = pulp.LpVariable.dicts("solar_used", hours, lowBound=0)
    battery_flow = pulp.LpVariable.dicts(
        "battery_flow",
        hours,
        lowBound=-battery.max_discharge_kwh_per_hour,
        upBound=battery.max_charge_kwh_per_hour,
    )
    battery_energy_after = pulp.LpVariable.dicts(
        "battery_energy_after",
        hours,
        lowBound=battery.minimum_energy_kwh,
        upBound=battery.capacity_kwh,
    )

    model += pulp.lpSum(grid[hour] * request.hours[hour].tariff_bdt_per_kwh for hour in hours)

    for hour in hours:
        # grid + solar = demand + positive charge flow (or less demand when discharging)
        model += (
            grid[hour] + solar_used[hour]
            == request.hours[hour].demand_kwh + battery_flow[hour]
        )
        model += solar_used[hour] <= effective_solar[hour]
        model += battery_energy_after[hour] >= minimum_energy[hour]
        if hour == 0:
            model += battery_energy_after[hour] == battery.initial_energy_kwh + battery_flow[hour]
        else:
            model += (
                battery_energy_after[hour]
                == battery_energy_after[hour - 1] + battery_flow[hour]
            )
        if grid_cap[hour] != float("inf"):
            model += grid[hour] <= grid_cap[hour]
        if hour in no_charge:
            model += battery_flow[hour] <= 0
        if hour in no_discharge:
            model += battery_flow[hour] >= 0

    model += battery_energy_after[23] == battery.initial_energy_kwh

    status = model.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise OptimizationError(f"optimization failed with status {pulp.LpStatus[status]}")

    plan: list[HourlyPlanEntry] = []
    for hour in hours:
        flow = _clean(pulp.value(battery_flow[hour]))
        if flow > EPSILON:
            action, magnitude = "charge", flow
        elif flow < -EPSILON:
            action, magnitude = "discharge", -flow
        else:
            action, magnitude = "idle", 0.0
        plan.append(
            HourlyPlanEntry(
                hour=hour,
                grid_kwh=_clean(pulp.value(grid[hour])),
                solar_used_kwh=_clean(pulp.value(solar_used[hour])),
                battery_action=action,
                battery_kwh=_clean(magnitude),
                battery_energy_after_kwh=_clean(pulp.value(battery_energy_after[hour])),
            )
        )

    total_grid = _clean(sum(entry.grid_kwh for entry in plan))
    total_cost = _clean(
        sum(
            entry.grid_kwh * request.hours[entry.hour].tariff_bdt_per_kwh
            for entry in plan
        )
    )
    peak_grid = _clean(max(entry.grid_kwh for entry in plan))
    result = OptimizationResult(
        hourly_plan=plan,
        total_grid_kwh=total_grid,
        total_cost_bdt=total_cost,
        peak_grid_kwh=peak_grid,
    )
    # Keep the post-solve check separate from model construction so the returned
    # schedule is independently replayed before any caller can treat it as valid.
    from .replay_validator import validate_solved_plan

    validate_solved_plan(
        request,
        directives,
        result.hourly_plan,
        result.total_grid_kwh,
        result.total_cost_bdt,
        result.peak_grid_kwh,
    )
    return result
