"""Independent deterministic replay validation for solved GridWise plans."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from .schemas import (
    DirectiveInterpretation,
    HourlyPlanEntry,
    MaxGridWindowAdjustment,
    MinimumBatteryReserveAdjustment,
    OptimizeEnergyRequest,
    SolarReductionAdjustment,
)


logger = logging.getLogger(__name__)
TOLERANCE = 0.01


class PlanValidationError(RuntimeError):
    """A returned schedule fails independent deterministic replay."""


@dataclass(frozen=True)
class ReplayedTotals:
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float


def _fail(scenario_id: str, rule: str, hour: int | None = None) -> None:
    location = f" at hour {hour}" if hour is not None else ""
    message = f"plan replay failed for scenario {scenario_id}{location}: {rule}"
    logger.error(message)
    raise PlanValidationError(message)


def _directive_limits(
    request: OptimizeEnergyRequest,
    directives: list[DirectiveInterpretation],
) -> tuple[list[float], list[float], list[float], set[int], set[int]]:
    """Recompute directive effects independently of the optimizer implementation."""
    effective_solar = [hour.solar_kwh for hour in request.hours]
    active_minimum = [request.battery.minimum_energy_kwh] * 24
    active_grid_cap = [float("inf")] * 24
    no_charge_hours: set[int] = set()
    no_discharge_hours: set[int] = set()
    solar_reduction_hours: set[int] = set()

    for directive in directives:
        if not directive.applies or directive.structured_adjustment is None:
            continue
        adjustment = directive.structured_adjustment
        if directive.directive_type == "solar_reduction":
            if not isinstance(adjustment, SolarReductionAdjustment):
                _fail(request.scenario_id, "invalid solar_reduction adjustment")
            for hour in adjustment.hours:
                if hour in solar_reduction_hours:
                    _fail(request.scenario_id, "undefined overlapping solar_reduction directives", hour)
                solar_reduction_hours.add(hour)
                effective_solar[hour] = request.hours[hour].solar_kwh * adjustment.factor
        elif directive.directive_type == "minimum_battery_reserve":
            if not isinstance(adjustment, MinimumBatteryReserveAdjustment):
                _fail(request.scenario_id, "invalid minimum_battery_reserve adjustment")
            for hour in adjustment.hours:
                active_minimum[hour] = max(
                    active_minimum[hour], adjustment.minimum_energy_kwh
                )
        elif directive.directive_type == "no_charge_window":
            no_charge_hours.update(adjustment.hours)
        elif directive.directive_type == "no_discharge_window":
            no_discharge_hours.update(adjustment.hours)
        elif directive.directive_type == "max_grid_window":
            if not isinstance(adjustment, MaxGridWindowAdjustment):
                _fail(request.scenario_id, "invalid max_grid_window adjustment")
            for hour in adjustment.hours:
                active_grid_cap[hour] = min(active_grid_cap[hour], adjustment.max_grid_kwh)

    return active_minimum, active_grid_cap, effective_solar, no_charge_hours, no_discharge_hours


def _is_close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0, abs_tol=TOLERANCE)


def validate_solved_plan(
    request: OptimizeEnergyRequest,
    directives: list[DirectiveInterpretation],
    hourly_plan: list[HourlyPlanEntry],
    total_grid_kwh: float,
    total_cost_bdt: float,
    peak_grid_kwh: float,
) -> ReplayedTotals:
    """Replay a complete result and verify all physical and directive constraints."""
    actual_hours = [entry.hour for entry in hourly_plan]
    if actual_hours != list(range(24)):
        _fail(request.scenario_id, "hourly_plan must contain exactly ordered hours 0 through 23")

    active_minimum, grid_caps, effective_solar, no_charge, no_discharge = _directive_limits(
        request, directives
    )
    previous_energy = request.battery.initial_energy_kwh
    recalculated_grid = 0.0
    recalculated_cost = 0.0
    recalculated_peak = 0.0

    for entry in hourly_plan:
        hour = entry.hour
        values = {
            "grid_kwh": entry.grid_kwh,
            "solar_used_kwh": entry.solar_used_kwh,
            "battery_kwh": entry.battery_kwh,
            "battery_energy_after_kwh": entry.battery_energy_after_kwh,
        }
        if any(not math.isfinite(value) for value in values.values()):
            _fail(request.scenario_id, "plan values must be finite", hour)
        if entry.grid_kwh < -TOLERANCE:
            _fail(request.scenario_id, "grid_kwh must be non-negative", hour)
        if entry.solar_used_kwh < -TOLERANCE:
            _fail(request.scenario_id, "solar_used_kwh must be non-negative", hour)
        if entry.battery_kwh < -TOLERANCE:
            _fail(request.scenario_id, "battery_kwh must be non-negative", hour)
        if entry.solar_used_kwh > effective_solar[hour] + TOLERANCE:
            _fail(request.scenario_id, "solar_used_kwh exceeds effective solar", hour)

        if entry.battery_action == "charge":
            if entry.battery_kwh <= TOLERANCE:
                _fail(request.scenario_id, "charge action requires positive battery_kwh", hour)
            flow = entry.battery_kwh
        elif entry.battery_action == "discharge":
            if entry.battery_kwh <= TOLERANCE:
                _fail(request.scenario_id, "discharge action requires positive battery_kwh", hour)
            flow = -entry.battery_kwh
        elif entry.battery_action == "idle":
            if not _is_close(entry.battery_kwh, 0):
                _fail(request.scenario_id, "idle action requires battery_kwh=0", hour)
            flow = 0.0
        else:
            _fail(request.scenario_id, "unsupported battery_action", hour)

        if entry.battery_kwh > request.battery.max_charge_kwh_per_hour + TOLERANCE and flow > 0:
            _fail(request.scenario_id, "charge rate limit exceeded", hour)
        if (
            entry.battery_kwh > request.battery.max_discharge_kwh_per_hour + TOLERANCE
            and flow < 0
        ):
            _fail(request.scenario_id, "discharge rate limit exceeded", hour)
        if hour in no_charge and flow > TOLERANCE:
            _fail(request.scenario_id, "no_charge_window violated", hour)
        if hour in no_discharge and flow < -TOLERANCE:
            _fail(request.scenario_id, "no_discharge_window violated", hour)
        if entry.grid_kwh > grid_caps[hour] + TOLERANCE:
            _fail(request.scenario_id, "max_grid_window violated", hour)
        if entry.battery_energy_after_kwh < request.battery.minimum_energy_kwh - TOLERANCE:
            _fail(request.scenario_id, "base battery minimum violated", hour)
        if entry.battery_energy_after_kwh > request.battery.capacity_kwh + TOLERANCE:
            _fail(request.scenario_id, "battery capacity exceeded", hour)
        if entry.battery_energy_after_kwh < active_minimum[hour] - TOLERANCE:
            _fail(request.scenario_id, "minimum_battery_reserve violated", hour)
        if not _is_close(entry.battery_energy_after_kwh, previous_energy + flow):
            _fail(request.scenario_id, "battery state transition is incorrect", hour)

        demand = request.hours[hour].demand_kwh
        if not _is_close(entry.grid_kwh + entry.solar_used_kwh, demand + flow):
            _fail(request.scenario_id, "energy balance is incorrect", hour)

        previous_energy = entry.battery_energy_after_kwh
        recalculated_grid += entry.grid_kwh
        recalculated_cost += entry.grid_kwh * request.hours[hour].tariff_bdt_per_kwh
        recalculated_peak = max(recalculated_peak, entry.grid_kwh)

    if not _is_close(hourly_plan[23].battery_energy_after_kwh, request.battery.initial_energy_kwh):
        _fail(request.scenario_id, "end-of-day battery energy does not equal initial energy", 23)

    if not _is_close(total_grid_kwh, recalculated_grid):
        _fail(request.scenario_id, "reported total_grid_kwh does not match replay")
    if not _is_close(total_cost_bdt, recalculated_cost):
        _fail(request.scenario_id, "reported total_cost_bdt does not match replay")
    if not _is_close(peak_grid_kwh, recalculated_peak):
        _fail(request.scenario_id, "reported peak_grid_kwh does not match replay")

    return ReplayedTotals(
        total_grid_kwh=recalculated_grid,
        total_cost_bdt=recalculated_cost,
        peak_grid_kwh=recalculated_peak,
    )
