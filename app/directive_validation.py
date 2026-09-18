"""Deterministic validation for untrusted LLM directive output."""

from __future__ import annotations

import math

from .schemas import (
    DirectiveInterpretation,
    HoursAdjustment,
    MaxGridWindowAdjustment,
    MinimumBatteryReserveAdjustment,
    OperatorNoteInterpretationResult,
    OptimizeEnergyRequest,
    SolarReductionAdjustment,
)


class DirectiveValidationError(ValueError):
    """An LLM-provided directive violates the GridWise contract."""


ALLOWED_DIRECTIVE_TYPES = {
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
}


def _validate_hours(hours: list[int]) -> None:
    if not hours:
        raise DirectiveValidationError("directive hours must not be empty")
    if any(type(hour) is not int for hour in hours):
        raise DirectiveValidationError("directive hours must be integers")
    if any(hour < 0 or hour > 23 for hour in hours):
        raise DirectiveValidationError("directive hours must be between 0 and 23")
    if hours != sorted(hours):
        raise DirectiveValidationError("directive hours must be ascending")
    if len(hours) != len(set(hours)):
        raise DirectiveValidationError("directive hours must not contain duplicates")


def _validate_directive(
    directive: DirectiveInterpretation, request: OptimizeEnergyRequest
) -> None:
    if directive.directive_type not in ALLOWED_DIRECTIVE_TYPES:
        raise DirectiveValidationError("unsupported directive_type")

    if directive.directive_type == "no_op":
        if directive.applies or directive.structured_adjustment is not None:
            raise DirectiveValidationError(
                "no_op requires applies=false and structured_adjustment=null"
            )
        return

    if not directive.applies or directive.structured_adjustment is None:
        raise DirectiveValidationError(
            "non-no_op directives require applies=true and structured_adjustment"
        )

    adjustment = directive.structured_adjustment
    if not isinstance(adjustment, HoursAdjustment):
        raise DirectiveValidationError("directive adjustment must include hours")
    _validate_hours(adjustment.hours)

    if directive.directive_type == "solar_reduction":
        if not isinstance(adjustment, SolarReductionAdjustment):
            raise DirectiveValidationError("solar_reduction requires factor")
        if not math.isfinite(adjustment.factor) or not 0 <= adjustment.factor <= 1:
            raise DirectiveValidationError("solar factor must be finite and between 0 and 1")

    if directive.directive_type == "minimum_battery_reserve":
        if not isinstance(adjustment, MinimumBatteryReserveAdjustment):
            raise DirectiveValidationError(
                "minimum_battery_reserve requires minimum_energy_kwh"
            )
        reserve = adjustment.minimum_energy_kwh
        if not math.isfinite(reserve) or reserve < 0:
            raise DirectiveValidationError("battery reserve must be finite and non-negative")
        if reserve > request.battery.capacity_kwh:
            raise DirectiveValidationError("battery reserve cannot exceed battery capacity")

    if directive.directive_type == "max_grid_window":
        if not isinstance(adjustment, MaxGridWindowAdjustment):
            raise DirectiveValidationError("max_grid_window requires max_grid_kwh")
        if not math.isfinite(adjustment.max_grid_kwh) or adjustment.max_grid_kwh < 0:
            raise DirectiveValidationError("max grid value must be finite and non-negative")


def validate_llm_interpretation(
    result: OperatorNoteInterpretationResult, request: OptimizeEnergyRequest
) -> OperatorNoteInterpretationResult:
    """Validate the complete LLM result against the original input scenario."""
    directives = result.directive_interpretation
    expected_indices = list(range(len(request.operator_notes)))
    actual_indices = [directive.note_index for directive in directives]

    if len(directives) != len(request.operator_notes):
        raise DirectiveValidationError(
            "LLM output must contain exactly one interpretation per operator note"
        )
    if sorted(actual_indices) != expected_indices:
        raise DirectiveValidationError(
            "LLM output must contain every note_index exactly once"
        )
    if actual_indices != expected_indices:
        raise DirectiveValidationError(
            "LLM output note_index values must follow the original note order"
        )

    for directive in directives:
        _validate_directive(directive, request)
    return result
