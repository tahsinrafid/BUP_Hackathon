"""Pydantic models for the GridWise API contract."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    field_validator,
    model_validator,
)


def _finite_non_negative(value: float, field_name: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


class HourInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hour: StrictInt
    demand_kwh: float
    solar_kwh: float
    tariff_bdt_per_kwh: float

    @field_validator("demand_kwh", "solar_kwh", "tariff_bdt_per_kwh")
    @classmethod
    def validate_numeric_values(cls, value: float, info: Any) -> float:
        return _finite_non_negative(value, info.field_name)


class BatteryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capacity_kwh: float
    initial_energy_kwh: float
    minimum_energy_kwh: float
    max_charge_kwh_per_hour: float
    max_discharge_kwh_per_hour: float

    @field_validator(
        "capacity_kwh",
        "initial_energy_kwh",
        "minimum_energy_kwh",
        "max_charge_kwh_per_hour",
        "max_discharge_kwh_per_hour",
    )
    @classmethod
    def validate_numeric_values(cls, value: float, info: Any) -> float:
        return _finite_non_negative(value, info.field_name)

    @model_validator(mode="after")
    def validate_relationships(self) -> "BatteryInput":
        if self.capacity_kwh <= 0:
            raise ValueError("capacity_kwh must be greater than zero")
        if self.minimum_energy_kwh > self.capacity_kwh:
            raise ValueError("minimum_energy_kwh cannot exceed capacity_kwh")
        if self.initial_energy_kwh > self.capacity_kwh:
            raise ValueError("initial_energy_kwh cannot exceed capacity_kwh")
        if self.initial_energy_kwh < self.minimum_energy_kwh:
            raise ValueError("initial_energy_kwh cannot be below minimum_energy_kwh")
        return self


class OptimizeEnergyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str = Field(min_length=1)
    operator_notes: list[str] = Field(min_length=1, max_length=3)
    hours: list[HourInput] = Field(min_length=24, max_length=24)
    battery: BatteryInput

    @field_validator("operator_notes")
    @classmethod
    def validate_operator_notes(cls, notes: list[str]) -> list[str]:
        if any(not note.strip() for note in notes):
            raise ValueError("operator_notes must contain non-empty strings")
        return notes

    @model_validator(mode="after")
    def validate_hours(self) -> "OptimizeEnergyRequest":
        actual_hours = [entry.hour for entry in self.hours]
        if sorted(actual_hours) != list(range(24)):
            raise ValueError("hours must contain every integer from 0 through 23 exactly once")
        self.hours.sort(key=lambda entry: entry.hour)
        return self


DirectiveType = Literal[
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
]
BatteryAction = Literal["charge", "discharge", "idle"]


class HoursAdjustment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hours: list[StrictInt] = Field(min_length=1)

    @field_validator("hours")
    @classmethod
    def validate_hours(cls, hours: list[int]) -> list[int]:
        if hours != sorted(set(hours)) or any(hour < 0 or hour > 23 for hour in hours):
            raise ValueError(
                "directive hours must be unique ascending integers from 0 through 23"
            )
        return hours


class SolarReductionAdjustment(HoursAdjustment):
    factor: float

    @field_validator("factor")
    @classmethod
    def validate_factor(cls, factor: float) -> float:
        if not math.isfinite(factor) or not 0 <= factor <= 1:
            raise ValueError("factor must be finite and between zero and one")
        return factor


class MinimumBatteryReserveAdjustment(HoursAdjustment):
    minimum_energy_kwh: float

    @field_validator("minimum_energy_kwh")
    @classmethod
    def validate_minimum_energy(cls, value: float) -> float:
        return _finite_non_negative(value, "minimum_energy_kwh")


class MaxGridWindowAdjustment(HoursAdjustment):
    max_grid_kwh: float

    @field_validator("max_grid_kwh")
    @classmethod
    def validate_max_grid(cls, value: float) -> float:
        return _finite_non_negative(value, "max_grid_kwh")


StructuredAdjustment = (
    SolarReductionAdjustment
    | MinimumBatteryReserveAdjustment
    | MaxGridWindowAdjustment
    | HoursAdjustment
)


class DirectiveInterpretation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    note_index: StrictInt = Field(ge=0)
    applies: bool
    directive_type: DirectiveType
    structured_adjustment: StructuredAdjustment | None
    explanation: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_directive_shape(self) -> "DirectiveInterpretation":
        if self.directive_type == "no_op":
            if self.applies or self.structured_adjustment is not None:
                raise ValueError(
                    "no_op requires applies=false and structured_adjustment=null"
                )
            return self

        if not self.applies or self.structured_adjustment is None:
            raise ValueError(
                "non-no_op directives require applies=true and an adjustment"
            )

        adjustment_models: dict[str, type[BaseModel]] = {
            "solar_reduction": SolarReductionAdjustment,
            "minimum_battery_reserve": MinimumBatteryReserveAdjustment,
            "no_charge_window": HoursAdjustment,
            "no_discharge_window": HoursAdjustment,
            "max_grid_window": MaxGridWindowAdjustment,
        }
        if type(self.structured_adjustment) is not adjustment_models[self.directive_type]:
            raise ValueError("structured_adjustment does not match directive_type")
        return self


class OperatorNoteInterpretationResult(BaseModel):
    """Structured output returned by the LLM for one complete note list."""

    model_config = ConfigDict(extra="forbid")

    directive_interpretation: list[DirectiveInterpretation]


class HourlyPlanEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hour: int
    grid_kwh: float
    solar_used_kwh: float
    battery_action: BatteryAction
    battery_kwh: float
    battery_energy_after_kwh: float


class OptimizeEnergyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    directive_interpretation: list[DirectiveInterpretation]
    hourly_plan: list[HourlyPlanEntry]
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float
    plan_summary: str
