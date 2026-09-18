import math

from app.optimizer import optimize_energy_schedule
from app.schemas import OptimizeEnergyRequest


def make_request(
    *,
    demand_by_hour: dict[int, float] | None = None,
    tariff_by_hour: dict[int, float] | None = None,
    capacity: float = 10,
    initial_energy: float = 5,
    minimum_energy: float = 0,
    max_charge: float = 5,
    max_discharge: float = 5,
) -> OptimizeEnergyRequest:
    demand_by_hour = demand_by_hour or {}
    tariff_by_hour = tariff_by_hour or {}
    return OptimizeEnergyRequest.model_validate(
        {
            "scenario_id": "OPTIMIZER-TEST",
            "operator_notes": ["No additional energy constraints today."],
            "hours": [
                {
                    "hour": hour,
                    "demand_kwh": demand_by_hour.get(hour, 0),
                    "solar_kwh": 0,
                    "tariff_bdt_per_kwh": tariff_by_hour.get(hour, 5),
                }
                for hour in range(24)
            ],
            "battery": {
                "capacity_kwh": capacity,
                "initial_energy_kwh": initial_energy,
                "minimum_energy_kwh": minimum_energy,
                "max_charge_kwh_per_hour": max_charge,
                "max_discharge_kwh_per_hour": max_discharge,
            },
        }
    )


def battery_flow(entry) -> float:
    if entry.battery_action == "charge":
        return entry.battery_kwh
    if entry.battery_action == "discharge":
        return -entry.battery_kwh
    return 0.0


def test_low_early_tariff_charges_before_high_tariff_discharge() -> None:
    request = make_request(
        demand_by_hour={1: 15},
        tariff_by_hour={0: 1, 1: 10, 23: 1},
        max_discharge=10,
    )

    result = optimize_energy_schedule(request)

    assert result.hourly_plan[0].battery_action == "charge"
    assert result.hourly_plan[0].battery_kwh == 5
    assert result.hourly_plan[1].battery_action == "discharge"
    assert result.hourly_plan[1].battery_kwh == 10
    assert result.hourly_plan[23].battery_action == "charge"
    assert result.hourly_plan[23].battery_kwh == 5


def test_battery_energy_never_exceeds_capacity_or_drops_below_minimum() -> None:
    request = make_request(
        demand_by_hour={6: 20},
        tariff_by_hour={5: 1, 6: 20, 23: 1},
        capacity=8,
        initial_energy=4,
        minimum_energy=3,
        max_charge=4,
        max_discharge=4,
    )

    result = optimize_energy_schedule(request)

    for entry in result.hourly_plan:
        assert request.battery.minimum_energy_kwh <= entry.battery_energy_after_kwh
        assert entry.battery_energy_after_kwh <= request.battery.capacity_kwh


def test_final_battery_energy_equals_initial_energy() -> None:
    request = make_request(
        demand_by_hour={10: 10}, tariff_by_hour={9: 1, 10: 15, 23: 1}
    )

    result = optimize_energy_schedule(request)

    assert math.isclose(
        result.hourly_plan[23].battery_energy_after_kwh,
        request.battery.initial_energy_kwh,
        abs_tol=1e-7,
    )


def test_energy_balance_and_battery_state_hold_for_every_hour() -> None:
    request = make_request(
        demand_by_hour={0: 2, 4: 8, 18: 12},
        tariff_by_hour={3: 1, 4: 12, 17: 1, 18: 20, 23: 1},
        capacity=12,
        initial_energy=6,
        minimum_energy=2,
        max_charge=4,
        max_discharge=4,
    )

    result = optimize_energy_schedule(request)
    previous_energy = request.battery.initial_energy_kwh
    for entry in result.hourly_plan:
        flow = battery_flow(entry)
        hour = request.hours[entry.hour]
        assert math.isclose(
            entry.grid_kwh + entry.solar_used_kwh,
            hour.demand_kwh + flow,
            abs_tol=1e-7,
        )
        assert math.isclose(
            entry.battery_energy_after_kwh,
            previous_energy + flow,
            abs_tol=1e-7,
        )
        previous_energy = entry.battery_energy_after_kwh
