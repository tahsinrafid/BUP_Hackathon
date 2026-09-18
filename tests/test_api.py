from fastapi.testclient import TestClient

import app.main as main
from app.main import app
from app.schemas import OperatorNoteInterpretationResult

client = TestClient(app)


def valid_request() -> dict:
    return {
        "scenario_id": "TEST-001",
        "operator_notes": ["The cafeteria menu changes tomorrow."],
        "hours": [
            {
                "hour": hour,
                "demand_kwh": 100,
                "solar_kwh": 0,
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


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_optimize_energy_is_validated_before_placeholder() -> None:
    response = client.post("/optimize-energy", json=valid_request())
    assert response.status_code == 501
    assert response.json()["detail"] == "not implemented"


def test_hours_must_be_exactly_zero_through_twenty_three() -> None:
    payload = valid_request()
    payload["hours"][23]["hour"] = 24
    response = client.post("/optimize-energy", json=payload)
    assert response.status_code == 422


def test_hours_must_contain_exactly_24_entries() -> None:
    payload = valid_request()
    payload["hours"] = payload["hours"][:-1]
    response = client.post("/optimize-energy", json=payload)
    assert response.status_code == 422


def test_numeric_values_must_be_finite_and_non_negative() -> None:
    payload = valid_request()
    payload["hours"][0]["demand_kwh"] = -1
    response = client.post("/optimize-energy", json=payload)
    assert response.status_code == 422


def test_operator_notes_must_have_one_to_three_non_empty_strings() -> None:
    payload = valid_request()
    payload["operator_notes"] = ["   "]
    response = client.post("/optimize-energy", json=payload)
    assert response.status_code == 422


def test_battery_relationships_are_validated() -> None:
    payload = valid_request()
    payload["battery"]["initial_energy_kwh"] = 250
    response = client.post("/optimize-energy", json=payload)
    assert response.status_code == 422


def test_development_interpretation_endpoint(monkeypatch) -> None:
    class FakeInterpreter:
        def interpret(self, _request):
            return OperatorNoteInterpretationResult.model_validate(
                {
                    "directive_interpretation": [
                        {
                            "note_index": 0,
                            "applies": False,
                            "directive_type": "no_op",
                            "structured_adjustment": None,
                            "explanation": "No scheduling impact.",
                        }
                    ]
                }
            )

    monkeypatch.setattr(main, "GeminiDirectiveInterpreter", FakeInterpreter)
    response = client.post("/interpret-operator-notes", json=valid_request())
    assert response.status_code == 200
    assert response.json()["directive_interpretation"][0]["directive_type"] == "no_op"
