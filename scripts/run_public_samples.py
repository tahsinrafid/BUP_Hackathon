"""Run every organizer public sample through the production API pipeline.

The default mode is deterministic: the organizer's expected structured
directives stand in for the external Gemini response.  Request validation,
directive guardrails, optimization, response serialization, and replay
validation all remain production code paths.

Use ``--live-llm`` to include the real configured Gemini interpretation layer.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.directive_validation import validate_llm_interpretation
from app.main import app
from app.replay_validator import validate_solved_plan
from app.schemas import (
    OperatorNoteInterpretationResult,
    OptimizeEnergyRequest,
    OptimizeEnergyResponse,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE_FILE = ROOT / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json"
COST_TOLERANCE = 0.01


@dataclass
class SampleResult:
    case_id: str
    passed: bool
    our_cost: float | None
    reference_cost: float | None
    difference: float | None
    ratio: float | None
    diagnostic: str = ""


class OrganizerReferenceInterpreter:
    """Test double at the external-provider boundary, not the pipeline boundary."""

    model = "organizer-reference"
    expected_by_scenario: dict[str, list[dict[str, Any]]] = {}

    def interpret(
        self, request: OptimizeEnergyRequest
    ) -> OperatorNoteInterpretationResult:
        raw = {
            "directive_interpretation": self.expected_by_scenario[request.scenario_id]
        }
        parsed = OperatorNoteInterpretationResult.model_validate(raw)
        return validate_llm_interpretation(parsed, request)


def load_cases(path: Path = DEFAULT_SAMPLE_FILE) -> list[dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"No public sample cases found in {path}")
    return cases


def structured_meaning(directives: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare only the organizer's machine-checkable interpretation fields."""
    return [
        {
            "note_index": item["note_index"],
            "applies": item["applies"],
            "directive_type": item["directive_type"],
            "structured_adjustment": item["structured_adjustment"],
        }
        for item in directives
    ]


def _run_case(client: TestClient, case: dict[str, Any]) -> SampleResult:
    case_id = case["id"]
    expected = case["expected_output"]
    reference_cost = expected.get("total_cost_bdt")
    try:
        request = OptimizeEnergyRequest.model_validate(case["input"])
        http_response = client.post("/optimize-energy", json=case["input"])
        if http_response.status_code != 200:
            return SampleResult(
                case_id, False, None, reference_cost, None, None,
                f"HTTP {http_response.status_code}: {http_response.text}",
            )

        body = http_response.json()
        response = OptimizeEnergyResponse.model_validate(body)
        expected_meaning = structured_meaning(expected["directive_interpretation"])
        actual_meaning = structured_meaning(body["directive_interpretation"])
        if actual_meaning != expected_meaning:
            return SampleResult(
                case_id, False, response.total_cost_bdt, reference_cost, None, None,
                f"directive mismatch: expected {expected_meaning}; got {actual_meaning}",
            )

        if response.scenario_id != request.scenario_id:
            raise AssertionError("scenario_id was not preserved")
        if len(response.hourly_plan) != 24:
            raise AssertionError("hourly_plan does not contain exactly 24 entries")

        # This independent replay checks ordered hours, energy balance, battery
        # transitions/rates/bounds, every directive, final state, and all totals.
        validate_solved_plan(
            request,
            response.directive_interpretation,
            response.hourly_plan,
            response.total_grid_kwh,
            response.total_cost_bdt,
            response.peak_grid_kwh,
        )

        difference = None
        ratio = None
        cost_matches = True
        if reference_cost is not None:
            difference = response.total_cost_bdt - float(reference_cost)
            ratio = (
                response.total_cost_bdt / float(reference_cost)
                if float(reference_cost) != 0
                else (1.0 if response.total_cost_bdt == 0 else math.inf)
            )
            cost_matches = abs(difference) <= COST_TOLERANCE

        return SampleResult(
            case_id=case_id,
            passed=cost_matches,
            our_cost=response.total_cost_bdt,
            reference_cost=reference_cost,
            difference=difference,
            ratio=ratio,
            diagnostic="" if cost_matches else "optimal cost differs from organizer reference",
        )
    except Exception as exc:  # runner must report all cases, not stop at the first one
        return SampleResult(
            case_id, False, None, reference_cost, None, None,
            f"{type(exc).__name__}: {exc}",
        )


def run_public_samples(
    *, live_llm: bool = False, sample_file: Path = DEFAULT_SAMPLE_FILE
) -> list[SampleResult]:
    cases = load_cases(sample_file)
    client = TestClient(app)
    if live_llm:
        return [_run_case(client, case) for case in cases]

    OrganizerReferenceInterpreter.expected_by_scenario = {
        case["input"]["scenario_id"]: case["expected_output"][
            "directive_interpretation"
        ]
        for case in cases
    }
    with patch(
        "app.main.GeminiDirectiveInterpreter", OrganizerReferenceInterpreter
    ):
        return [_run_case(client, case) for case in cases]


def _number(value: float | None, decimals: int = 2) -> str:
    return "-" if value is None else f"{value:.{decimals}f}"


def print_results(results: list[SampleResult], *, live_llm: bool) -> None:
    mode = "LIVE LLM" if live_llm else "DETERMINISTIC REFERENCE-DIRECTIVE"
    print(f"Public sample runner mode: {mode}")
    headers = ("Case", "Result", "Our cost", "Reference", "Difference", "Ratio")
    rows = [
        (
            result.case_id,
            "PASS" if result.passed else "FAIL",
            _number(result.our_cost),
            _number(result.reference_cost),
            _number(result.difference),
            _number(result.ratio, 6),
        )
        for result in results
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))

    failures = [result for result in results if not result.passed]
    if failures:
        print("\nDiagnostics:")
        for failure in failures:
            print(f"- {failure.case_id}: {failure.diagnostic}")
    print(f"\nSummary: {len(results) - len(failures)}/{len(results)} passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live-llm",
        action="store_true",
        help="Use the configured Gemini model instead of organizer reference directives.",
    )
    parser.add_argument(
        "--sample-file",
        type=Path,
        default=DEFAULT_SAMPLE_FILE,
        help="Path to the organizer public sample JSON file.",
    )
    args = parser.parse_args()
    results = run_public_samples(live_llm=args.live_llm, sample_file=args.sample_file)
    print_results(results, live_llm=args.live_llm)
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
