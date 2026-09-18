"""Full deterministic production-pipeline checks for all public samples."""

from scripts.run_public_samples import load_cases, run_public_samples


def test_all_organizer_public_samples_pass_complete_pipeline() -> None:
    cases = load_cases()
    results = run_public_samples()

    assert len(cases) == 10
    assert len(results) == len(cases)
    assert all(result.passed for result in results), {
        result.case_id: result.diagnostic
        for result in results
        if not result.passed
    }
