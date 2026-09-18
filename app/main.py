"""HTTP entrypoint for the initial GridWise service."""

from fastapi import FastAPI, HTTPException, status

from .interpretation import GeminiDirectiveInterpreter, LLMInterpretationError
from .schemas import OptimizeEnergyRequest, OperatorNoteInterpretationResult

app = FastAPI(
    title="GridWise Energy Optimizer",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/optimize-energy", status_code=status.HTTP_501_NOT_IMPLEMENTED)
def optimize_energy(_: OptimizeEnergyRequest) -> None:
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="not implemented",
    )


@app.post(
    "/interpret-operator-notes",
    response_model=OperatorNoteInterpretationResult,
    tags=["development"],
)
def interpret_operator_notes(
    request: OptimizeEnergyRequest,
) -> OperatorNoteInterpretationResult:
    """Development endpoint for testing the LLM layer before optimization exists."""
    try:
        return GeminiDirectiveInterpreter().interpret(request)
    except LLMInterpretationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="operator-note interpretation is unavailable",
        ) from exc
