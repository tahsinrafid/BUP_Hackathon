# GridWise Energy Optimizer

GridWise is a FastAPI service for the BUP CSE Fest 2026 preliminary challenge.
It converts 1-3 natural-language operator notes into a fixed directive schema,
then builds and independently validates a minimum-cost 24-hour energy schedule.

## Architecture

```text
POST /optimize-energy
        |
        v
Pydantic request validation
        |
        v
One Gemini structured-output call for all notes
        |
        v
Deterministic directive guardrails
        |
        v
PuLP/CBC linear optimization
        |
        v
Independent 24-hour replay and total calculation
        |
        v
Pydantic response serialization
```

The LLM only translates operator notes into structured directives. It never
generates the hourly plan, calculates costs, modifies scenario data, or decides
whether a physically invalid result is acceptable.

## Requirements and setup

- Python 3.12 (the Docker image also uses Python 3.12)
- A Gemini API key
- Docker is optional

PowerShell setup:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
```

Edit `.env`:

```dotenv
LLM_API_KEY=your_real_gemini_api_key
LLM_MODEL=gemini-3.5-flash-lite
PORT=8000
```

`LLM_MODEL` selects the Gemini model. The current tested selection is
`gemini-3.5-flash-lite`; model availability depends on the Gemini account and
region. `PORT` is used by Docker and the PowerShell run script.

## Guardrails

Gemini is called once per scenario with all notes and battery capacity context.
Native JSON structured output is requested with temperature zero. The returned
JSON is still treated as untrusted and deterministically checked for:

- exactly one ordered interpretation per input note;
- only the six official directive types;
- correct `applies` and adjustment shapes;
- unique, ascending hours in `0..23`;
- finite factors, reserve values, and grid caps;
- solar factors in `[0,1]` and reserves within battery capacity;
- no missing or unexpected critical fields.

A transient provider server error receives at most one controlled retry. Invalid
output is never guessed or silently ignored.

## Optimization

The optimizer is a 24-hour linear program implemented with PuLP and CBC. It uses
non-negative grid and solar variables, a signed battery-flow variable, battery
state equations, power/energy bounds, all validated directives, and end-of-day
battery neutrality. The objective minimizes:

```text
sum(grid_kwh[h] * tariff_bdt_per_kwh[h])
```

No undocumented battery-efficiency loss is added. A separate replay validator
checks every hour, every directive, battery transitions, energy balance, final
battery state, and independently recalculates all totals before a response is
returned.

## Run locally

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

Or run directly:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /health` -> `{"status":"ok"}`
- `POST /optimize-energy` -> official interpretation and 24-hour plan schema

Malformed JSON returns HTTP 400, semantically invalid requests return HTTP 422,
provider failures return HTTP 503, and solver/replay failures return a controlled
HTTP 500 response. Internal stack traces and credentials are not returned.

## Sample request

```json
{
  "scenario_id": "README-001",
  "operator_notes": ["The cafeteria menu changes at 12 PM."],
  "hours": [
    {"hour":0,"demand_kwh":100,"solar_kwh":0,"tariff_bdt_per_kwh":8},
    {"hour":1,"demand_kwh":100,"solar_kwh":0,"tariff_bdt_per_kwh":8},
    {"hour":2,"demand_kwh":100,"solar_kwh":0,"tariff_bdt_per_kwh":8},
    {"hour":3,"demand_kwh":100,"solar_kwh":0,"tariff_bdt_per_kwh":8},
    {"hour":4,"demand_kwh":100,"solar_kwh":0,"tariff_bdt_per_kwh":8},
    {"hour":5,"demand_kwh":100,"solar_kwh":0,"tariff_bdt_per_kwh":8},
    {"hour":6,"demand_kwh":100,"solar_kwh":0,"tariff_bdt_per_kwh":8},
    {"hour":7,"demand_kwh":100,"solar_kwh":0,"tariff_bdt_per_kwh":8},
    {"hour":8,"demand_kwh":100,"solar_kwh":0,"tariff_bdt_per_kwh":8},
    {"hour":9,"demand_kwh":100,"solar_kwh":0,"tariff_bdt_per_kwh":8},
    {"hour":10,"demand_kwh":100,"solar_kwh":0,"tariff_bdt_per_kwh":8},
    {"hour":11,"demand_kwh":100,"solar_kwh":0,"tariff_bdt_per_kwh":8},
    {"hour":12,"demand_kwh":100,"solar_kwh":0,"tariff_bdt_per_kwh":8},
    {"hour":13,"demand_kwh":100,"solar_kwh":0,"tariff_bdt_per_kwh":8},
    {"hour":14,"demand_kwh":100,"solar_kwh":0,"tariff_bdt_per_kwh":8},
    {"hour":15,"demand_kwh":100,"solar_kwh":0,"tariff_bdt_per_kwh":8},
    {"hour":16,"demand_kwh":100,"solar_kwh":0,"tariff_bdt_per_kwh":8},
    {"hour":17,"demand_kwh":100,"solar_kwh":0,"tariff_bdt_per_kwh":8},
    {"hour":18,"demand_kwh":100,"solar_kwh":0,"tariff_bdt_per_kwh":8},
    {"hour":19,"demand_kwh":100,"solar_kwh":0,"tariff_bdt_per_kwh":8},
    {"hour":20,"demand_kwh":100,"solar_kwh":0,"tariff_bdt_per_kwh":8},
    {"hour":21,"demand_kwh":100,"solar_kwh":0,"tariff_bdt_per_kwh":8},
    {"hour":22,"demand_kwh":100,"solar_kwh":0,"tariff_bdt_per_kwh":8},
    {"hour":23,"demand_kwh":100,"solar_kwh":0,"tariff_bdt_per_kwh":8}
  ],
  "battery": {
    "capacity_kwh": 100,
    "initial_energy_kwh": 50,
    "minimum_energy_kwh": 50,
    "max_charge_kwh_per_hour": 0,
    "max_discharge_kwh_per_hour": 0
  }
}
```

Send it from Postman to `http://127.0.0.1:8000/optimize-energy` with
`Content-Type: application/json`.

## Sample response

The real response contains exactly 24 `hourly_plan` entries; the middle entries
below are omitted only to keep this README concise.

```json
{
  "scenario_id": "README-001",
  "directive_interpretation": [
    {
      "note_index": 0,
      "applies": false,
      "directive_type": "no_op",
      "structured_adjustment": null,
      "explanation": "The note does not affect today's energy schedule."
    }
  ],
  "hourly_plan": [
    {"hour":0,"grid_kwh":100.0,"solar_used_kwh":0.0,"battery_action":"idle","battery_kwh":0.0,"battery_energy_after_kwh":50.0},
    {"hour":1,"grid_kwh":100.0,"solar_used_kwh":0.0,"battery_action":"idle","battery_kwh":0.0,"battery_energy_after_kwh":50.0},
    {"hour":23,"grid_kwh":100.0,"solar_used_kwh":0.0,"battery_action":"idle","battery_kwh":0.0,"battery_energy_after_kwh":50.0}
  ],
  "total_grid_kwh": 2400.0,
  "total_cost_bdt": 19200.0,
  "peak_grid_kwh": 100.0,
  "plan_summary": "Optimized the 24-hour schedule with deterministic battery and energy constraints. Grid import: 2400.00 kWh; cost: 19200.00 BDT; active directives: none."
}
```

## Tests

Run the complete deterministic suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -ra
```

Run all organizer public samples through the production API pipeline using
reference directives at only the external-provider boundary:

```powershell
powershell -ExecutionPolicy Bypass -File .\run-public-samples.ps1
```

Include the real configured Gemini layer:

```powershell
powershell -ExecutionPolicy Bypass -File .\run-public-samples.ps1 -LiveLlm
```

## Docker

```powershell
docker build -t gridwise .
docker run --rm --name gridwise-api --env-file .env -p 8000:8000 gridwise
```

Check the container:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

To use another internal port, set `PORT` and map the same container port:

```powershell
docker run --rm --name gridwise-api --env-file .env -e PORT=8080 -p 8080:8080 gridwise
```

## Limitations

- Gemini availability, quotas, model access, and latency are external dependencies.
- Only one controlled retry is made for a transient Gemini server error.
- Overlapping `solar_reduction` directives are rejected because the official
  specification does not define how their factors combine.
- Natural-language ranges ending at the word “midnight” are not assigned an
  invented interpretation because that wording is not explicitly defined.
- The optimizer follows the official lossless battery model; real-world battery
  efficiency, degradation, and grid export are outside scope.
- Equivalent optimal schedules may differ when tariffs create ties.
