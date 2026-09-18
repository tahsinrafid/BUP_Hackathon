$hours = 0..23 | ForEach-Object {
    @{
        hour = $_
        demand_kwh = 100
        solar_kwh = if ($_ -ge 8 -and $_ -le 16) { 40 } else { 0 }
        tariff_bdt_per_kwh = if ($_ -ge 18 -and $_ -le 21) { 20 } else { 5 }
    }
}

$payload = @{
    scenario_id = "LIVE-OPTIMIZE-001"
    operator_notes = @(
        "Rooftop solar is reduced by 80% from 1 PM until 3 PM.",
        "Keep at least 50% of battery capacity from 6 PM until 9 PM.",
        "The cafeteria menu changes tomorrow."
    )
    hours = $hours
    battery = @{
        capacity_kwh = 200
        initial_energy_kwh = 100
        minimum_energy_kwh = 40
        max_charge_kwh_per_hour = 50
        max_discharge_kwh_per_hour = 50
    }
}

$body = $payload | ConvertTo-Json -Depth 5
Invoke-RestMethod `
    -Method Post `
    -Uri "http://127.0.0.1:8000/optimize-energy" `
    -ContentType "application/json" `
    -Body $body | ConvertTo-Json -Depth 8
