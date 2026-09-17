param(
    [string]$Config = "configs/experiments/all_experiments.yaml",
    [string]$Out = "runs/reproduce_all"
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $Out | Out-Null

foreach ($Exp in @("E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10", "E11", "E12", "E13", "E14", "E15", "E16", "E17", "E18", "E19", "E20", "E21", "E22", "E23", "E24")) {
    python scripts/run_experiment.py --config $Config --experiment $Exp --output (Join-Path $Out $Exp)
}

python scripts/reproduce_figures.py --results $Out --figures (Join-Path $Out "figures")
