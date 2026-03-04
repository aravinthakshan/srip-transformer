<#
.SYNOPSIS
    Full Ablation Run - 7 configs x 7 stations = 49 experiments.

.DESCRIPTION
    Runs every combination of model config and station CSV.
    All station CSVs are taken from the hierarchial\ folder so that every
    config (including those that need hierarchical features) works correctly.

    Configs (in ablation order):
      1. baseline          - Vanilla LSTM
      2. seq               - Sequential LSTM
      3. seq_hier          - Sequential + Hierarchical features
      4. seq_fitter        - Sequential + Hierarchical + Fitter
      5. full              - Sequential + Hierarchical + Fitter + MHA
      6. baseline_mha      - Baseline + MHA
      7. bidirectional     - Bidirectional LSTM

    Stations (7 stations with hierarchical quantile CSVs):
      Barmanghat, Garudeshwar, Handia, Hoshangabad,
      Mandleshwar, Manot, Sandia

.USAGE
    From the project root (srip-transformer\):
        .\run_all.ps1

    To resume from a specific run (e.g. skip first 10):
        .\run_all.ps1 -StartFrom 11

    To run only a specific config:
        .\run_all.ps1 -ConfigFilter "seq_hier"

    To do a dry-run (print commands without executing):
        .\run_all.ps1 -DryRun
#>

param(
    [int]    $StartFrom    = 1,         # 1-indexed run number to start from
    [string] $ConfigFilter = "",        # If set, only run this config name
    [switch] $DryRun       = $false     # Print commands without executing
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Paths (all relative to project root - run this script from srip-transformer\)
# ---------------------------------------------------------------------------
$SCRIPT_ROOT   = $PSScriptRoot
$SRC_DIR       = Join-Path $SCRIPT_ROOT "src"
$CONFIGS_DIR   = Join-Path $SRC_DIR "configs"
$STATIONS_DIR  = Join-Path $SCRIPT_ROOT "hierarchial"
$OUTPUT_DIR    = Join-Path $SCRIPT_ROOT "runs"
$TRAIN_SCRIPT  = Join-Path $SRC_DIR "train.py"
$LOG_DIR       = Join-Path $SCRIPT_ROOT "run_logs"

# ---------------------------------------------------------------------------
# Config list (order = ablation progression)
# ---------------------------------------------------------------------------
$CONFIGS = @(
    "baseline",       # 1 - Vanilla LSTM
    "seq",            # 2 - Sequential
    "seq_hier",       # 3 - Sequential + Hierarchical
    "seq_fitter",     # 4 - Sequential + Hierarchical + Fitter
    "full",           # 5 - Sequential + Hierarchical + Fitter + MHA
    "baseline_mha",   # 6 - Baseline + MHA
    "bidirectional"   # 7 - Bidirectional
)

# ---------------------------------------------------------------------------
# Station list - pick up all CSVs in the hierarchial\ folder
# ---------------------------------------------------------------------------
$STATION_CSVS = Get-ChildItem -Path $STATIONS_DIR -Filter "*.csv" | Sort-Object Name

if ($STATION_CSVS.Count -eq 0) {
    Write-Error "No station CSVs found in: $STATIONS_DIR"
    exit 1
}

# ---------------------------------------------------------------------------
# Filter configs if requested
# ---------------------------------------------------------------------------
if ($ConfigFilter -ne "") {
    $CONFIGS = $CONFIGS | Where-Object { $_ -eq $ConfigFilter }
    if ($CONFIGS.Count -eq 0) {
        Write-Error "No config matched filter: '$ConfigFilter'"
        exit 1
    }
}

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
New-Item -ItemType Directory -Force -Path $OUTPUT_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $LOG_DIR    | Out-Null

$TOTAL_RUNS    = $CONFIGS.Count * $STATION_CSVS.Count
$CURRENT_RUN   = 0
$FAILED_RUNS   = @()
$START_TIME    = Get-Date

Write-Host ""
Write-Host "===============================================================" -ForegroundColor Cyan
Write-Host "  SRIP Full Ablation Suite" -ForegroundColor Cyan
Write-Host "  $($CONFIGS.Count) configs x $($STATION_CSVS.Count) stations = $TOTAL_RUNS runs" -ForegroundColor Cyan
Write-Host "  Started: $($START_TIME.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor Cyan
if ($DryRun) {
    Write-Host "  *** DRY-RUN MODE - no commands will be executed ***" -ForegroundColor Yellow
}
Write-Host "===============================================================" -ForegroundColor Cyan
Write-Host ""

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
foreach ($config in $CONFIGS) {
    $config_path = Join-Path $CONFIGS_DIR "$config.yaml"

    if (-not (Test-Path $config_path)) {
        Write-Warning "Config file not found, skipping: $config_path"
        continue
    }

    foreach ($station_csv in $STATION_CSVS) {
        $CURRENT_RUN++

        # Derive a clean station name from the filename
        # e.g. "new_Barmanghat_with_hierarchical_quantiles.csv" -> "Barmanghat"
        $station_name = $station_csv.BaseName `
            -replace "^new_", "" `
            -replace "_with_hierarchical_quantiles$", "" `
            -replace "_with_hierarchical$", ""

        # ── Skip runs before StartFrom ────────────────────────────────────
        if ($CURRENT_RUN -lt $StartFrom) {
            Write-Host "  [SKIP $CURRENT_RUN/$TOTAL_RUNS] $config @ $station_name (before StartFrom=$StartFrom)" -ForegroundColor DarkGray
            continue
        }

        # ── Banner ────────────────────────────────────────────────────────
        Write-Host "---------------------------------------------------------------" -ForegroundColor DarkCyan
        Write-Host "  RUN $CURRENT_RUN / $TOTAL_RUNS" -ForegroundColor White
        Write-Host "  Config  : $config" -ForegroundColor White
        Write-Host "  Station : $station_name" -ForegroundColor White
        Write-Host "  CSV     : $($station_csv.Name)" -ForegroundColor DarkGray
        Write-Host "---------------------------------------------------------------" -ForegroundColor DarkCyan

        # ── Log file per run ─────────────────────────────────────────────
        $log_file = Join-Path $LOG_DIR "$($CURRENT_RUN.ToString('D2'))_${config}_${station_name}.log"

        # ── Build command ─────────────────────────────────────────────────
        $cmd_args = @(
            $TRAIN_SCRIPT,
            "--config",        $config_path,
            "--station",       $station_csv.FullName,
            "--station_name",  $station_name,
            "--output_dir",    $OUTPUT_DIR
        )

        $cmd_display = "python " + ($cmd_args -join " ")

        if ($DryRun) {
            Write-Host "  [DRY-RUN] $cmd_display" -ForegroundColor Yellow
            continue
        }

        # ── Execute ───────────────────────────────────────────────────────
        $run_start = Get-Date
        Write-Host "  Started : $($run_start.ToString('HH:mm:ss'))" -ForegroundColor Gray

        try {
            # Tee output to both console and log file
            python @cmd_args 2>&1 | Tee-Object -FilePath $log_file

            $exit_code = $LASTEXITCODE
            $elapsed   = (Get-Date) - $run_start

            if ($exit_code -ne 0) {
                throw "Exit code $exit_code"
            }

            Write-Host ""
            Write-Host "  DONE  ($([int]$elapsed.TotalMinutes)m $($elapsed.Seconds)s)" -ForegroundColor Green

        } catch {
            $elapsed = (Get-Date) - $run_start
            Write-Host ""
            Write-Host "  FAILED after $([int]$elapsed.TotalMinutes)m $($elapsed.Seconds)s - $_" -ForegroundColor Red
            Write-Host "    Log: $log_file" -ForegroundColor Red
            $FAILED_RUNS += [PSCustomObject]@{
                Run     = $CURRENT_RUN
                Config  = $config
                Station = $station_name
                Error   = "$_"
                Log     = $log_file
            }
            # Continue with next run instead of aborting the whole suite
        }

        Write-Host ""
    }
}

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
$END_TIME    = Get-Date
$TOTAL_TIME  = $END_TIME - $START_TIME
$SUCCESS     = $TOTAL_RUNS - $FAILED_RUNS.Count

Write-Host ""
Write-Host "===============================================================" -ForegroundColor Cyan
Write-Host "  ABLATION SUITE COMPLETE" -ForegroundColor Cyan
Write-Host "===============================================================" -ForegroundColor Cyan
Write-Host "  Total runs  : $TOTAL_RUNS"
Write-Host "  Successful  : $SUCCESS" -ForegroundColor Green

$failed_color = "Green"
if ($FAILED_RUNS.Count -gt 0) { $failed_color = "Red" }
Write-Host "  Failed      : $($FAILED_RUNS.Count)" -ForegroundColor $failed_color

Write-Host "  Total time  : $([int]$TOTAL_TIME.TotalHours)h $($TOTAL_TIME.Minutes)m $($TOTAL_TIME.Seconds)s"
Write-Host "  Outputs in  : $OUTPUT_DIR"
Write-Host "  Logs in     : $LOG_DIR"

if ($FAILED_RUNS.Count -gt 0) {
    Write-Host ""
    Write-Host "  Failed runs:" -ForegroundColor Red
    foreach ($f in $FAILED_RUNS) {
        Write-Host "    Run $($f.Run): $($f.Config) @ $($f.Station)" -ForegroundColor Red
        Write-Host "      Error: $($f.Error)" -ForegroundColor DarkRed
        Write-Host "      Log  : $($f.Log)" -ForegroundColor DarkRed
    }
    Write-Host ""
    Write-Host "  To retry only the failed runs, use -StartFrom <run_number>" -ForegroundColor Yellow
}

Write-Host "===============================================================" -ForegroundColor Cyan
Write-Host ""
