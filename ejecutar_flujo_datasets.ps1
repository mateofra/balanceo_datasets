Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ROOT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ROOT_DIR

function Require-Command {
    param([Parameter(Mandatory = $true)][string]$CommandName)

    if (-not (Get-Command $CommandName -ErrorAction SilentlyContinue)) {
        throw "Falta el comando requerido: $CommandName"
    }
}

function Log-Step {
    param([Parameter(Mandatory = $true)][string]$Label)

    Write-Host ""
    Write-Host "==> $Label"
}

function Get-EnvOrDefault {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$DefaultValue
    )

    $value = [System.Environment]::GetEnvironmentVariable($Name)
    if ([string]::IsNullOrWhiteSpace($value)) {
        return $DefaultValue
    }
    return $value
}

function Ensure-ParentDirectory {
    param([Parameter(Mandatory = $true)][string]$Path)

    $parent = Split-Path -Parent $Path
    if (-not [string]::IsNullOrWhiteSpace($parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
}

function Detect-FreihandZip {
    $freihandZip = [System.Environment]::GetEnvironmentVariable("FREIHAND_ZIP")
    if (-not [string]::IsNullOrWhiteSpace($freihandZip)) {
        return $freihandZip
    }

    $candidates = @(
        "datasets/FreiHAND_pub_v2.zip",
        "datasets/FreiHAND_pub_v2_eval.zip"
    )

    foreach ($candidate in $candidates) {
        if (Test-Path -Path $candidate -PathType Leaf) {
            return $candidate
        }
    }

    if (Test-Path -Path "datasets" -PathType Container) {
        $found = Get-ChildItem -Path "datasets" -Recurse -File -Filter "FreiHAND*.zip" -ErrorAction SilentlyContinue |
            Sort-Object FullName |
            Select-Object -First 1
        if ($null -ne $found) {
            return $found.FullName
        }
    }

    return ""
}

function Ensure-FreihandExtracted {
    param([Parameter(Mandatory = $true)][string]$FreihandZip)

    if (Test-Path -Path "datasets/FreiHAND_pub_v2/training/rgb" -PathType Container) {
        Write-Host "FreiHAND ya esta descomprimido en datasets/FreiHAND_pub_v2/."
        return
    }

    if ([string]::IsNullOrWhiteSpace($FreihandZip)) {
        throw "No se encontro un zip de FreiHAND. Define FREIHAND_ZIP o coloca FreiHAND*.zip en datasets/."
    }

    if (-not (Test-Path -Path $FreihandZip -PathType Leaf)) {
        throw "No existe el zip de FreiHAND indicado: $FreihandZip"
    }

    Write-Host "Descomprimiendo FreiHAND desde: $FreihandZip"
    New-Item -ItemType Directory -Path "datasets" -Force | Out-Null
    Expand-Archive -Path $FreihandZip -DestinationPath "datasets" -Force

    if (-not (Test-Path -Path "datasets/FreiHAND_pub_v2/training/rgb" -PathType Container)) {
        throw "La extraccion de FreiHAND no dejo datasets/FreiHAND_pub_v2/training/rgb/."
    }
}

Require-Command "uv"

$HAGRID_DATASET = Get-EnvOrDefault -Name "HAGRID_DATASET" -DefaultValue "innominate817/hagrid-sample-30k-384p"
$HAGRID_DOWNLOAD_DIR = Get-EnvOrDefault -Name "HAGRID_DOWNLOAD_DIR" -DefaultValue "datasets/hagrid_sample_30k_384p"
$HAGRID_ANN_DIR = Get-EnvOrDefault -Name "HAGRID_ANN_DIR" -DefaultValue "datasets/ann_subsample"
$FREIHAND_RGB_DIR = Get-EnvOrDefault -Name "FREIHAND_RGB_DIR" -DefaultValue "datasets/FreiHAND_pub_v2/training/rgb"
$AUDIT_CSV = Get-EnvOrDefault -Name "AUDIT_CSV" -DefaultValue "csv/mst_real_dataset.csv"
$AUDIT_SUMMARY = Get-EnvOrDefault -Name "AUDIT_SUMMARY" -DefaultValue "output/mst_real_summary.json"
$REQUEST_MANIFEST = Get-EnvOrDefault -Name "REQUEST_MANIFEST" -DefaultValue "datasets/synthetic_mst/metadata/manifest_synthetic_requests_blocks_qc_adjusted.csv"
$REQUEST_SUMMARY = Get-EnvOrDefault -Name "REQUEST_SUMMARY" -DefaultValue "output/auditoria/synthetic_request_summary_blocks_qc_adjusted.json"
$GENERATED_MANIFEST = Get-EnvOrDefault -Name "GENERATED_MANIFEST" -DefaultValue "datasets/synthetic_mst/metadata/manifest_synthetic_generated_blocks_qc_adjusted.csv"
$ACCEPTED_MANIFEST = Get-EnvOrDefault -Name "ACCEPTED_MANIFEST" -DefaultValue "datasets/synthetic_mst/metadata/manifest_synthetic_accepted_blocks_qc_adjusted.csv"
$GENERATION_SUMMARY = Get-EnvOrDefault -Name "GENERATION_SUMMARY" -DefaultValue "output/auditoria/synthetic_generation_summary_blocks_qc_adjusted.json"
$QC_REPORT = Get-EnvOrDefault -Name "QC_REPORT" -DefaultValue "output/auditoria/synthetic_qc_report_blocks_qc_adjusted.json"
$BALANCED_MANIFEST = Get-EnvOrDefault -Name "BALANCED_MANIFEST" -DefaultValue "output/manifest_balanced_blocks.csv"
$BALANCED_SUMMARY = Get-EnvOrDefault -Name "BALANCED_SUMMARY" -DefaultValue "output/auditoria/manifest_balanced_blocks_summary.json"
$SYNTHETIC_IMAGES_DIR = Get-EnvOrDefault -Name "SYNTHETIC_IMAGES_DIR" -DefaultValue "datasets/synthetic_mst/images_blocks_qc_adjusted"
$SYNTHETIC_STRENGTH = Get-EnvOrDefault -Name "SYNTHETIC_STRENGTH" -DefaultValue "0.85"
$QC_TOLERANCE = Get-EnvOrDefault -Name "QC_TOLERANCE" -DefaultValue "1"
$MANO_SAMPLES_DIR = Get-EnvOrDefault -Name "MANO_SAMPLES_DIR" -DefaultValue "datasets/synthetic_mst/mano_samples_balanced"
$MANO_SAMPLES_MANIFEST = Get-EnvOrDefault -Name "MANO_SAMPLES_MANIFEST" -DefaultValue "datasets/synthetic_mst/metadata/manifest_mano_samples_balanced.csv"
$MANO_SEQUENCES_DIR = Get-EnvOrDefault -Name "MANO_SEQUENCES_DIR" -DefaultValue "data/processed/secuencias_stgcn/mano"
$MANO_SEQUENCES_MANIFEST = Get-EnvOrDefault -Name "MANO_SEQUENCES_MANIFEST" -DefaultValue "output/manifest_mano_secuencias.csv"
$MANO_MODEL = Get-EnvOrDefault -Name "MANO_MODEL" -DefaultValue "manos/models/MANO_RIGHT.pkl"
$MANO_SIZE = Get-EnvOrDefault -Name "MANO_SIZE" -DefaultValue "1000"
$MANO_POSE_MODE = Get-EnvOrDefault -Name "MANO_POSE_MODE" -DefaultValue "balanced"
$MANO_GENERATE = Get-EnvOrDefault -Name "MANO_GENERATE" -DefaultValue "1"

# ST-GCN: Landmarks sinteticos y secuencias temporales
$BALANCED_FOR_STGCN = Get-EnvOrDefault -Name "BALANCED_FOR_STGCN" -DefaultValue $BALANCED_MANIFEST
$LANDMARKS_DIR = Get-EnvOrDefault -Name "LANDMARKS_DIR" -DefaultValue "data/processed/landmarks"
$LANDMARKS_STGCN_MANIFEST = Get-EnvOrDefault -Name "LANDMARKS_STGCN_MANIFEST" -DefaultValue "output/train_manifest_stgcn.csv"
$SECUENCIAS_DIR = Get-EnvOrDefault -Name "SECUENCIAS_DIR" -DefaultValue "data/processed/secuencias_stgcn"
$SECUENCIAS_MANIFEST = Get-EnvOrDefault -Name "SECUENCIAS_MANIFEST" -DefaultValue "output/train_manifest_stgcn_secuencias.csv"
$STGCN_SEED = Get-EnvOrDefault -Name "STGCN_SEED" -DefaultValue "42"
$STGCN_FRAMES = Get-EnvOrDefault -Name "STGCN_FRAMES" -DefaultValue "16"

Log-Step "1/9 Descarga de HaGRID"
uv run python src/datasets/setup/download_hagrid_kaggle.py `
  --dataset "$HAGRID_DATASET" `
  --download-dir "$HAGRID_DOWNLOAD_DIR" `
  --execute `
  --prepare-ann-subsample

Log-Step "2/9 Descompresion de FreiHAND"
$FREIHAND_ZIP_PATH = Detect-FreihandZip
Ensure-FreihandExtracted -FreihandZip $FREIHAND_ZIP_PATH

Log-Step "3/9 Auditoria MST real"
Ensure-ParentDirectory -Path $AUDIT_CSV
Ensure-ParentDirectory -Path $AUDIT_SUMMARY
uv run python src/datasets/setup/generar_mst_real_datasets.py `
  --freihand-rgb-dir "$FREIHAND_RGB_DIR" `
  --hagrid-image-roots "$HAGRID_DOWNLOAD_DIR" `
  --hagrid-annotations-dir "$HAGRID_ANN_DIR" `
  --output-csv "$AUDIT_CSV" `
  --output-summary "$AUDIT_SUMMARY"

Log-Step "4/9 Balanceo y generacion de solicitudes sinteticas"
Ensure-ParentDirectory -Path $REQUEST_MANIFEST
Ensure-ParentDirectory -Path $GENERATED_MANIFEST
Ensure-ParentDirectory -Path $ACCEPTED_MANIFEST
Ensure-ParentDirectory -Path $GENERATION_SUMMARY
Ensure-ParentDirectory -Path $QC_REPORT
Ensure-ParentDirectory -Path $BALANCED_MANIFEST
Ensure-ParentDirectory -Path $BALANCED_SUMMARY
uv run python src/datasets/manos/build_synthetic_manifest.py `
  --real-csv "$AUDIT_CSV" `
  --existing-accepted-csv "$ACCEPTED_MANIFEST" `
  --output-manifest "$REQUEST_MANIFEST" `
  --output-summary "$REQUEST_SUMMARY"

Log-Step "5/9 Creacion y QC de sinteticos"
uv run python src/datasets/manos/generate_synthetic_skin_tones.py `
  --request-manifest "$REQUEST_MANIFEST" `
  --output-images-dir "$SYNTHETIC_IMAGES_DIR" `
  --output-manifest "$GENERATED_MANIFEST" `
  --output-summary "$GENERATION_SUMMARY" `
  --strength "$SYNTHETIC_STRENGTH"

uv run python src/datasets/manos/qc_synthetic_dataset.py `
  --generated-manifest "$GENERATED_MANIFEST" `
  --accepted-manifest "$ACCEPTED_MANIFEST" `
  --report-json "$QC_REPORT" `
  --tolerance "$QC_TOLERANCE"

if ($MANO_GENERATE -eq "1") {
    Write-Host ""
    Write-Host "==> Generacion de muestras MANO"
    uv run python manos/src/generator.py `
      --model "$MANO_MODEL" `
      --size "$MANO_SIZE" `
      --pose-mode "$MANO_POSE_MODE" `
      --out "$MANO_SAMPLES_DIR"
}
else {
    Write-Host ""
    Write-Host "==> Se omite generacion MANO (MANO_GENERATE=$MANO_GENERATE)"
}

uv run python src/datasets/manos/build_mano_samples_manifest.py `
  --sample-dir "$MANO_SAMPLES_DIR" `
  --output-manifest "$MANO_SAMPLES_MANIFEST"

Log-Step "6/9 Generacion de secuencias temporales MANO"
New-Item -ItemType Directory -Path $MANO_SEQUENCES_DIR -Force | Out-Null
uv run python src/datasets/manos/generar_secuencias_temporales.py `
  --input-dir "$MANO_SAMPLES_DIR" `
  --output-dir "$MANO_SEQUENCES_DIR" `
  --output-manifest "$MANO_SEQUENCES_MANIFEST" `
  --T "$STGCN_FRAMES" `
  --seed "$STGCN_SEED"

Log-Step "7/9 Balance final por bloques MST"
uv run python src/datasets/manos/build_balanced_block_manifest.py `
  --input-csv "$AUDIT_CSV" `
  --input-csv "$MANO_SAMPLES_MANIFEST" `
  --output-manifest "$BALANCED_MANIFEST" `
  --output-summary "$BALANCED_SUMMARY"

Log-Step "8/9 Generacion de landmarks sinteticos para ST-GCN"
New-Item -ItemType Directory -Path $LANDMARKS_DIR -Force | Out-Null
uv run python src/preprocessing/generate_synthetic_landmarks.py `
  --balanced-manifest "$BALANCED_FOR_STGCN" `
  --output-dir "$LANDMARKS_DIR" `
  --output-stgcn-csv "$LANDMARKS_STGCN_MANIFEST" `
  --seed "$STGCN_SEED"

Log-Step "9/9 Generacion de secuencias temporales sinteticas"
New-Item -ItemType Directory -Path $SECUENCIAS_DIR -Force | Out-Null
uv run python src/preprocessing/generar_secuencias_sinteticas.py `
  --manifest "$LANDMARKS_STGCN_MANIFEST" `
  --landmarks-dir "$LANDMARKS_DIR" `
  --output-dir "$SECUENCIAS_DIR" `
  --output-manifest "$SECUENCIAS_MANIFEST" `
  --T "$STGCN_FRAMES" `
  --seed "$STGCN_SEED" `
  --verbose

Write-Host ""
Write-Host "Flujo completado."
Write-Host "- Auditoria: $AUDIT_CSV"
Write-Host "- Solicitudes sinteticas: $REQUEST_MANIFEST"
Write-Host "- Imagenes sinteticas: $SYNTHETIC_IMAGES_DIR"
Write-Host "- QC aceptados: $ACCEPTED_MANIFEST"
Write-Host "- MANO model: $MANO_MODEL"
Write-Host "- MANO size: $MANO_SIZE"
Write-Host "- MANO pose_mode: $MANO_POSE_MODE"
Write-Host "- MANO output: $MANO_SAMPLES_DIR"
Write-Host "- MANO samples manifest: $MANO_SAMPLES_MANIFEST"
Write-Host "- MANO sequences: $MANO_SEQUENCES_DIR"
Write-Host "- MANO sequences manifest: $MANO_SEQUENCES_MANIFEST"
Write-Host "- Balance final: $BALANCED_MANIFEST"
Write-Host ""
Write-Host "ST-GCN pipeline:"
Write-Host "- Landmarks: $LANDMARKS_DIR"
Write-Host "- Landmarks manifest: $LANDMARKS_STGCN_MANIFEST"
Write-Host "- Secuencias temporales: $SECUENCIAS_DIR"
Write-Host "- Secuencias manifest: $SECUENCIAS_MANIFEST"
