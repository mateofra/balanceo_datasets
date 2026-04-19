#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

require_command() {
  local command_name="$1"
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Falta el comando requerido: $command_name" >&2
    exit 1
  fi
}

log_step() {
  echo
  echo "==> $1"
}

detect_freihand_zip() {
  if [[ -n "${FREIHAND_ZIP:-}" ]]; then
    echo "$FREIHAND_ZIP"
    return 0
  fi

  local candidate
  for candidate in \
    "datasets/FreiHAND_pub_v2.zip" \
    "datasets/FreiHAND_pub_v2_eval.zip"
  do
    if [[ -f "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done

  candidate="$(find datasets -maxdepth 2 -type f -name 'FreiHAND*.zip' | sort | head -n 1 || true)"
  if [[ -n "$candidate" ]]; then
    echo "$candidate"
    return 0
  fi

  echo ""
}

ensure_freihand_extracted() {
  local freihand_zip="$1"

  if [[ -d "datasets/FreiHAND_pub_v2/training/rgb" ]]; then
    echo "FreiHAND ya esta descomprimido en datasets/FreiHAND_pub_v2/."
    return 0
  fi

  if [[ -z "$freihand_zip" ]]; then
    echo "No se encontro un zip de FreiHAND. Define FREIHAND_ZIP o coloca FreiHAND*.zip en datasets/." >&2
    exit 1
  fi

  if [[ ! -f "$freihand_zip" ]]; then
    echo "No existe el zip de FreiHAND indicado: $freihand_zip" >&2
    exit 1
  fi

  echo "Descomprimiendo FreiHAND desde: $freihand_zip"
  mkdir -p datasets
  unzip -q "$freihand_zip" -d datasets

  if [[ ! -d "datasets/FreiHAND_pub_v2/training/rgb" ]]; then
    echo "La extraccion de FreiHAND no dejo datasets/FreiHAND_pub_v2/training/rgb/." >&2
    exit 1
  fi
}

require_command uv
require_command unzip

HAGRID_DATASET="${HAGRID_DATASET:-innominate817/hagrid-sample-30k-384p}"
HAGRID_DOWNLOAD_DIR="${HAGRID_DOWNLOAD_DIR:-datasets/hagrid_sample_30k_384p}"
HAGRID_ANN_DIR="${HAGRID_ANN_DIR:-datasets/ann_subsample}"
FREIHAND_RGB_DIR="${FREIHAND_RGB_DIR:-datasets/FreiHAND_pub_v2/training/rgb}"
AUDIT_CSV="${AUDIT_CSV:-csv/mst_real_dataset.csv}"
AUDIT_SUMMARY="${AUDIT_SUMMARY:-output/mst_real_summary.json}"
REQUEST_MANIFEST="${REQUEST_MANIFEST:-datasets/synthetic_mst/metadata/manifest_synthetic_requests_blocks_qc_adjusted.csv}"
REQUEST_SUMMARY="${REQUEST_SUMMARY:-output/auditoria/synthetic_request_summary_blocks_qc_adjusted.json}"
GENERATED_MANIFEST="${GENERATED_MANIFEST:-datasets/synthetic_mst/metadata/manifest_synthetic_generated_blocks_qc_adjusted.csv}"
ACCEPTED_MANIFEST="${ACCEPTED_MANIFEST:-datasets/synthetic_mst/metadata/manifest_synthetic_accepted_blocks_qc_adjusted.csv}"
GENERATION_SUMMARY="${GENERATION_SUMMARY:-output/auditoria/synthetic_generation_summary_blocks_qc_adjusted.json}"
QC_REPORT="${QC_REPORT:-output/auditoria/synthetic_qc_report_blocks_qc_adjusted.json}"
BALANCED_MANIFEST="${BALANCED_MANIFEST:-output/manifest_balanced_blocks.csv}"
BALANCED_SUMMARY="${BALANCED_SUMMARY:-output/auditoria/manifest_balanced_blocks_summary.json}"
SYNTHETIC_IMAGES_DIR="${SYNTHETIC_IMAGES_DIR:-datasets/synthetic_mst/images_blocks_qc_adjusted}"
SYNTHETIC_STRENGTH="${SYNTHETIC_STRENGTH:-0.85}"
QC_TOLERANCE="${QC_TOLERANCE:-1}"
MANO_SAMPLES_DIR="${MANO_SAMPLES_DIR:-datasets/synthetic_mst/mano_samples_balanced}"
MANO_SAMPLES_MANIFEST="${MANO_SAMPLES_MANIFEST:-datasets/synthetic_mst/metadata/manifest_mano_samples_balanced.csv}"
MANO_SEQUENCES_DIR="${MANO_SEQUENCES_DIR:-data/processed/secuencias_stgcn/mano}"
MANO_SEQUENCES_MANIFEST="${MANO_SEQUENCES_MANIFEST:-output/manifest_mano_secuencias.csv}"
MANO_MODEL="${MANO_MODEL:-manos/models/MANO_RIGHT.pkl}"
MANO_SIZE="${MANO_SIZE:-1000}"
MANO_POSE_MODE="${MANO_POSE_MODE:-balanced}"
MANO_GENERATE="${MANO_GENERATE:-1}"

# ST-GCN: Landmarks sintéticos y secuencias temporales
BALANCED_FOR_STGCN="${BALANCED_FOR_STGCN:-$BALANCED_MANIFEST}"
LANDMARKS_DIR="${LANDMARKS_DIR:-data/processed/landmarks}"
LANDMARKS_STGCN_MANIFEST="${LANDMARKS_STGCN_MANIFEST:-output/train_manifest_stgcn.csv}"
SECUENCIAS_DIR="${SECUENCIAS_DIR:-data/processed/secuencias_stgcn}"
SECUENCIAS_MANIFEST="${SECUENCIAS_MANIFEST:-output/train_manifest_stgcn_secuencias.csv}"
STGCN_SEED="${STGCN_SEED:-42}"
STGCN_FRAMES="${STGCN_FRAMES:-16}"

log_step "1/9 Descarga de HaGRID"
uv run python src/datasets/setup/download_hagrid_kaggle.py \
  --dataset "$HAGRID_DATASET" \
  --download-dir "$HAGRID_DOWNLOAD_DIR" \
  --execute \
  --prepare-ann-subsample

log_step "2/9 Descompresion de FreiHAND"
FREIHAND_ZIP_PATH="$(detect_freihand_zip)"
ensure_freihand_extracted "$FREIHAND_ZIP_PATH"

log_step "3/9 Auditoria MST real"
mkdir -p "$(dirname "$AUDIT_CSV")" "$(dirname "$AUDIT_SUMMARY")"
uv run python src/datasets/setup/generar_mst_real_datasets.py \
  --freihand-rgb-dir "$FREIHAND_RGB_DIR" \
  --hagrid-image-roots "$HAGRID_DOWNLOAD_DIR" \
  --hagrid-annotations-dir "$HAGRID_ANN_DIR" \
  --output-csv "$AUDIT_CSV" \
  --output-summary "$AUDIT_SUMMARY"

log_step "4/9 Balanceo y generacion de solicitudes sinteticas"
mkdir -p "$(dirname "$REQUEST_MANIFEST")" "$(dirname "$GENERATED_MANIFEST")" "$(dirname "$ACCEPTED_MANIFEST")" "$(dirname "$GENERATION_SUMMARY")" "$(dirname "$QC_REPORT")" "$(dirname "$BALANCED_MANIFEST")" "$(dirname "$BALANCED_SUMMARY")"
uv run python src/datasets/manos/build_synthetic_manifest.py \
  --real-csv "$AUDIT_CSV" \
  --existing-accepted-csv "$ACCEPTED_MANIFEST" \
  --output-manifest "$REQUEST_MANIFEST" \
  --output-summary "$REQUEST_SUMMARY"

log_step "5/9 Creacion y QC de sinteticos"
uv run python src/datasets/manos/generate_synthetic_skin_tones.py \
  --request-manifest "$REQUEST_MANIFEST" \
  --output-images-dir "$SYNTHETIC_IMAGES_DIR" \
  --output-manifest "$GENERATED_MANIFEST" \
  --output-summary "$GENERATION_SUMMARY" \
  --strength "$SYNTHETIC_STRENGTH"

uv run python src/datasets/manos/qc_synthetic_dataset.py \
  --generated-manifest "$GENERATED_MANIFEST" \
  --accepted-manifest "$ACCEPTED_MANIFEST" \
  --report-json "$QC_REPORT" \
  --tolerance "$QC_TOLERANCE"

if [[ "$MANO_GENERATE" == "1" ]]; then
  echo
  echo "==> Generacion de muestras MANO"
  uv run python manos/src/generator.py \
    --model "$MANO_MODEL" \
    --size "$MANO_SIZE" \
    --pose-mode "$MANO_POSE_MODE" \
    --out "$MANO_SAMPLES_DIR"
else
  echo
  echo "==> Se omite generacion MANO (MANO_GENERATE=$MANO_GENERATE)"
fi

uv run python src/datasets/manos/build_mano_samples_manifest.py \
  --sample-dir "$MANO_SAMPLES_DIR" \
  --output-manifest "$MANO_SAMPLES_MANIFEST"

log_step "6/9 Generacion de secuencias temporales MANO"
mkdir -p "$MANO_SEQUENCES_DIR"
uv run python src/datasets/manos/generar_secuencias_temporales.py \
  --input-dir "$MANO_SAMPLES_DIR" \
  --output-dir "$MANO_SEQUENCES_DIR" \
  --output-manifest "$MANO_SEQUENCES_MANIFEST" \
  --T "$STGCN_FRAMES" \
  --seed "$STGCN_SEED"

log_step "7/9 Balance final por bloques MST"
uv run python src/datasets/manos/build_balanced_block_manifest.py \
  --input-csv "$AUDIT_CSV" \
  --input-csv "$MANO_SAMPLES_MANIFEST" \
  --output-manifest "$BALANCED_MANIFEST" \
  --output-summary "$BALANCED_SUMMARY"

log_step "8/9 Generacion de landmarks sinteticos para ST-GCN"
mkdir -p "$LANDMARKS_DIR"
uv run python src/preprocessing/generate_synthetic_landmarks.py \
  --balanced-manifest "$BALANCED_FOR_STGCN" \
  --output-dir "$LANDMARKS_DIR" \
  --output-stgcn-csv "$LANDMARKS_STGCN_MANIFEST" \
  --seed "$STGCN_SEED"

log_step "9/9 Generacion de secuencias temporales sinteticas"
mkdir -p "$SECUENCIAS_DIR"
uv run python src/preprocessing/generar_secuencias_sinteticas.py \
  --manifest "$LANDMARKS_STGCN_MANIFEST" \
  --landmarks-dir "$LANDMARKS_DIR" \
  --output-dir "$SECUENCIAS_DIR" \
  --output-manifest "$SECUENCIAS_MANIFEST" \
  --T "$STGCN_FRAMES" \
  --seed "$STGCN_SEED" \
  --verbose

echo
echo "Flujo completado."
echo "- Auditoria: $AUDIT_CSV"
echo "- Solicitudes sinteticas: $REQUEST_MANIFEST"
echo "- Imagenes sinteticas: $SYNTHETIC_IMAGES_DIR"
echo "- QC aceptados: $ACCEPTED_MANIFEST"
echo "- MANO model: $MANO_MODEL"
echo "- MANO size: $MANO_SIZE"
echo "- MANO pose_mode: $MANO_POSE_MODE"
echo "- MANO output: $MANO_SAMPLES_DIR"
echo "- MANO samples manifest: $MANO_SAMPLES_MANIFEST"
echo "- MANO sequences: $MANO_SEQUENCES_DIR"
echo "- MANO sequences manifest: $MANO_SEQUENCES_MANIFEST"
echo "- Balance final: $BALANCED_MANIFEST"
echo ""
echo "ST-GCN pipeline:"
echo "- Landmarks: $LANDMARKS_DIR"
echo "- Landmarks manifest: $LANDMARKS_STGCN_MANIFEST"
echo "- Secuencias temporales: $SECUENCIAS_DIR"
echo "- Secuencias manifest: $SECUENCIAS_MANIFEST"