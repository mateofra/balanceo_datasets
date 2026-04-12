from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# Fix para permitir ejecución como script con imports desde src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


from src.balancer.core import (
    DEFAULT_SEED,
    DEFAULT_TARGET_SIZE,
    DEFAULT_HAGRID_RATIO,
    DEFAULT_EXTREME_MST_LEVELS,
    DEFAULT_EXTREME_FACTOR,
    DEFAULT_JITTER_FACTOR,
    DEFAULT_IMPUTE_MISSING_MST,
)
from src.balancer.loaders import DataLoaders, MstLoader
from src.balancer.imputer import MstImputer
from src.balancer.sampler import DatasetBalancer
from src.balancer.writers import ManifestWriters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construye un manifiesto de entrenamiento balanceado FreiHAND + HaGRID "
            "con opcion de cuotas por bloque MST si se entrega CSV de auditoria."
        )
    )
    parser.add_argument(
        "--freihand-training-xyz",
        type=Path,
        default=Path("datasets/training_xyz.json"),
        help="Ruta a datasets/training_xyz.json de FreiHAND.",
    )
    parser.add_argument(
        "--hagrid-annotations-dir",
        type=Path,
        default=Path("datasets/ann_subsample"),
        help="Directorio de anotaciones HaGRID para entrenamiento.",
    )
    parser.add_argument(
        "--gestures",
        nargs="+",
        default=["call", "dislike", "fist", "four", "like", "mute", "ok", "one", "palm", "peace", "peace_inverted", "rock", "stop", "stop_inverted", "three", "three2", "two_up", "two_up_inverted"],
        help="Gestos HaGRID a incluir.",
    )
    parser.add_argument(
        "--mst-csv",
        type=Path,
        default=None,
        help=(
            "CSV opcional de auditoria previa con MST por sample_id/image_id. "
            "Si existe, activa balance por bloques claro/medio/oscuro."
        ),
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=DEFAULT_TARGET_SIZE,
        help="Cantidad total objetivo para el manifiesto de entrenamiento.",
    )
    parser.add_argument(
        "--hagrid-ratio",
        type=float,
        default=DEFAULT_HAGRID_RATIO,
        help="Fraccion objetivo de HaGRID en el entrenamiento (0.0-1.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Semilla para reproducibilidad del muestreo.",
    )
    parser.add_argument(
        "--extreme-mst-levels",
        nargs="+",
        type=int,
        default=list(DEFAULT_EXTREME_MST_LEVELS),
        help="Niveles MST a sobrerrepresentar durante el muestreo cuando hay MST.",
    )
    parser.add_argument(
        "--extreme-factor",
        type=float,
        default=DEFAULT_EXTREME_FACTOR,
        help="Factor de peso para niveles MST extremos (>= 1.0).",
    )
    parser.add_argument(
        "--dark-jitter-factor",
        type=float,
        default=DEFAULT_JITTER_FACTOR,
        help=(
            "Factor de replicacion virtual para candidatos MST 8-9 en entrenamiento "
            "(0.0 desactiva; por ejemplo 0.5 agrega 50%% de candidatos extra)."
        ),
    )
    parser.add_argument(
        "--output-tone-sets-dir",
        type=Path,
        default=None,
        help=(
            "Directorio opcional para exportar tres sets de entrenamiento por tono "
            "(claro, medio, oscuro). Requiere MST disponible."
        ),
    )
    parser.add_argument(
        "--output-landmark-train-dir",
        type=Path,
        default=None,
        help=(
            "Directorio opcional para exportar datos de entrenamiento del modelo "
            "de landmarks separados en tres subdirectorios por tono "
            "(claro/medio/oscuro). Requiere MST disponible."
        ),
    )
    parser.add_argument(
        "--impute-missing-mst",
        action="store_true",
        default=DEFAULT_IMPUTE_MISSING_MST,
        help="Imputa MST faltante para clasificar todos los objetos del dataset.",
    )
    parser.add_argument(
        "--no-impute-missing-mst",
        action="store_false",
        dest="impute_missing_mst",
        help="Desactiva imputacion de MST faltante.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("csv/train_manifest_balanceado_freihand_hagrid.csv"),
        help="Ruta del manifiesto CSV de salida.",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("csv/resumen_balanceo_freihand_hagrid.json"),
        help="Ruta del resumen JSON de salida.",
    )
    parser.add_argument(
        "--output-stgcn-manifest-csv",
        type=Path,
        default=None,
        help=(
            "Ruta opcional para exportar manifiesto ST-GCN (CSV) con columnas "
            "sample_id,path_landmarks,label,condition,dataset,mst,mst_origin,split."
        ),
    )
    parser.add_argument(
        "--landmarks-root-dir",
        type=Path,
        default=Path("data/processed/landmarks"),
        help=(
            "Directorio base de landmarks preprocesados (.npy). "
            "Se usa para construir path_landmarks en manifiesto ST-GCN."
        ),
    )
    parser.add_argument(
        "--hagrid-landmarks-mapping-json",
        type=Path,
        default=Path("csv/hagrid_landmarks_mapping.json"),
        help=(
            "JSON opcional sample_id->path_landmarks generado al procesar HaGRID. "
            "Si existe, se usa para resolver rutas de HaGRID de forma exacta."
        ),
    )
    parser.add_argument(
        "--hagrid-landmarks-quality-json",
        type=Path,
        default=Path("csv/hagrid_landmarks_quality.json"),
        help=(
            "JSON opcional sample_id->landmark_quality para etiquetar calidad de "
            "landmarks HaGRID (por ejemplo annotation_2d_projected/synthetic_gesture_mean)."
        ),
    )
    parser.add_argument(
        "--include-missing-mst-in-stgcn",
        action="store_true",
        default=False,
        help=(
            "Incluye filas sin MST en el manifiesto ST-GCN con condition=sin_mst. "
            "Por defecto se excluyen para mantener claro/medio/oscuro."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.target_size <= 0:
        raise ValueError("--target-size debe ser mayor que 0.")
    if not (0.0 <= args.hagrid_ratio <= 1.0):
        raise ValueError("--hagrid-ratio debe estar entre 0.0 y 1.0.")
    if args.extreme_factor < 1.0:
        raise ValueError("--extreme-factor debe ser >= 1.0.")
    if args.dark_jitter_factor < 0.0:
        raise ValueError("--dark-jitter-factor debe ser >= 0.0.")

    extreme_mst_levels = {int(level) for level in args.extreme_mst_levels if 1 <= int(level) <= 10}
    if not extreme_mst_levels:
        raise ValueError("--extreme-mst-levels debe incluir al menos un valor entre 1 y 10.")

    rng = random.Random(args.seed)

    freihand_records = DataLoaders.load_freihand_records(args.freihand_training_xyz)
    hagrid_records = DataLoaders.load_hagrid_records(args.hagrid_annotations_dir, args.gestures)

    if args.mst_csv is not None:
        if not args.mst_csv.exists():
            raise FileNotFoundError(f"No existe --mst-csv en la ruta indicada: {args.mst_csv}")

        mst_map = MstLoader.load_mst_map(args.mst_csv)
        match_report = MstLoader.compute_match_report(freihand_records, hagrid_records, mst_map)
        MstLoader.print_match_report(match_report)
        freihand_records = MstLoader.attach_mst(freihand_records, mst_map)
        hagrid_records = MstLoader.attach_mst(hagrid_records, mst_map)

    if args.impute_missing_mst:
        freihand_records = MstImputer.impute_missing_mst(freihand_records, rng)
        hagrid_records = MstImputer.impute_missing_mst(hagrid_records, rng)

    balancer = DatasetBalancer(rng, extreme_mst_levels, args.extreme_factor)
    balanced_records = balancer.build_balanced_manifest(
        freihand_records=freihand_records,
        hagrid_records=hagrid_records,
        target_size=args.target_size,
        hagrid_ratio=args.hagrid_ratio,
    )
    balanced_records = balancer.expand_with_dark_jitter_candidates(
        balanced_records,
        jitter_factor=args.dark_jitter_factor,
    )

    summary = ManifestWriters.compute_summary(balanced_records)
    ManifestWriters.write_manifest_csv(
        args.output_csv,
        balanced_records,
        extreme_mst_levels=extreme_mst_levels,
        extreme_factor=args.extreme_factor,
    )
    ManifestWriters.write_summary_json(
        args.output_summary,
        summary,
        seed=args.seed,
        extreme_mst_levels=extreme_mst_levels,
        extreme_factor=args.extreme_factor,
        jitter_factor=args.dark_jitter_factor,
    )

    if args.output_tone_sets_dir is not None:
        has_mst = any(record.mst is not None for record in balanced_records)
        if not has_mst:
            print("Aviso: no se exportan sets por tono porque no hay MST en el manifiesto.")
        else:
            ManifestWriters.write_tone_sets(args.output_tone_sets_dir, balanced_records)
            print(f"Sets por tono: {args.output_tone_sets_dir}")

    if args.output_landmark_train_dir is not None:
        has_mst = any(record.mst is not None for record in balanced_records)
        if not has_mst:
            print("Aviso: no se exporta entrenamiento de landmarks por tono porque no hay MST.")
        else:
            ManifestWriters.write_landmark_training_dirs(args.output_landmark_train_dir, balanced_records)
            print(f"Entrenamiento landmarks por tono: {args.output_landmark_train_dir}")

    if args.output_stgcn_manifest_csv is not None:
        hagrid_mapping = DataLoaders.load_landmark_mapping(args.hagrid_landmarks_mapping_json)
        hagrid_quality_mapping = DataLoaders.load_quality_mapping(args.hagrid_landmarks_quality_json)
        ManifestWriters.write_stgcn_manifest_csv(
            args.output_stgcn_manifest_csv,
            balanced_records,
            landmarks_root=args.landmarks_root_dir,
            include_missing_mst=args.include_missing_mst_in_stgcn,
            hagrid_mapping=hagrid_mapping,
            hagrid_quality_mapping=hagrid_quality_mapping,
        )
        print(f"Manifiesto ST-GCN: {args.output_stgcn_manifest_csv}")

    print("Balanceo completado.")
    print(f"Manifiesto: {args.output_csv}")
    print(f"Resumen: {args.output_summary}")


if __name__ == "__main__":
    main()
