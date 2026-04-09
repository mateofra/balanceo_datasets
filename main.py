from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run_script(script_rel_path: str, script_args: list[str]) -> int:
    """Execute a project script with the current Python interpreter."""
    script_path = ROOT / script_rel_path
    command = [sys.executable, str(script_path), *script_args]
    completed = subprocess.run(command)
    return completed.returncode


def run_train(modo: str, epochs: int | None) -> int:
    """Run training in the requested mode."""
    if modo == "supervisado":
        if epochs is None:
            return run_script("scripts/training/train_supervisado.py", [])
        train_cmd = (
            "import scripts.training.train_supervisado as t; "
            f"t.train(epochs={int(epochs)})"
        )
        return subprocess.run([sys.executable, "-c", train_cmd]).returncode

    if epochs is None:
        return run_script("scripts/training/train_autosupervisado.py", [])
    train_cmd = (
        "import scripts.training.train_autosupervisado as t; "
        f"t.train(epochs={int(epochs)})"
    )
    return subprocess.run([sys.executable, "-c", train_cmd]).returncode


def run_auditoria(sin_grafica: bool) -> int:
    """Run DPR audit and optionally generate the summary plot."""
    code = run_script("scripts/auditoria/auditoria_dpr.py", [])
    if code != 0:
        return code
    if not sin_grafica:
        return run_script("scripts/generate/generar_grafica_auditoria_dpr.py", [])
    return 0


def run_setup_data(
    *,
    download_hagrid: bool,
    execute_download: bool,
    prepare_ann_subsample: bool,
    strict_download: bool,
) -> int:
    """Bootstrap local data directories, optionally run HaGRID download assistant, then verify."""
    bootstrap_code = run_script("scripts/setup/bootstrap_local_data.py", [])
    if bootstrap_code not in (0, 2):
        return bootstrap_code

    download_code = 0
    download_fatal = False
    if download_hagrid:
        download_args: list[str] = []
        if execute_download:
            download_args.append("--execute")
        if prepare_ann_subsample:
            download_args.append("--prepare-ann-subsample")
        download_code = run_script("scripts/setup/download_hagrid_kaggle.py", download_args)
        if download_code != 0 and execute_download and strict_download:
            download_fatal = True

    verify_code = run_script("scripts/setup/bootstrap_local_data.py", ["--verify-only"])
    print("\n=== Resumen setup-data ===")
    print(f"bootstrap: code={bootstrap_code}")
    print(f"download_hagrid: {'si' if download_hagrid else 'no'} (code={download_code})")
    print(f"strict_download: {'si' if strict_download else 'no'}")
    print(f"verify: code={verify_code}")
    if download_fatal:
        print("resultado: fallo en descarga real de HaGRID")
        return download_code
    if download_code != 0 and execute_download and not strict_download:
        print("resultado: descarga real falló, pero se continúa por --strict-download desactivado")
    return verify_code


def print_pipeline_summary(
    *,
    modo: str,
    epochs: int | None,
    skip_train: bool,
    sin_grafica: bool,
    train_code: int,
    auditoria_code: int,
) -> None:
    """Print a concise execution summary for the pipeline command."""
    training_status = "omitido" if skip_train else "ok"
    grafica_status = "omitida" if sin_grafica else "ok"
    epochs_txt = str(epochs) if epochs is not None else "default del script"

    print("\n=== Resumen pipeline ===")
    print(f"modo: {modo}")
    print(f"epochs: {epochs_txt}")
    print(f"entrenamiento: {training_status} (code={train_code})")
    print(f"auditoria: ok (code={auditoria_code})")
    print(f"grafica auditoria: {grafica_status}")
    print(f"artefactos training: {ROOT / 'output' / 'training'}")
    print(f"artefactos auditoria: {ROOT / 'output' / 'auditoria'}")
    if not sin_grafica:
        print(f"grafica: {ROOT / 'graficos' / 'auditoria_dpr'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launcher unificado para balanceo_datasets",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_balanceo = subparsers.add_parser(
        "balanceo",
        help="Ejecuta el balanceador principal",
    )
    parser_balanceo.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Argumentos extra para src/balancer/balancear_freihand_hagrid.py",
    )

    parser_train = subparsers.add_parser(
        "train",
        help="Entrenamiento ST-GCN (supervisado o auto-supervisado)",
    )
    parser_train.add_argument(
        "--modo",
        choices=["supervisado", "autosupervisado"],
        default="supervisado",
        help="Modo de entrenamiento",
    )
    parser_train.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Si se define, sobreescribe epochs en el script elegido",
    )

    parser_auditoria = subparsers.add_parser(
        "auditoria",
        help="Ejecuta auditoría DPR y, opcionalmente, su gráfica",
    )
    parser_auditoria.add_argument(
        "--sin-grafica",
        action="store_true",
        help="Si se activa, no genera la gráfica de auditoría",
    )

    parser_pipeline = subparsers.add_parser(
        "pipeline",
        help="Ejecuta entrenamiento y luego auditoría de forma secuencial",
    )
    parser_pipeline.add_argument(
        "--modo",
        choices=["supervisado", "autosupervisado"],
        default="supervisado",
        help="Modo de entrenamiento para el pipeline",
    )
    parser_pipeline.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Si se define, sobreescribe epochs en el entrenamiento",
    )
    parser_pipeline.add_argument(
        "--sin-grafica",
        action="store_true",
        help="Si se activa, no genera la gráfica de auditoría",
    )
    parser_pipeline.add_argument(
        "--skip-train",
        action="store_true",
        help="Si se activa, omite entrenamiento y ejecuta solo auditoría",
    )

    parser_setup_data = subparsers.add_parser(
        "setup-data",
        help="Bootstrap de datos locales y verificación; opcionalmente descarga asistida de HaGRID",
    )
    parser_setup_data.add_argument(
        "--download-hagrid",
        action="store_true",
        help="Ejecuta el asistente de descarga de HaGRID (Kaggle CLI)",
    )
    parser_setup_data.add_argument(
        "--execute-download",
        action="store_true",
        help="Si se usa con --download-hagrid, ejecuta descarga real (sin esto, dry-run)",
    )
    parser_setup_data.add_argument(
        "--prepare-ann-subsample",
        action="store_true",
        help="Si se usa con --download-hagrid, prepara datasets/ann_subsample",
    )
    parser_setup_data.add_argument(
        "--strict-download",
        action="store_true",
        help=(
            "Si se usa con --execute-download, un fallo de descarga hace fallar setup-data; "
            "si no se activa, setup-data continúa con verify"
        ),
    )

    subparsers.add_parser(
        "test-forward",
        help="Ejecuta el test rápido de forward del modelo",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "balanceo":
        return run_script("src/balancer/balancear_freihand_hagrid.py", args.args)

    if args.command == "train":
        return run_train(modo=args.modo, epochs=args.epochs)

    if args.command == "auditoria":
        return run_auditoria(sin_grafica=args.sin_grafica)

    if args.command == "pipeline":
        train_code = 0
        if not args.skip_train:
            train_code = run_train(modo=args.modo, epochs=args.epochs)
            if train_code != 0:
                return train_code
        auditoria_code = run_auditoria(sin_grafica=args.sin_grafica)
        if auditoria_code != 0:
            return auditoria_code
        print_pipeline_summary(
            modo=args.modo,
            epochs=args.epochs,
            skip_train=args.skip_train,
            sin_grafica=args.sin_grafica,
            train_code=train_code,
            auditoria_code=auditoria_code,
        )
        return 0

    if args.command == "setup-data":
        return run_setup_data(
            download_hagrid=args.download_hagrid,
            execute_download=args.execute_download,
            prepare_ann_subsample=args.prepare_ann_subsample,
            strict_download=args.strict_download,
        )

    if args.command == "test-forward":
        return run_script("scripts/training/test_forward.py", [])

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
