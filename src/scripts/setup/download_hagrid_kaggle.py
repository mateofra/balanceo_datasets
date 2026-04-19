from __future__ import annotations

import argparse
import getpass
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_DATASET = "kapitanov/hagrid"
DEFAULT_LOCAL_KAGGLE_DIR = Path("secrets") / "kaggle"
DEFAULT_KAGGLE_JSON_TEMPLATE = {
    "username": "YOUR_KAGGLE_USERNAME",
    "key": "YOUR_KAGGLE_KEY",
}


def has_kaggle_cli() -> bool:
    return shutil.which("kaggle") is not None


def find_kaggle_credentials() -> Path | None:
    config_dir = os.environ.get("KAGGLE_CONFIG_DIR")
    candidates: list[Path] = []
    if config_dir:
        candidates.append(Path(config_dir) / "kaggle.json")
    candidates.append(Path.home() / ".kaggle" / "kaggle.json")

    for path in candidates:
        if path.exists():
            return path
    return None


def _is_valid_kaggle_json(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    username = str(payload.get("username", "")).strip()
    key = str(payload.get("key", "")).strip()
    if not username or not key:
        return False
    if username == DEFAULT_KAGGLE_JSON_TEMPLATE["username"]:
        return False
    if key == DEFAULT_KAGGLE_JSON_TEMPLATE["key"]:
        return False
    return True


def _write_kaggle_json_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(DEFAULT_KAGGLE_JSON_TEMPLATE, indent=2) + "\n",
        encoding="utf-8",
    )


def ensure_local_kaggle_config() -> tuple[Path, bool, str]:
    """Ensure secrets/kaggle/kaggle.json exists and report whether it is usable."""
    local_dir = DEFAULT_LOCAL_KAGGLE_DIR
    local_json = local_dir / "kaggle.json"

    # Default to local project credentials path unless user explicitly sets another one.
    if not os.environ.get("KAGGLE_CONFIG_DIR"):
        os.environ["KAGGLE_CONFIG_DIR"] = str(local_dir)

    if not local_json.exists():
        _write_kaggle_json_template(local_json)
        return (
            local_json,
            False,
            "Se creó plantilla local de credenciales en secrets/kaggle/kaggle.json",
        )

    if not _is_valid_kaggle_json(local_json):
        backup_path = local_json.with_suffix(".json.bak")
        shutil.copy2(local_json, backup_path)
        _write_kaggle_json_template(local_json)
        return (
            local_json,
            False,
            (
                "kaggle.json local inválido; se guardó backup en "
                f"{backup_path} y se regeneró plantilla"
            ),
        )

    return local_json, True, "Credenciales locales detectadas"


def run_cmd(command: list[str], dry_run: bool) -> int:
    print("$ " + " ".join(command))
    if dry_run:
        return 0
    return subprocess.run(command, check=False).returncode


def collect_annotation_jsons(search_root: Path) -> list[Path]:
    # JSON esperados por gesto para el pipeline actual.
    gesture_jsons = sorted(
        p for p in search_root.rglob("*.json")
        if p.name.lower() != "kaggle.json"
    )
    return gesture_jsons


def copy_annotations_to_ann_subsample(
    source_root: Path,
    target_dir: Path,
    dry_run: bool,
) -> int:
    json_files = collect_annotation_jsons(source_root)
    if not json_files:
        print(f"No se encontraron JSON de anotaciones en: {source_root}")
        return 1

    if not dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for json_file in json_files:
        dst = target_dir / json_file.name
        print(f"Copiar: {json_file} -> {dst}")
        if not dry_run:
            shutil.copy2(json_file, dst)
        copied += 1

    print(f"Anotaciones preparadas en {target_dir} (total: {copied})")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Descarga asistida de HaGRID con Kaggle CLI y preparación opcional "
            "de anotaciones para datasets/ann_subsample."
        )
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Dataset de Kaggle en formato owner/name",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("datasets") / "hagrid_kaggle_raw",
        help="Directorio donde Kaggle descargará y descomprimirá el dataset",
    )
    parser.add_argument(
        "--prepare-ann-subsample",
        action="store_true",
        help=(
            "Si se activa, intenta copiar JSON de anotaciones encontrados en la "
            "descarga hacia datasets/ann_subsample"
        ),
    )
    parser.add_argument(
        "--ann-source",
        type=Path,
        default=None,
        help=(
            "Ruta base para buscar JSON de anotaciones. Si no se define, usa "
            "--download-dir"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Muestra comandos y acciones sin ejecutar cambios (default: activo)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Ejecuta realmente la descarga y copias (desactiva dry-run)",
    )
    parser.add_argument(
        "--kaggle-username",
        default=None,
        help="Username de Kaggle para escribir/actualizar secrets/kaggle/kaggle.json",
    )
    parser.add_argument(
        "--kaggle-key",
        default=None,
        help="API key de Kaggle para escribir/actualizar secrets/kaggle/kaggle.json",
    )
    parser.add_argument(
        "--ask-credentials",
        action="store_true",
        help="Solicita username y key por terminal para completar kaggle.json",
    )
    return parser.parse_args()


def write_kaggle_credentials(path: Path, username: str, key: str) -> None:
    payload = {"username": username.strip(), "key": key.strip()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def maybe_fill_credentials_from_args_or_prompt(
    *,
    kaggle_json_path: Path,
    username_arg: str | None,
    key_arg: str | None,
    ask_credentials: bool,
) -> tuple[bool, str]:
    username = (username_arg or "").strip()
    key = (key_arg or "").strip()

    if username and key:
        write_kaggle_credentials(kaggle_json_path, username, key)
        return True, "Credenciales Kaggle guardadas desde argumentos de terminal"

    if ask_credentials:
        if not username:
            username = input("Kaggle username: ").strip()
        if not key:
            key = getpass.getpass("Kaggle key: ").strip()
        if username and key:
            write_kaggle_credentials(kaggle_json_path, username, key)
            return True, "Credenciales Kaggle guardadas desde prompt interactivo"
        return False, "No se pudieron capturar credenciales validas desde terminal"

    return False, "No se proporcionaron credenciales por argumentos ni prompt"


def main() -> int:
    args = parse_args()
    dry_run = args.dry_run and not args.execute

    local_kaggle_json, local_kaggle_ok, local_kaggle_msg = ensure_local_kaggle_config()

    provided_ok, provided_msg = maybe_fill_credentials_from_args_or_prompt(
        kaggle_json_path=local_kaggle_json,
        username_arg=args.kaggle_username,
        key_arg=args.kaggle_key,
        ask_credentials=args.ask_credentials,
    )
    if provided_ok:
        local_kaggle_ok = _is_valid_kaggle_json(local_kaggle_json)
        local_kaggle_msg = provided_msg

    print(local_kaggle_msg)
    if not provided_ok and (args.kaggle_username or args.kaggle_key or args.ask_credentials):
        print(provided_msg)
    print(f"KAGGLE_CONFIG_DIR activo: {os.environ.get('KAGGLE_CONFIG_DIR')}")
    print(f"kaggle.json local: {local_kaggle_json}")

    if not has_kaggle_cli():
        print("No se encontró Kaggle CLI en PATH.")
        if dry_run:
            print("Modo DRY-RUN: se omite la validación estricta y no se ejecuta descarga.")
            print("Instala con: uv add kaggle")
            print("Hecho (simulación).")
            return 0
        print("Instala con: uv add kaggle")
        return 1

    creds = find_kaggle_credentials()
    using_local_creds = creds is not None and creds.resolve() == local_kaggle_json.resolve()
    if creds is None or (using_local_creds and not local_kaggle_ok):
        print("No se encontró kaggle.json de credenciales.")
        if dry_run:
            print("Modo DRY-RUN: se omite la validación estricta y no se ejecuta descarga.")
            print("Ubicación esperada: ~/.kaggle/kaggle.json o KAGGLE_CONFIG_DIR/kaggle.json")
            print("Hecho (simulación).")
            return 0
        print("Ubicación esperada: ~/.kaggle/kaggle.json o KAGGLE_CONFIG_DIR/kaggle.json")
        print(
            "Completa username y key reales en "
            f"{local_kaggle_json} y vuelve a ejecutar."
        )
        return 1

    print(f"Kaggle CLI: OK")
    print(f"Credenciales: {creds}")
    print(f"Modo: {'DRY-RUN' if dry_run else 'EXECUTE'}")

    download_dir = args.download_dir
    if not dry_run:
        download_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        args.dataset,
        "-p",
        str(download_dir),
        "--unzip",
    ]

    code = run_cmd(cmd, dry_run=dry_run)
    if code != 0:
        print("Falló la descarga con Kaggle CLI.")
        return code

    if args.prepare_ann_subsample:
        ann_source = args.ann_source or download_dir
        target = Path("datasets") / "ann_subsample"
        code = copy_annotations_to_ann_subsample(
            source_root=ann_source,
            target_dir=target,
            dry_run=dry_run,
        )
        if code != 0:
            return code

    print("Hecho.")
    print("Siguiente paso recomendado: uv run python src/scripts/setup/bootstrap_local_data.py --verify-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
