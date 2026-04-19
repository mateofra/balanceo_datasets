"""src/generator.py — ManoSynthesizer: generador de dataset sintetico de manos
con diversidad demografica por escala MST (niveles 1..10).

Uso (lote de prueba, 10 por nivel = 100 total):
    uv run src/generator.py --size 10

Uso (1 000 por nivel = 10 000 total):
    uv run src/generator.py --size 1000

Opciones:
    --model   Ruta al archivo MANO .pkl (por defecto: models/MANO_RIGHT.pkl)
    --size    Muestras por nivel MST (por defecto: 1000)
    --pose-mode Modo de diversidad de pose: conservative|balanced|extreme
    --out     Directorio de salida (por defecto: data/synthetic_samples)
"""

#from __future__ import annotations

import argparse
import inspect
import os
import shutil
import sys
import warnings
from collections import namedtuple
from pathlib import Path

import cv2
import numpy as np

warnings.filterwarnings("ignore")


def _enable_chumpy_py313_compat() -> None:
    """Compatibilidad mínima para chumpy en Python 3.13 + NumPy recientes."""
    if not hasattr(inspect, "getargspec"):
        ArgSpec = namedtuple("ArgSpec", ["args", "varargs", "keywords", "defaults"])

        def _getargspec(func):
            spec = inspect.getfullargspec(func)
            return ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)

        inspect.getargspec = _getargspec  # type: ignore[attr-defined]

    # chumpy usa aliases antiguos de NumPy removidos en versiones recientes.
    if not hasattr(np, "int"):
        np.int = int  # type: ignore[attr-defined]
    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]
    if not hasattr(np, "bool"):
        np.bool = bool  # type: ignore[attr-defined]
    if not hasattr(np, "object"):
        np.object = object  # type: ignore[attr-defined]
    if not hasattr(np, "str"):
        np.str = str  # type: ignore[attr-defined]
    if not hasattr(np, "complex"):
        np.complex = complex  # type: ignore[attr-defined]
    if not hasattr(np, "unicode"):
        np.unicode = str  # type: ignore[attr-defined]


_enable_chumpy_py313_compat()

# ── Configuracion de rutas legacy de MANO ────────────────────────────────────
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "mano_v1_2"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "mano_v1_2", "webuser"))

# Importamos el wrapper legacy de MANO (si existe). Si no, usamos fallback con smplx.
try:
    from webuser.smpl_handpca_wrapper_HAND_only import load_model as legacy_load_model  # noqa: E402  # pyright: ignore[reportMissingImports]
    _HAS_LEGACY_WRAPPER = True
except Exception:
    legacy_load_model = None
    _HAS_LEGACY_WRAPPER = False

# ── Escala MST: 10 niveles, RGB canonico ─────────────────────────────────────
_MST_HEX = [
    "#f6ede4",
    "#f3e7db",
    "#f7ead0",
    "#eadaba",
    "#d7bd96",
    "#a07e56",
    "#825c43",
    "#604134",
    "#3a312a",
    "#292420",
]


def _hex_to_rgb_tuple(color: str) -> tuple[int, int, int]:
    value = color.lstrip("#")
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


MST_RGB: dict[str, tuple[int, int, int]] = {
    f"MST_{i + 1}": _hex_to_rgb_tuple(hex_color)
    for i, hex_color in enumerate(_MST_HEX)
}

# MANO expone 16 articulaciones. El formato comun de 21 puntos se obtiene
# agregando un vertice de la malla en la punta de cada dedo.
_MANO_FINGERTIP_VERTICES = {
    "thumb": 745,
    "index": 317,
    "middle": 444,
    "ring": 556,
    "pinky": 673,
}

# Orden canonico de 21 landmarks usado por la mayoria de pipelines de mano:
# muneca,
# pulgar(1-4), indice(1-4), medio(1-4), anular(1-4), menique(1-4)
_MANO_21_ORDER = [
    0,
    13, 14, 15,
    1, 2, 3,
    4, 5, 6,
    10, 11, 12,
    7, 8, 9,
]

# ── Intrinsecos de camara (pinhole, imagen 128 × 128) ───────────────────────
_IMG_SIZE    = 128
_FOCAL       = 300.0
_CX = _CY    = _IMG_SIZE / 2.0
_CAMERA_MTX  = np.array(
    [[_FOCAL, 0.0,    _CX],
     [0.0,    _FOCAL, _CY],
     [0.0,    0.0,    1.0]],
    dtype=np.float64,
)
_DIST_COEFFS = np.zeros(5, dtype=np.float64)
_Z_OFFSET    = 0.6   # metros: profundidad de trabajo antes del reescalado
_MARGIN      = 0.12  # fraccion de borde preservada tras ajustar al lienzo
_POSE_MODES  = ("conservative", "balanced", "extreme")
_MAX_FRONTAL_PALM_SCORE_BALANCED = 0.45
_MAX_FRONTAL_PALM_SCORE_EXTREME = 0.20  # |dot(normal_palma, eje_camara)|: extremo estricto


def _resolve_model_path(model_path: str) -> str:
    """Resuelve ruta de modelo MANO desde cwd o desde la carpeta manos/."""
    candidate = Path(model_path)
    if candidate.exists():
        return str(candidate)

    fallback = Path(_REPO_ROOT) / model_path
    if fallback.exists():
        return str(fallback)

    fallback_by_name = Path(_REPO_ROOT) / "models" / candidate.name
    if fallback_by_name.exists():
        return str(fallback_by_name)

    return str(candidate)


def _rotation_matrix_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    """Matriz de rotacion 3D (convencion X→Y→Z)."""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return rot_z @ rot_y @ rot_x


def _safe_normalize(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return vec.copy()
    return vec / norm


def _rotation_from_vectors(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Rotacion que alinea src con dst (formula de Rodrigues)."""
    a = _safe_normalize(src)
    b = _safe_normalize(dst)
    if np.linalg.norm(a) < 1e-8 or np.linalg.norm(b) < 1e-8:
        return np.eye(3, dtype=np.float64)

    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))

    if s < 1e-8:
        if c > 0.0:
            return np.eye(3, dtype=np.float64)
        # Vectores opuestos: rotar 180° alrededor de un eje ortogonal a src.
        ortho = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(a[0]) > 0.9:
            ortho = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = _safe_normalize(np.cross(a, ortho))
        k = np.array(
            [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
            dtype=np.float64,
        )
        return np.eye(3, dtype=np.float64) + 2.0 * (k @ k)

    k = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + k + k @ k * ((1.0 - c) / (s * s))


def _palm_frontal_score(joints_3d: np.ndarray) -> float:
    """Score de frontalidad de palma en [0, 1] (1 = totalmente frontal)."""
    palm_normal = np.cross(joints_3d[5] - joints_3d[0], joints_3d[17] - joints_3d[0])
    palm_normal = _safe_normalize(palm_normal)
    if np.linalg.norm(palm_normal) < 1e-8:
        return 1.0
    return float(abs(np.dot(palm_normal, np.array([0.0, 0.0, 1.0], dtype=np.float64))))


# ─────────────────────────────────────────────────────────────────────────────
class ManoSynthesizer:
    """Carga un modelo MANO de mano y genera un dataset con diversidad demografica."""

    def __init__(self, model_path: str = "models/MANO_RIGHT.pkl", ncomps: int = 45) -> None:
        self.model_path = _resolve_model_path(model_path)
        self._backend = ""

        if _HAS_LEGACY_WRAPPER and legacy_load_model is not None:
            try:
                self._m = legacy_load_model(self.model_path, ncomps=ncomps, flat_hand_mean=False)
                self._backend = "legacy"
            except Exception:
                self._backend = ""

        if not self._backend:
            try:
                import smplx
                import torch
            except Exception as exc:
                raise RuntimeError(
                    "No se pudo cargar MANO con wrapper legacy y tampoco está disponible smplx. "
                    "Instala smplx o agrega mano_v1_2/webuser."
                ) from exc

            self._torch = torch
            self._device = torch.device("cpu")

            model_file = Path(self.model_path).resolve()

            # smplx espera un layout tipo <root>/mano/MANO_RIGHT.pkl.
            # Si el .pkl está plano en models/, creamos la subcarpeta requerida.
            mano_subdir = model_file.parent / "mano"
            mano_subdir.mkdir(parents=True, exist_ok=True)
            mano_target = mano_subdir / model_file.name
            if not mano_target.exists():
                shutil.copy2(model_file, mano_target)

            model_dir = str(model_file.parent)
            is_rhand = "LEFT" not in model_file.name.upper()

            self._smplx_model = smplx.create(
                model_path=model_dir,
                model_type="mano",
                is_rhand=is_rhand,
                use_pca=False,
                flat_hand_mean=False,
                num_betas=10,
            ).to(self._device)
            self._smplx_model.eval()
            self._faces = np.asarray(self._smplx_model.faces, dtype=np.int32)
            self._pose_global = np.zeros(3, dtype=np.float32)
            self._pose_hand = np.zeros(45, dtype=np.float32)
            self._betas = np.zeros(10, dtype=np.float32)
            self._backend = "smplx"

    # ── Helpers privados ──────────────────────────────────────────────────────

    def _pose_size(self) -> int:
        if self._backend == "legacy":
            return int(self._m.pose.size)
        return 48

    def _set_pose_vector(self, pose: np.ndarray) -> None:
        if self._backend == "legacy":
            self._m.pose[:] = pose
            return

        self._pose_global = pose[:3].astype(np.float32)
        self._pose_hand = pose[3:].astype(np.float32)

    def _evaluate_model(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evalúa el modelo MANO y retorna (verts, faces, joints_16)."""
        if self._backend == "legacy":
            verts = np.array(self._m.r)
            faces = np.array(self._m.f)
            joints = np.array(self._m.J_transformed.r)
            return verts, faces, joints

        torch = self._torch
        with torch.no_grad():
            out = self._smplx_model(
                global_orient=torch.tensor(self._pose_global, dtype=torch.float32, device=self._device).unsqueeze(0),
                hand_pose=torch.tensor(self._pose_hand, dtype=torch.float32, device=self._device).unsqueeze(0),
                betas=torch.tensor(self._betas, dtype=torch.float32, device=self._device).unsqueeze(0),
                transl=torch.zeros((1, 3), dtype=torch.float32, device=self._device),
                return_verts=True,
            )

        verts = out.vertices[0].detach().cpu().numpy().astype(np.float64)
        joints_all = out.joints[0].detach().cpu().numpy().astype(np.float64)
        if joints_all.shape[0] < 16:
            raise RuntimeError(f"SMPL-X/MANO retornó {joints_all.shape[0]} joints; se esperaban al menos 16.")

        joints = joints_all[:16]
        return verts, self._faces, joints

    def _set_random_shape(self, std: float = 0.5) -> None:
        """Aleatoriza betas (diversidad morfologica)."""
        if self._backend == "legacy":
            self._m.betas[:] = np.random.normal(0.0, std, self._m.betas.size)
        else:
            self._betas = np.random.normal(0.0, std, size=10).astype(np.float32)

    def _set_random_pose(self, mode: str = "balanced") -> None:
        """Aleatoriza la pose con distintos niveles de diversidad.

        - conservative: variaciones suaves cercanas a mano relajada
        - balanced: mezcla de poses abierta/relajada/cierre parcial
        - extreme: mayor cobertura con outliers controlados
        """
        if mode not in _POSE_MODES:
            raise ValueError(f"pose_mode invalido: {mode}. Usa uno de {_POSE_MODES}")

        pose = np.zeros(self._pose_size(), dtype=np.float64)

        global_sigma = {
            "conservative": 0.12,
            "balanced": 0.25,
            "extreme": 0.38,
        }[mode]
        pose[:3] = np.random.normal(0.0, global_sigma, size=3)

        finger_size = pose.size - 3
        if finger_size <= 0:
            self._set_pose_vector(np.random.normal(0.0, 0.1, self._pose_size()))
            return

        if finger_size % 5 != 0:
            sigma = {"conservative": 0.10, "balanced": 0.18, "extreme": 0.28}[mode]
            pose[3:] = np.clip(np.random.normal(0.0, sigma, size=finger_size), -1.5, 1.5)
            self._set_pose_vector(pose)
            return

        block = finger_size // 5

        def _make_proto(levels: list[float]) -> np.ndarray:
            proto = np.zeros(finger_size, dtype=np.float64)
            for finger, level in enumerate(levels):
                start = finger * block
                end = (finger + 1) * block
                proto[start:end] = level
            return proto

        proto_open = _make_proto([0.00, 0.00, 0.00, 0.00, 0.00])
        proto_relaxed = _make_proto([0.12, 0.35, 0.45, 0.38, 0.30])
        proto_flexed = _make_proto([0.60, 0.68, 0.72, 0.70, 0.66])

        probs = {
            "conservative": np.array([0.45, 0.50, 0.05]),
            "balanced": np.array([0.25, 0.50, 0.25]),
            "extreme": np.array([0.15, 0.35, 0.50]),
        }[mode]
        idx = int(np.random.choice(3, p=probs))
        base = [proto_open, proto_relaxed, proto_flexed][idx].copy()

        gain_range = {
            "conservative": (0.85, 1.15),
            "balanced": (0.70, 1.35),
            "extreme": (0.40, 1.70),
        }[mode]
        jitter_sigma = {
            "conservative": 0.08,
            "balanced": 0.13,
            "extreme": 0.30,
        }[mode]

        for finger in range(5):
            start = finger * block
            end = (finger + 1) * block
            gain = np.random.uniform(gain_range[0], gain_range[1])
            jitter = np.random.normal(0.0, jitter_sigma, size=end - start)
            base[start:end] = base[start:end] * gain + jitter

        tail_prob = {"conservative": 0.05, "balanced": 0.18, "extreme": 0.40}[mode]
        tail_sigma = {"conservative": 0.18, "balanced": 0.32, "extreme": 0.70}[mode]
        tail_count = {"conservative": 3, "balanced": 6, "extreme": 14}[mode]
        if np.random.random() < tail_prob:
            n = min(tail_count, finger_size)
            spike_idx = np.random.choice(finger_size, size=n, replace=False)
            base[spike_idx] += np.random.normal(0.0, tail_sigma, size=n)

        clip = {"conservative": 1.0, "balanced": 1.5, "extreme": 2.5}[mode]
        pose[3:] = np.clip(base, -clip, clip)
        self._set_pose_vector(pose)

    def _mano_landmarks_21(self, verts_3d: np.ndarray, joints_3d: np.ndarray) -> np.ndarray:
        """Devuelve el layout canonico de 21 puntos a partir de la salida MANO.

        MANO provee 16 articulaciones: muneca + tres por dedo. El layout de
        21 puntos agrega las puntas de los dedos desde vertices fijos de malla,
        y luego reordena todo al convenio habitual
        muneca/pulgar/indice/medio/anular/menique.
        """
        ordered_joints = joints_3d[_MANO_21_ORDER]
        fingertip_points = np.array(
            [
                verts_3d[_MANO_FINGERTIP_VERTICES["thumb"]],
                verts_3d[_MANO_FINGERTIP_VERTICES["index"]],
                verts_3d[_MANO_FINGERTIP_VERTICES["middle"]],
                verts_3d[_MANO_FINGERTIP_VERTICES["ring"]],
                verts_3d[_MANO_FINGERTIP_VERTICES["pinky"]],
            ],
            dtype=np.float64,
        )

        landmarks_21 = np.vstack(
            [
                ordered_joints[0:1],
                ordered_joints[1:4], fingertip_points[0:1],
                ordered_joints[4:7], fingertip_points[1:2],
                ordered_joints[7:10], fingertip_points[2:3],
                ordered_joints[10:13], fingertip_points[3:4],
                ordered_joints[13:16], fingertip_points[4:5],
            ]
        )
        return landmarks_21

    def _project_and_fit(
        self,
        verts_3d: np.ndarray,
        joints_3d: np.ndarray,
        pose_mode: str,
        img_size: int = _IMG_SIZE,
        margin: float = _MARGIN,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Orienta, proyecta y escala la malla de mano a una vista frontal canonica.

        Pasos:
          1. Centrar la malla en el origen.
          2. Rotar por PCA para alinear el eje mas largo con Y (vertical),
             el segundo con X (horizontal) y la normal de la palma hacia
             la camara (+Z).
          3. Proyectar con un modelo de camara pinhole.
          4. Escalar uniformemente la proyeccion 2-D para ocupar
             ``(1 - 2·margin)`` del lienzo y quedar centrada.

        La misma transformacion afin se aplica a vertices y articulaciones,
        preservando su correspondencia en pixeles.

        Devuelve:
            (verts_2d, joints_2d): ambos arreglos (N, 2) en coordenadas de pixel.
        """
        centroid = verts_3d.mean(axis=0)
        v_c = (verts_3d - centroid).astype(np.float64)
        j_c = (joints_3d - centroid).astype(np.float64)

        # ── Orientacion PCA ────────────────────────────────────────────────
        # Filas de Vt: PC0 = eje mas largo (muneca→punta),
        #              PC1 = ancho de nudillos, PC2 = normal de palma (prof.)
        _, _, Vt = np.linalg.svd(v_c, full_matrices=False)
        pc0, pc1, pc2 = Vt[0], Vt[1], Vt[2]

        # Mapeo: pc0→Y (vertical), pc1→X (horizontal), pc2→Z (profundidad)
        R = np.stack([pc1, pc0, pc2], axis=0)   # shape (3, 3)
        v_r = v_c @ R.T
        j_r = j_c @ R.T

        # Forzar dedos por encima de la muneca sobre Y de imagen.
        # La articulacion 0 es la muneca; el resto incluye nudillos y puntas.
        if j_r[0, 1] > j_r[1:, 1].mean():      # si muneca queda arriba, invertir Y
            v_r[:, 1] *= -1
            j_r[:, 1] *= -1

        if pose_mode == "conservative":
            # En conservative mantenemos vista frontal para estabilidad.
            if v_r[:, 2].mean() < 0:
                v_r[:, 2] *= -1
                j_r[:, 2] *= -1
        else:
            # En balanced/extreme evitamos que la palma quede frontal/centrada.
            best_v = v_r
            best_j = j_r
            best_score = float("inf")

            max_score = (
                _MAX_FRONTAL_PALM_SCORE_EXTREME
                if pose_mode == "extreme"
                else _MAX_FRONTAL_PALM_SCORE_BALANCED
            )

            if pose_mode == "extreme":
                yaw_range = (50.0, 125.0)
                pitch_range = (20.0, 70.0)
                roll_range = (-35.0, 35.0)
                attempts = 10
            else:
                yaw_range = (28.0, 75.0)
                pitch_range = (10.0, 38.0)
                roll_range = (-20.0, 20.0)
                attempts = 8

            for _ in range(attempts):
                yaw = np.deg2rad(np.random.uniform(yaw_range[0], yaw_range[1]) * np.random.choice([-1.0, 1.0]))
                pitch = np.deg2rad(np.random.uniform(pitch_range[0], pitch_range[1]) * np.random.choice([-1.0, 1.0]))
                roll = np.deg2rad(np.random.uniform(roll_range[0], roll_range[1]))

                rot_view = _rotation_matrix_xyz(pitch, yaw, roll)
                v_cand = v_r @ rot_view.T
                j_cand = j_r @ rot_view.T
                frontal_score = _palm_frontal_score(j_cand)

                if frontal_score < best_score:
                    best_score = frontal_score
                    best_v = v_cand
                    best_j = j_cand

                if frontal_score <= max_score:
                    break

            v_r, j_r = best_v, best_j

            # Garantia: la palma no debe quedar frontal/centrada.
            if best_score > max_score:
                normal = np.cross(j_r[5] - j_r[0], j_r[17] - j_r[0])
                normal = _safe_normalize(normal)
                target = np.array([normal[0], normal[1], 0.0], dtype=np.float64)
                if np.linalg.norm(target) < 1e-8:
                    target = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                target = _safe_normalize(target)

                rot_force = _rotation_from_vectors(normal, target)
                v_r = v_r @ rot_force.T
                j_r = j_r @ rot_force.T
                best_score = _palm_frontal_score(j_r)

            if best_score > max_score:
                # Ultimo recurso determinista: vistas laterales fuertes.
                candidates = []
                for sign in (-1.0, 1.0):
                    rot_side = _rotation_matrix_xyz(np.deg2rad(35.0), np.deg2rad(90.0 * sign), 0.0)
                    v_cand = v_r @ rot_side.T
                    j_cand = j_r @ rot_side.T
                    candidates.append((_palm_frontal_score(j_cand), v_cand, j_cand))

                score, v_best, j_best = min(candidates, key=lambda x: x[0])
                v_r, j_r = v_best, j_best
                best_score = score

            if best_score > max_score:
                raise RuntimeError(
                    f"No se pudo imponer la restriccion de palma no frontal en modo {pose_mode}."
                )

        # Ubicar la malla en profundidad de trabajo
        v_r[:, 2] += _Z_OFFSET
        j_r[:, 2] += _Z_OFFSET

        # ── Proyeccion pinhole ─────────────────────────────────────────────
        rvec = tvec = np.zeros(3, dtype=np.float64)
        v2, _ = cv2.projectPoints(v_r, rvec, tvec, _CAMERA_MTX, _DIST_COEFFS)
        j2, _ = cv2.projectPoints(j_r, rvec, tvec, _CAMERA_MTX, _DIST_COEFFS)
        v2 = v2.reshape(-1, 2)
        j2 = j2.reshape(-1, 2)

        # ── Ajuste al lienzo ────────────────────────────────────────────────
        # Escalar para que la caja de vertices ocupe (1 - 2·margin) del lienzo,
        # y desplazar para centrar. Los joints usan exactamente la misma afin.
        lo, hi = v2.min(axis=0), v2.max(axis=0)
        extent = (hi - lo).max()
        target = img_size * (1.0 - 2.0 * margin)
        scale  = target / max(extent, 1e-8)
        center_2d = (lo + hi) / 2.0
        offset    = img_size / 2.0 - center_2d * scale

        v2 = v2 * scale + offset
        j2 = j2 * scale + offset
        return v2, j2

    def _render(
        self,
        verts_2d: np.ndarray,
        faces: np.ndarray,
        landmarks_2d: np.ndarray,
        skin_rgb: tuple[int, int, int],
        img_size: int = _IMG_SIZE,
    ) -> np.ndarray:
        """Rasteriza la malla de mano con OpenCV y devuelve una imagen BGR.

        - Fondo oscuro con ruido para simular entornos reales.
        - Triangulos de mano pintados con color MST.
        - Ruido de superficie para simular variacion de textura.
        - Landmarks dibujados como pequenos circulos verdes.
        """
        # Fondo oscuro con ruido
        canvas = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        bg_noise = np.random.normal(15.0, 20.0, canvas.shape)
        canvas = np.clip(canvas.astype(np.int32) + bg_noise, 0, 255).astype(np.uint8)

        # Pintar triangulos de mano (RGB → BGR en OpenCV)
        bgr = (int(skin_rgb[2]), int(skin_rgb[1]), int(skin_rgb[0]))
        verts_px = verts_2d.astype(np.int32)
        for tri in faces:
            pts = verts_px[tri].reshape(1, -1, 2)
            cv2.fillPoly(canvas, [pts], bgr)

        # Ruido de textura superficial
        surf_noise = np.random.normal(0.0, 12.0, canvas.shape)
        canvas = np.clip(canvas.astype(np.int32) + surf_noise, 0, 255).astype(np.uint8)

        # Superposicion de landmarks
        for pt in landmarks_2d:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < img_size and 0 <= y < img_size:
                cv2.circle(canvas, (x, y), 2, (0, 255, 0), -1)

        return canvas

    # ── API publica ───────────────────────────────────────────────────────────

    def generate_dataset(
        self,
        samples_per_tone: int = 1000,
        pose_mode: str = "balanced",
        out_dir: str = "data/synthetic_samples",
    ) -> None:
        """Genera un dataset balanceado: ``samples_per_tone`` imagenes × 10 niveles MST.

        Para cada muestra se guardan en *out_dir*:
        - ``sample_XXXXX_Type_N.png``                : imagen BGR 128×128 renderizada
        - ``sample_XXXXX_Type_N_landmarks.npy``      : landmarks 2-D GT (21, 2)
        - ``sample_XXXXX_Type_N_landmarks3d.npy``    : landmarks 3-D en espacio MANO (21, 3)

        Los archivos 2-D y 3-D comparten exactamente el mismo orden canonico.
        """
        os.makedirs(out_dir, exist_ok=True)
        sample_id = 0

        for tone_name, rgb in MST_RGB.items():
            for _ in range(samples_per_tone):
                self._set_random_shape()
                self._set_random_pose(mode=pose_mode)

                # Evaluar modelo MANO (backend legacy o smplx) → arreglos numpy
                verts, faces, joints = self._evaluate_model()
                landmarks_3d = self._mano_landmarks_21(verts, joints)  # (21, 3)

                # Orientar por PCA → proyectar → ajustar al lienzo (misma afin)
                verts_2d, landmarks_2d = self._project_and_fit(verts, landmarks_3d, pose_mode=pose_mode)

                img  = self._render(verts_2d, faces, landmarks_2d, rgb)
                stem = f"sample_{sample_id:05d}_{tone_name}"
                cv2.imwrite(os.path.join(out_dir, f"{stem}.png"), img)
                np.save(os.path.join(out_dir, f"{stem}_landmarks.npy"), landmarks_2d)
                np.save(os.path.join(out_dir, f"{stem}_landmarks3d.npy"), landmarks_3d)

                sample_id += 1

            print(f"  {tone_name}: {samples_per_tone} muestras guardadas.")

        print(f"\nDataset completo: {sample_id} muestras totales en '{out_dir}'.")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generador sintetico de dataset MANO")
    parser.add_argument(
        "--model", default="models/MANO_RIGHT.pkl",
        help="Ruta al archivo de modelo MANO .pkl (por defecto: models/MANO_RIGHT.pkl)",
    )
    parser.add_argument(
        "--size", type=int, default=1000,
        help="Muestras por nivel MST (por defecto: 1000; usa 10 para prueba rapida)",
    )
    parser.add_argument(
        "--pose-mode",
        default="balanced",
        choices=list(_POSE_MODES),
        help="Modo de diversidad de pose: conservative, balanced o extreme",
    )
    parser.add_argument(
        "--out", default="data/synthetic_samples",
        help="Directorio de salida (por defecto: data/synthetic_samples)",
    )
    args = parser.parse_args()

    print(f"Cargando modelo: {args.model}")
    synth = ManoSynthesizer(model_path=args.model)
    print(f"Generando {args.size} muestras × 10 niveles MST (pose_mode={args.pose_mode}) ...")
    synth.generate_dataset(samples_per_tone=args.size, pose_mode=args.pose_mode, out_dir=args.out)
