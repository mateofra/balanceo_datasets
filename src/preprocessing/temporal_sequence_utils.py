from __future__ import annotations

import zlib

import numpy as np


EXPECTED_LANDMARK_SHAPE = (21, 3)


def sample_seed(base_seed: int, sample_id: str) -> int:
    """Genera una semilla estable por muestra."""
    sample_hash = zlib.crc32(sample_id.encode("utf-8")) & 0xFFFFFFFF
    return (base_seed + sample_hash) % (2**32)


def _rotation_z(angle_rad: float) -> np.ndarray:
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    return np.array(
        [
            [cos_theta, -sin_theta, 0.0],
            [sin_theta, cos_theta, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def validate_temporal_sequence(sequence: np.ndarray) -> None:
    """Valida que la secuencia tenga temporalidad real."""
    if sequence.ndim != 3 or sequence.shape[1:] != EXPECTED_LANDMARK_SHAPE:
        raise ValueError(f"Forma invalida: {sequence.shape}; se esperaba (T, 21, 3)")

    if np.allclose(sequence[0], sequence[-1]):
        raise ValueError("La secuencia no cambia entre el primer y ultimo frame.")

    frame_variance = float(np.var(np.diff(sequence, axis=0)))
    if not np.isfinite(frame_variance) or frame_variance <= 0.0:
        raise ValueError("La secuencia no presenta varianza temporal positiva.")


def generate_temporal_sequence(
    landmarks: np.ndarray,
    T: int = 16,
    sigma: float = 0.015,
    rotation_std_deg: float = 1.5,
    translation_std: float = 0.01,
    scale_std: float = 0.015,
    seed: int | None = None,
    max_attempts: int = 5,
) -> np.ndarray:
    """Genera una secuencia temporal suave a partir de landmarks estaticos.

    La variacion por frame se construye con una trayectoria correlacionada de
    rotacion, traslacion y escala globales. No usa ruido independiente por
    coordenada.
    """
    landmarks = np.asarray(landmarks, dtype=np.float32)
    if landmarks.shape != EXPECTED_LANDMARK_SHAPE:
        raise ValueError(f"Shape invalido de landmarks: {landmarks.shape}")

    rng = np.random.default_rng(seed)
    motion_factor = max(float(sigma) / 0.015, 0.25)
    frame_count = max(1, int(T))
    step_denom = max(1, frame_count - 1)

    center = landmarks.mean(axis=0, keepdims=True)
    centered = landmarks - center

    for _ in range(max_attempts):
        angle_steps = rng.normal(
            0.0,
            np.deg2rad(rotation_std_deg) * motion_factor / step_denom,
            size=frame_count,
        ).astype(np.float32)
        angle_traj = np.cumsum(angle_steps)
        angle_traj += np.linspace(
            0.0,
            rng.normal(0.0, np.deg2rad(rotation_std_deg) * motion_factor * 0.5),
            frame_count,
        )

        translation_steps = rng.normal(
            0.0,
            translation_std * motion_factor / step_denom,
            size=(frame_count, 3),
        ).astype(np.float32)
        translation_traj = np.cumsum(translation_steps, axis=0)
        translation_traj += np.linspace(
            np.zeros(3, dtype=np.float32),
            rng.normal(0.0, translation_std * motion_factor * 0.5, size=3).astype(np.float32),
            frame_count,
        )

        scale_steps = rng.normal(
            0.0,
            scale_std * motion_factor / step_denom,
            size=frame_count,
        ).astype(np.float32)
        scale_traj = 1.0 + np.cumsum(scale_steps)
        scale_traj += np.linspace(0.0, rng.normal(0.0, scale_std * motion_factor * 0.5), frame_count)
        scale_traj = np.clip(scale_traj, 0.95, 1.05)

        sequence = np.empty((frame_count, 21, 3), dtype=np.float32)
        for t in range(frame_count):
            transformed = centered @ _rotation_z(float(angle_traj[t])).T
            transformed = transformed * float(scale_traj[t]) + center + translation_traj[t]
            sequence[t] = transformed.astype(np.float32)

        validate_temporal_sequence(sequence)
        return sequence

    raise RuntimeError("No se pudo generar una secuencia temporal valida.")