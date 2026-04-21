import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

def get_angle(v1, v2):
    # Ángulo en grados entre dos vectores
    unit_v1 = v1 / (np.linalg.norm(v1) + 1e-6)
    unit_v2 = v2 / (np.linalg.norm(v2) + 1e-6)
    dot_product = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)
    return np.degrees(np.arccos(dot_product))

def normalize_orientation(landmarks):
    """
    Alinea la mano al eje vertical canónico.
    landmarks: array (21, 2) con coordenadas (x, y) normalizadas [0,1]
    """
    # Eje principal: muñeca (0) → base dedo medio (9)
    wrist = landmarks[0]
    mid_base = landmarks[9]
    axis = mid_base - wrist

    # Ángulo respecto al eje vertical (0, -1) — y invertida en imagen
    angle = np.arctan2(axis[0], -axis[1])

    # Matriz de rotación inversa
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    R = np.array([[cos_a, -sin_a],
                  [sin_a,  cos_a]])

    # Centrar en muñeca, rotar, descentrar
    centered = landmarks - wrist
    rotated = (R @ centered.T).T
    return rotated + wrist

def calculate_advanced_features(lms):
    # lms: (21, 3)
    # Fingers: Thumb(1-4), Index(5-8), Middle(9-12), Ring(13-16), Pinky(17-20)
    
    # 1. Ángulos MCP-PIP por dedo (Proximal Flexion)
    # Vector MCP->PIP y PIP->DIP
    angles_prox = []
    angles_dist = []
    tips = [4, 8, 12, 16, 20]
    pips = [2, 6, 10, 14, 18]
    mcps = [1, 5, 9, 13, 17]
    dips = [3, 7, 11, 15, 19]
    
    for m, p, d, t in zip(mcps, pips, dips, tips):
        v1 = lms[p] - lms[m]
        v2 = lms[d] - lms[p]
        angles_prox.append(get_angle(v1, v2))
        
        v3 = lms[t] - lms[d]
        angles_dist.append(get_angle(v2, v3))
        
    # 2. Distancia Índice-Pulgar (OK detection)
    dist_thumb_index = np.linalg.norm(lms[4] - lms[8])
    # Normalizar por la longitud del hueso metacarpiano del índice
    scale = np.linalg.norm(lms[5] - lms[0]) + 1e-6
    dist_ok = dist_thumb_index / scale
    
    # 3. Abducción del pulgar
    # Ángulo entre el vector Wrist->ThumbMCP y Wrist->IndexMCP
    v_thumb = lms[1] - lms[0]
    v_index = lms[5] - lms[0]
    thumb_abduction = get_angle(v_thumb, v_index)
    
    # 4. Curvatura de la palma (Arco)
    # Distancia entre MCP del índice y MCP del meñique vs distancia sumada de huesos
    dist_palm_width = np.linalg.norm(lms[17] - lms[5])
    palm_arc = dist_palm_width / scale

    # 5. Elevación del pulgar relativa a nudillos (Y-axis)
    # En coordenadas MediaPipe, Y disminuye hacia arriba.
    knuckles_y = np.mean([lms[5,1], lms[9,1], lms[13,1], lms[17,1]])
    thumb_up_score = knuckles_y - lms[4,1] # Positivo si el pulgar está arriba

    return {
        'angles_prox': angles_prox,
        'angles_dist': angles_dist,
        'dist_ok': dist_ok,
        'thumb_abduction': thumb_abduction,
        'palm_arc': palm_arc,
        'thumb_up_score': thumb_up_score
    }

def suggest_refined_label(f):
    # f: dict de features
    prox = f['angles_prox'] # [T, I, M, R, P]
    dist = f['angles_dist']
    
    # Heurísticas de flexión (ángulo > 30 grados se considera doblado)
    is_bent = [a > 35 for a in prox]
    is_very_bent = [a > 60 for a in prox]
    
    # 1. Preparar discriminadores específicos
    thumb_is_up = f.get('thumb_up_score', 0) > 0.05
    thumb_is_bent = prox[0] > 45

    # CLASES
    
    # LIKE: Pulgar arriba y resto muy cerrados
    if thumb_is_up and all(is_very_bent[1:]):
        return "like", 0.98

    # FIST: Todos los dedos (incluido pulgar) muy doblados
    if all(is_very_bent[1:]) and thumb_is_bent:
        return "fist", 0.95
        
    # OK: Pulgar e índice se tocan, otros estirados
    if f['dist_ok'] < 0.4 and not any(is_bent[2:]):
        return "ok", 0.9

    # PEACE: Índice y medio estirados, otros doblados
    if not is_bent[1] and not is_bent[2] and all(is_bent[3:]):
        return "peace", 0.85
        
    # ONE: Solo índice estirado
    if not is_bent[1] and all(is_bent[2:]):
        return "one", 0.85
        
    # PALM: Todos estirados
    if not any(is_bent[1:]):
        return "palm", 0.9
        
    # THREE / FOUR
    bent_count = sum(is_bent[1:])
    if bent_count == 1: return "four", 0.7
    if bent_count == 2: return "three", 0.7

    return "unknown", 0.2

def classify_unknown_by_cluster(features):
    dist_ok = features['dist_ok']
    thumb_abd = features['thumb_abduction']
    prox_I = features['angles_prox'][1]
    prox_M = features['angles_prox'][2]
    prox_R = features['angles_prox'][3]
    prox_P = features['angles_prox'][4]

    # Cluster 2: outlier / bad_pose
    if dist_ok > 5.0:
        return 'bad_pose', 0.50

    # Cluster 3 y 4: pulgar muy abducido
    if thumb_abd > 100:
        # Rock: pulgar abierto + meñique estirado
        if prox_P < 50:
            return 'rock', 0.72
        # Call: pulgar abierto + índice flexionado + medio flexionado
        if prox_I > 110 and prox_M > 95:
            return 'call', 0.68
        # Like: pulgar abierto, resto cerrado
        if prox_I > 100 and prox_M > 80:
            return 'like', 0.60

    # Cluster 0: peace / two_up — medio menos flexionado
    if prox_M < 50 and prox_I < 120:
        return 'two_up', 0.65

    # Cluster 1: rock / pinky — meñique estirado
    if prox_P < 65 and thumb_abd < 60:
        return 'rock', 0.62

    return 'unknown', 0.0

def run_refined_suggestion():
    manifest_path = Path("output/train_manifest_stgcn_secuencias.csv")
    df = pd.read_csv(manifest_path)
    mano_df = df[df['dataset'] == 'mano'].copy()
    
    print(f"🧠 Refinando sugerencias para {len(mano_df)} muestras MANO...")
    
    results = []
    for idx, row in tqdm(mano_df.iterrows(), total=len(mano_df)):
        try:
            lms = np.load(row['path_landmarks'])
            
            # Normalizar orientación (2D)
            lms_xy = lms[:, :2]
            lms_rotated_xy = normalize_orientation(lms_xy)
            lms_final = lms.copy()
            lms_final[:, :2] = lms_rotated_xy
            
            features = calculate_advanced_features(lms_final)
            label, conf = suggest_refined_label(features)
            
            # Fallback post-clustering para unknowns
            if label == "unknown":
                new_label, new_conf = classify_unknown_by_cluster(features)
                if new_label != "unknown":
                    label, conf = new_label, new_conf
            
            res = {
                'sample_id': row['sample_id'],
                'suggested_label': label,
                'confidence': conf,
                'path_landmarks': row['path_landmarks'],
                'dist_ok': features['dist_ok'],
                'thumb_abduction': features['thumb_abduction'],
                'palm_arc': features['palm_arc']
            }
            # Añadir ángulos individuales para auditoría
            for i, name in enumerate(['T', 'I', 'M', 'R', 'P']):
                res[f'prox_{name}'] = features['angles_prox'][i]
            
            results.append(res)
        except Exception as e:
            continue
            
    output_df = pd.DataFrame(results)
    output_path = Path("output/mano_refined_suggestions.csv")
    output_df.to_csv(output_path, index=False)
    
    print(f"✅ Sugerencias refinadas guardadas en {output_path}")
    print(output_df['suggested_label'].value_counts())

if __name__ == "__main__":
    run_refined_suggestion()
