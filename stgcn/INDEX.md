# 📦 ST-GCN Training - Índice Completo

## 🎯 Para Empezar (Elige una opción)

**Opción A: Tengo 5 minutos**
→ Leer: [`GUIA_RAPIDA.md`](GUIA_RAPIDA.md) (quick start en 5 pasos)

**Opción B: Tengo 15 minutos**
→ Leer: [`README.md`](README.md) (documentación completa)

**Opción C: Quiero instalar y ejecutar ahora**
→ Leer: [`INSTALL.md`](INSTALL.md) (setup + troubleshooting)

---

## 📚 Documentación

| Archivo | Contenido | Para Quién |
|---------|----------|-----------|
| [`README.md`](README.md) | Visión general, features, roadmap | Todos |
| [`GUIA_RAPIDA.md`](GUIA_RAPIDA.md) | 5 pasos rápidos + ejemplos código | Apurados |
| [`INSTALL.md`](INSTALL.md) | Setup de dependencias + troubleshooting | Nuevos usuarios |
| [`data/README.md`](data/README.md) | Cómo obtener/vincular datos | Usuarios avanzados |

---

## 🔧 Scripts Disponibles

| Script | Qué hace | Cuándo usar |
|--------|----------|-----------|
| `scripts/validate_setup.py` | Verifica que todo esté instalado | **PRIMERO SIEMPRE** |
| `scripts/train.py` | Entrena modelo ST-GCN | Training |
| `scripts/analyze_fairness.py` | Analiza sesgo por tono MST | Post-training |

**Ejecutar en orden**:

```bash
# 1. VALIDAR
python scripts/validate_setup.py

# 2. ENTRENAR
python scripts/train.py

# 3. ANALIZAR (opcional)
python scripts/analyze_fairness.py logs/training_log_final.json
```

---

## 💻 Código Principal

| Módulo | Función | Líneas |
|--------|---------|--------|
| `src/dataloader.py` | Carga datos + balanceo MST | ~180 |
| `scripts/train.py` | Loop de training | ~240 |
| `scripts/validate_setup.py` | Verificaciones | ~100 |

---

## 🚀 Quick Commands

```bash
# Setup (primera vez)
pip install -r requirements.txt

# Validar
python scripts/validate_setup.py

# Training (CPU, ~10 min)
python scripts/train.py

# Training (GPU, ~2 min)
python scripts/train.py --device cuda --batch-size 128

# Prueba rápida (30 seg)
python scripts/train.py --num-epochs 1 --batch-size 64
```

---

## 📊 Datos Incluidos

**Manifiesto CSV** (10,000 muestras)
- Ubicación en repo: `../output/train_manifest_stgcn_fixed.csv`
- Copiar a: `data/train_manifest_stgcn_fixed.csv` (opcional)

**Landmarks .npy** (10,000 archivos)
- Ubicación en repo: `../data/processed/landmarks/freihand_*.npy`
- Tamaño: ~12 MB

Ver: `data/README.md` para instrucciones de descarga/copia

---

## 🎓 Flujo Típico de Uso

```
1. INSTALAR
   pip install -r requirements.txt
   ↓
2. VALIDAR
   python scripts/validate_setup.py ← Ver que todo OK
   ↓
3. ENTRENAR
   python scripts/train.py ← Training automático
   ↓
4. VER RESULTADOS
   logs/checkpoints/model_final.pth ← Modelo guardado
   logs/training_log_final.json ← Métricas
   ↓
5. ANÁLISIS (OPCIONAL)
   python scripts/analyze_fairness.py logs/training_log_final.json
```

---

## ⚙️ Configuración

Archivo por defecto: `config/default_config.yaml`

```yaml
BATCH_SIZE: 32
NUM_EPOCHS: 20
LEARNING_RATE: 0.001
BALANCE_BY_MST: true  # ⭐ Balancea por tono
DEVICE: "cpu"         # Cambiar a "cuda" si tienes GPU
```

Pasar argumentos CLI:

```bash
python scripts/train.py --batch-size 64 --num-epochs 30 --learning-rate 0.0005
```

---

## 🔗 Relación con Repo Principal

Este directorio está **separado pero vinculado** a `balanceo_datasets/`:

```
balanceo_datasets/
├── src/                          ← Scripts de generación de datos
├── output/                        ← Manifiesos generados
│   └── train_manifest_stgcn_fixed.csv  ← (Referenciado aquí)
├── data/processed/landmarks/     ← (Referenciado aquí)
└── stgcn/                        ← (ESTE DIRECTORIO)
    ├── README.md                 ← Start here
    ├── GUIA_RAPIDA.md
    ├── INSTALL.md
    ├── src/dataloader.py         ← Copia adaptada
    └── scripts/train.py
```

**Ventajas**:
✅ Starter pack es **independiente** (puede moverse a otro lado)
✅ Pero usa datos **generados por balanceo_datasets**
✅ Sin repetir código

---

## 📈 Resultados Esperados

Con configuración por defecto (20 epochs, CPU):

```
Epoch 20/20 | Loss: 0.234 | Accuracy: 93.1%

Outputs:
- logs/checkpoints/model_final.pth (modelo guardado)
- logs/training_log_final.json (métricas)

Fairness (Accuracy por Tono MST):
  MST 1-3 (claro): 92-93% ✅ Equilibrado
  MST 5-7 (medio): 93% ✅ Equilibrado
  MST 8-10 (oscuro): 92-93% ✅ Equilibrado
```

---

## 🆘 Dónde Buscar Ayuda

| Problema | Ver |
|----------|-----|
| Setup no funciona | `INSTALL.md` → Troubleshooting |
| No entiendo los datos | `README.md` → Dataset Disponible |
| Cómo modificar training | `GUIA_RAPIDA.md` → Flujo Completo |
| Qué significan las métricas | `README.md` → Tips Avanzados |
| Datos no encontrados | `data/README.md` |

---

## ✅ Checklist Antes de Empezar

- [ ] Python 3.13+ instalado
- [ ] Leído `INSTALL.md` o `GUIA_RAPIDA.md`
- [ ] `validate_setup.py` ejecutado sin errores
- [ ] Decidido si usar CPU o GPU
- [ ] Datos (manifiesto + landmarks) accesibles
