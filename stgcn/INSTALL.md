# Installation & Setup

## Requisitos Mínimos

- Python 3.13+
- pip o uv
- 4 GB RAM
- 200 MB espacio en disco

## Opción 1: Setup Rápido (Recomendado)

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2.Validar setup
uv run python scripts/validate_setup.py

# 3. Entrenar
uv run python scripts/train.py
```

## Opción 2: Setup con `uv` (más rápido)

```bash
# 1. Instalar uv (si no lo tienes)
pip install uv

# 2. Crear entorno
uv venv
source .venv/bin/activate  # Linux/Mac
# o
.venv\Scripts\activate     # Windows

# 3. Instalar dependencias
uv pip install -r requirements.txt

# 4. Validar y entrenar
uv run python scripts/validate_setup.py
uv run python scripts/train.py
```

## Opción 3: GPU (CUDA)

Si tienes GPU NVIDIA:

```bash
# Instalar PyTorch con soporte CUDA
pip uninstall torch -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# O con uv:
uv pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu118
```

Luego:

```bash
uv run python scripts/train.py --device cuda --batch-size 128
```

## Verificar Instalación

```bash
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}')"
uv run python -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}')"
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"

```bash
pip install --upgrade pip
pip install torch numpy tqdm
```

### "CUDA out of memory"

```bash
# Reducir batch size
uv run python scripts/train.py --batch-size 16
```

### "Datos no encontrados"

```bash
uv run python scripts/validate_setup.py
# Ver qué falta y ajustar paths en config/default_config.yaml
```

## Próximo Paso

```bash
uv run python scripts/validate_setup.py
uv run python scripts/train.py
```

¡Listo! Ver `GUIA_RAPIDA.md` para detalles.
