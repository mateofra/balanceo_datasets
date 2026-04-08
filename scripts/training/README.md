# 🎓 Scripts de Entrenamiento

Scripts para entrenar modelos y visualizar la pipeline.

## Scripts

| Script | Propósito |
|--------|-----------|
| `train_stgcn.py` | Entrena modelo ST-GCN con datos balanceados |
| `generar_secuencias_stgcn.py` | Pre-genera secuencias para reducir overhead en training |
| `visualizar_pipeline.py` | Visualiza el flujo de datos en la pipeline |

## Uso

```bash
# Entrenar ST-GCN
python train_stgcn.py --config config.yaml

# Generar secuencias previo a training
python generar_secuencias_stgcn.py

# Visualizar pipeline
python visualizar_pipeline.py
```

## Logs

Los logs de entrenamiento se guardan en `output/training_logs/`

## Configuración

Consulta `config/` para parámetros de training.
