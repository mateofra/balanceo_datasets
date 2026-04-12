# 📋 Registros Matriciales en Texto (`/csv`)

En contraparte absoluta con las carpetas invisibles como `/data`, esta carpeta aloja **matrices estructurales puramente tabulares de formato abierto en texto plano**.

Dado su peso nominal en kilobytes y su valía para la auditoría científica histórica, **esta carpeta ES versionada por Git a perpetuidad y debe cuidarse**.

## 🧠 ¿Qué habita aquí?

- **Auditorías Pre-Balanceo (`mst_*.csv`)**: Mapeos masivos indexados por `sample_id` que fungen como diccionario de consulta (la "Verdad Terrenal") uniendo UUIDs a un número demográfico MST para la predicción de fenotipos oscuros/claros de la piel sin necesidad de re-evaluar la imagen real mil veces.
- **Manifestos de Entrada (`train_manifest_*.csv`)**: Las recetas maestras balanceadas salidas de nuestro *Dataset Balancer*. Estas listas son pasadas en frío a los `Dataloaders` en PyTorch.
- **Sets Fraccionados**: Divisiones por cuotas y cortes (como claro/medio/oscuro) preparadas para la ingesta en las ramas asimétricas e iteraciones generativas.

## ⚠️ Reglas Base

1. Jamás elimines un manifiesto CSV aquí si el pipeline ST-GCN aún se entrena derivado de este; romperá los punteros.
2. Todo script de Python en el orquestador (`src/balancer/writers.py`) dirige automáticamente la creación de tablas a esta carpeta como punto de colisión seguro.
