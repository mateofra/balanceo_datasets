# ⚖️ Módulo de Balanceo

Contiene la lógica para balancear datasets (FreiHAND + HaGRID) según tonos MST.

## Archivos

- `balancear_freihand_hagrid.py` - Función principal de balanceo

## Funcionalidad

- Combina FreiHAND y HaGRID manteniendo proporciones
- Balancea por tono de piel (MST 1-10)
- Categorización: claro/medio/oscuro
- Genera manifiestos reproducibles
- Conserva validación/test sin rebalance

## Uso

```python
from src.balancer import balancear_freihand_hagrid

manifest = balancear_freihand_hagrid(
    freihand_data=freihand_df,
    hagrid_data=hagrid_df,
    mst_weights={
        'claro': 0.33,
        'medio': 0.33,
        'oscuro': 0.34
    },
    seed=42
)
```

## Salidas

- CSV manifesto editable
- JSON reporte de balanceo
- Estadísticas por parámetro
