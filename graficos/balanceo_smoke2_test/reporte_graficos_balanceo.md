# Reporte de graficos de balanceo

## Conteos calculados desde manifiesto

- Por fuente: {'freihand': 100, 'hagrid': 100}
- Por bloque MST: {'sin_mst': 200}
- Top gestos: {'unknown': 100, 'stop_inverted': 9, 'dislike': 8, 'one': 8, 'like': 8, 'call': 7, 'stop': 7, 'three': 6, 'two_up_inverted': 6, 'mute': 6, 'three2': 6, 'four': 5}
- Por nivel MST: sin datos MST en manifiesto

## Resumen JSON asociado

```json
{
  "seed": 42,
  "sampling_config": {
    "extreme_mst_levels": [
      1,
      2,
      3,
      10
    ],
    "extreme_factor": 2.0
  },
  "summary": {
    "total_samples": 200,
    "by_source": {
      "freihand": 100,
      "hagrid": 100
    },
    "by_gesture": {
      "unknown": 100,
      "three": 6,
      "two_up": 4,
      "call": 7,
      "stop_inverted": 9,
      "stop": 7,
      "dislike": 8,
      "one": 8,
      "ok": 3,
      "like": 8,
      "two_up_inverted": 6,
      "mute": 6,
      "three2": 6,
      "four": 5,
      "palm": 3,
      "rock": 4,
      "fist": 5,
      "peace": 3,
      "peace_inverted": 2
    },
    "by_mst_block": {
      "sin_mst": 200
    },
    "by_mst_level": {}
  }
}
```