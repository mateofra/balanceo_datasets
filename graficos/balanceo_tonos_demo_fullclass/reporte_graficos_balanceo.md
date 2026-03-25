# Reporte de graficos de balanceo

## Conteos calculados desde manifiesto

- Por fuente: {'hagrid': 158, 'freihand': 157}
- Por bloque MST: {'oscuro': 115, 'claro': 100, 'medio': 100}
- Top gestos: {'unknown': 157, 'two_up_inverted': 15, 'fist': 14, 'rock': 13, 'mute': 13, 'dislike': 12, 'call': 11, 'ok': 11, 'one': 10, 'stop_inverted': 9, 'four': 8, 'three': 8}
- Por nivel MST: {'1': 34, '2': 23, '3': 28, '4': 15, '5': 35, '6': 43, '7': 22, '8': 26, '9': 40, '10': 49}

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
    "extreme_factor": 2.0,
    "dark_jitter_factor": 0.3
  },
  "summary": {
    "total_samples": 315,
    "by_source": {
      "hagrid": 158,
      "freihand": 157
    },
    "by_gesture": {
      "stop_inverted": 9,
      "two_up_inverted": 15,
      "like": 5,
      "peace": 7,
      "three2": 3,
      "unknown": 157,
      "dislike": 12,
      "call": 11,
      "four": 8,
      "fist": 14,
      "ok": 11,
      "peace_inverted": 7,
      "three": 8,
      "rock": 13,
      "palm": 3,
      "mute": 13,
      "two_up": 5,
      "one": 10,
      "stop": 4
    },
    "by_mst_block": {
      "oscuro": 115,
      "claro": 100,
      "medio": 100
    },
    "by_mst_level": {
      "1": 34,
      "2": 23,
      "3": 28,
      "4": 15,
      "5": 35,
      "6": 43,
      "7": 22,
      "8": 26,
      "9": 40,
      "10": 49
    },
    "by_mst_origin": {
      "imputed": 296,
      "original": 19
    }
  }
}
```