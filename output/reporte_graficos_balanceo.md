# Reporte de graficos de balanceo

## Conteos calculados desde manifiesto

- Por fuente: {'freihand': 10000, 'hagrid': 10000}
- Por bloque MST: {'medio': 6666, 'claro': 6666, 'oscuro': 6668}
- Top gestos: {'unknown': 10000, 'two_up': 616, 'stop': 594, 'call': 584, 'palm': 579, 'rock': 578, 'like': 568, 'peace_inverted': 564, 'one': 558, 'dislike': 558, 'three': 556, 'two_up_inverted': 554}
- Por nivel MST: {'1': 1873, '2': 1945, '3': 1888, '4': 960, '5': 2174, '6': 2307, '7': 2185, '8': 1600, '9': 1661, '10': 3407}

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
    "dark_jitter_factor": 0.0
  },
  "summary": {
    "total_samples": 20000,
    "by_source": {
      "freihand": 10000,
      "hagrid": 10000
    },
    "by_gesture": {
      "unknown": 10000,
      "call": 584,
      "ok": 548,
      "four": 508,
      "one": 558,
      "peace_inverted": 564,
      "two_up": 616,
      "rock": 578,
      "dislike": 558,
      "two_up_inverted": 554,
      "like": 568,
      "three": 556,
      "three2": 539,
      "fist": 523,
      "mute": 534,
      "stop": 594,
      "stop_inverted": 532,
      "peace": 507,
      "palm": 579
    },
    "by_mst_block": {
      "medio": 6666,
      "claro": 6666,
      "oscuro": 6668
    },
    "by_mst_level": {
      "1": 1873,
      "2": 1945,
      "3": 1888,
      "4": 960,
      "5": 2174,
      "6": 2307,
      "7": 2185,
      "8": 1600,
      "9": 1661,
      "10": 3407
    },
    "by_mst_origin": {
      "imputed": 20000
    }
  }
}
```