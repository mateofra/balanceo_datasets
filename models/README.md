# 🤖 Modelos Estáticos (`/models`)

A diferencia de `output/training_logs/` (donde orbitan los *checkpoints* volátiles `.pth` y tensores flotantes propios de tu autómata que muta tras cada Batch), esta carpeta funge como un silo de **pesos y descriptores duros y oficiales**. 

## ¿Qué binarios descansan aquí?

Normalmente, contendrá archivos compilados base de inteligencia estandarizada, como:

- `hand_landmarker.task`: El cerebro congelado oficial (en formato TFLite blindado) de Google MediaPipe que utiliza el subsistema de procesamiento contingente para extraer puntos esqueléticos (Landmarks) masivamente sobre nuestra fase de reparación visual. 

> [!TIP]
> Cualquier otro modelo pre-entrenado externo al pipeline central de la STGCN y que utilicemos "Out of the Box" dictaminado (tales como clasificadores genéticos para FairGenderGen, detectores faciales mediapipe auxiliares, discriminadores racializados congelados, etc.) también deberá arrojarse a esta bóveda estática.
