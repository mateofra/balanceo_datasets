# 🛡️ Entorno Demilitarizado de Secretos (`/secrets`)

> [!CAUTION]  
> **NUNCA modifiques el archivo `.gitignore` para saltarte la restricción sobre esta carpeta. NUNCA subas ningún integrante de esta carpeta a repositorios virtuales, repositorios temporales locales accesibles a la red o pastebins.**

Esta bóveda existe únicamente en tu máquina local. Actúa como el cajón seguro donde se almacenan las llaves codificadas para conectar los pipelines automáticos con la capa extraída de internet (API).

## Arquitectura

- `kaggle/`: Subdirectorio enfocado a aislar el Token API del repositorio *HaGRID*.
  - `kaggle.json`: Fichero que debes inyectar tú mismo siguiendo el formato clásico (`{"username":"...", "key":"..."}`) extraído desde la configuración de Kaggle.

Sin estos subficheros los comandos constructivos como `uv run python src/cli/main.py setup-data --download-hagrid` estallarán y te denegarán el acceso, forzándote incesantemente por la terminal a que introduzcas la llave manualmente.
