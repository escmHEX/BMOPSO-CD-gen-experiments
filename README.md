# Experimentos BMOPSO-CD

Portal local para evaluar soluciones, generar poblacion inicial, comparar estrategias de inicializacion, ejecutar turbulencia y correr el comparador de propuestas.

## Clonar

```bash
git clone --recurse-submodules https://github.com/escmHEX/BMOPSO-CD-gen-experiments.git
cd BMOPSO-CD-gen-experiments
git checkout dev
git submodule update --init --recursive
```

## Antes de ejecutar el instalador

- Tener red y espacio suficiente para modelos, entornos Python y PPDB.
- Configurar Kaggle API para descargar PPDB.
- En Windows, tener Git instalado y `winget` disponible.
- En Ubuntu no necesitas `sudo` si el servidor ya trae `git` y `curl`. Usa `--no-sudo` para instalar Ollama dentro del repo.
- Si en Ubuntu faltan herramientas base como `git` o `curl`, el instalador fallara antes de descargar dependencias.

Los scripts instalan o preparan automaticamente `uv`, Python 3.13, los venvs, dependencias Python, Kaggle CLI, Ollama cuando falta, modelos Ollama y modelos Python. En modo sin `sudo`, Ollama queda en `.local/ollama` y sus modelos en `.local/ollama-models`.

## Kaggle API

Antes de ejecutar el instalador, copia tu token de Kaggle y configura una de estas opciones.

Ubuntu:

```bash
export KAGGLE_API_TOKEN=<TU_TOKEN_KAGGLE>
```

O guardalo para que Kaggle CLI lo lea automaticamente:

```bash
mkdir -p ~/.kaggle
printf '%s' '<TU_TOKEN_KAGGLE>' > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

Windows PowerShell:

```powershell
$env:KAGGLE_API_TOKEN = "<TU_TOKEN_KAGGLE>"
```

Tambien se mantiene soporte para el formato clasico `~/.kaggle/kaggle.json` o las variables `KAGGLE_USERNAME` y `KAGGLE_KEY`.

## Ubuntu

```bash
bash scripts/install_ubuntu.sh
bash scripts/start_server_daemon_ubuntu.sh start --host 0.0.0.0
bash scripts/verify_install.py --base-url http://127.0.0.1:4173
```

Sin acceso a `sudo`:

```bash
bash scripts/install_ubuntu.sh --no-sudo
bash scripts/start_server_daemon_ubuntu.sh start --host 0.0.0.0 --nohup
bash scripts/verify_install.py --base-url http://127.0.0.1:4173
```

El instalador crea `.venv`, prepara venvs aislados para EVOLMD, EVOLMD-MO, MESAP y Binary MOPSO-CD, clona Binary en `baselines/external/binary-mopso-cd`, descarga PPDB desde Kaggle, genera `data/turbulence/ppdb_index.json`, instala Ollama si falta y precarga modelos.

## Windows

```powershell
.\scripts\install_windows.ps1
.\scripts\start_server_daemon_windows.ps1 -Action Start -HostName 0.0.0.0
.\.venv\Scripts\python.exe scripts\verify_install.py --base-url http://127.0.0.1:4173
```

El daemon de Windows usa Task Scheduler. El daemon de Ubuntu usa `systemd --user` cuando esta disponible y `nohup` como fallback. Sin `sudo`, no intenta configurar `loginctl enable-linger`; si el entorno cierra procesos al terminar la sesion, usa `--nohup`, `systemd --user` con linger ya habilitado o el gestor de procesos disponible.

## Servidor manual

```bash
python server.py
python server.py --port 4174
```

El puerto por defecto es `4173`. El runtime LLM por defecto es Ollama OpenAI-compatible:

- Base URL: `http://127.0.0.1:11434`
- API mode: `openai`
- Modelo: `llama3`

LM Studio sigue siendo usable manualmente seleccionando modo `native` y base URL `http://127.0.0.1:1234`.

## Verificacion

`scripts/verify_install.py` comprueba salud del portal, Ollama, chat LLM, SBERT, PPDB, poblacion inicial, comparacion inicial y comparador de propuestas con EVOLMD, EVOLMD-MO, MESAP y Binary disponibles.

Para comprobar que el servidor quedo funcionando, ejecuta este comando despues de iniciar el daemon:

Ubuntu:

```bash
bash scripts/verify_install.py --base-url http://127.0.0.1:4173
```

Windows:

```powershell
.\.venv\Scripts\python.exe scripts\verify_install.py --base-url http://127.0.0.1:4173
```

Los siguientes comandos son validacion de codigo para desarrollo. Ejecutalos al final solo si modificaste el repo y quieres comprobar tests/whitespace antes de commitear:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_server_restart tests.test_turbulence_comparison tests.test_sbert_service tests.test_repetition_aggregation
git diff --check
```
