# Experimentos BMOPSO-CD

Portal local para evaluar soluciones, generar poblacion inicial, comparar estrategias de inicializacion, ejecutar turbulencia y correr el comparador de propuestas.

## Clonar

```bash
git clone --recurse-submodules https://github.com/escmHEX/BMOPSO-CD-gen-experiments.git
cd BMOPSO-CD-gen-experiments
git checkout dev
git submodule update --init --recursive
```

## Requisitos

- Git.
- Red y espacio suficiente para modelos, entornos Python y PPDB.
- Python 3.13 gestionado por `uv` (los instaladores lo preparan).
- Kaggle API para PPDB: `~/.kaggle/kaggle.json` o `KAGGLE_USERNAME` / `KAGGLE_KEY`.
- Permisos para instalar/ejecutar Ollama.

## Ubuntu

```bash
bash scripts/install_ubuntu.sh
bash scripts/start_server_daemon_ubuntu.sh start --host 0.0.0.0
bash scripts/verify_install.py --base-url http://127.0.0.1:4173
```

El instalador crea `.venv`, prepara venvs aislados para EVOLMD, EVOLMD-MO, MESAP y Binary MOPSO-CD, clona Binary en `baselines/external/binary-mopso-cd`, descarga PPDB desde Kaggle, genera `data/turbulence/ppdb_index.json`, instala Ollama si falta y precarga modelos.

## Windows

```powershell
.\scripts\install_windows.ps1
.\scripts\start_server_daemon_windows.ps1 -Action Start -HostName 0.0.0.0
.\.venv\Scripts\python.exe scripts\verify_install.py --base-url http://127.0.0.1:4173
```

El daemon de Windows usa Task Scheduler. El daemon de Ubuntu usa `systemd --user` cuando esta disponible y `nohup` como fallback.

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

Validacion de codigo local:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_server_restart tests.test_turbulence_comparison tests.test_sbert_service tests.test_repetition_aggregation
git diff --check
```
