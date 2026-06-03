# Portal de herramientas LLM

Portal web estático para apoyar pruebas de implementación de la tesis. Incluye:

- Calculadora de embeddings y similitud coseno.
- Evaluador de plantillas LLM para operadores semánticos de movimiento hacia `pbest`/`gbest`, turbulencia o un target fijo.

## Ejecución local

Desde la raíz del repositorio:

```powershell
py .\server.py
```

Luego abrir:

```text
http://127.0.0.1:4173/LLM/
```

## LM Studio

El evaluador usa por defecto el endpoint nativo recomendado por LM Studio 0.4.x:

```text
http://127.0.0.1:1234/api/v1/chat
```

El portal se sirve con `server.py`. Las secciones propias de la web llaman a LM Studio a través de la API local `POST /api/lm-studio/chat` y `POST /api/lm-studio/models`, implementada en Python con `llm_studio.py`. El proxy `/lmstudio/...` queda disponible por compatibilidad, pero no es la ruta común usada por los módulos principales.

Cada ejecución se envía como una solicitud independiente con un único input. Se usa `store: false` en la API nativa para no conservar historial entre ejecuciones y aproximar chats separados.

La lista de modelos se obtiene desde:

```text
http://127.0.0.1:1234/api/v1/models
```

La herramienta también permite cambiar a modo OpenAI-compatible, usando `/v1/models` y `/v1/chat/completions`.

## Evaluación semántica

Para cada ejecución:

1. Renderiza la plantilla con los valores configurados.
2. Llama al LLM local.
3. Parsea hasta `num_candidates` líneas.
4. Calcula embeddings del target y de cada candidato.
5. Calcula similitud coseno y distancia coseno.
6. Marca éxito válido si el candidato supera `cos(current, target) + eta`, respeta el largo configurado, no copia exactamente `current` ni `target`, y no supera `tau_target_copy`.

La herramienta reporta tasa de éxito válida, mejor similitud promedio, mejora promedio, candidatos válidos y resultados por ejecución. Una ejecución sin candidatos parseables cuenta como ejecución completada sin éxito, no se excluye del denominador.

El umbral `tau_target_copy` queda en `0.94` por defecto. Si `cos(candidate, target) >= tau_target_copy`, el candidato se etiqueta como `semantic_copy_target` y no cuenta como éxito.

Las líneas duplicadas no se descartan silenciosamente: se conservan en la tabla y se marcan como `duplicate_output`.

## Prueba de evaluación de soluciones

La sección `Probar evaluación de soluciones` ejecuta un flujo completo:

1. Genera soluciones mock con `rol`, `topico` y `accion` a partir de textos de referencia breves publicados en la memoria de Nicolas Meneses o el paper EVOLMD-MO.
2. Renderiza un prompt mediante la plantilla determinística configurable.
3. Envía el prompt al LLM generador con el system prompt fijo de generación de texto.
4. Calcula embeddings con el runtime Python compartido `all-MiniLM-L6-v2` por defecto o `thenlper/gte-small`.
5. Calcula el vector F.O `[F1, F2]`, donde `F1 = cos(texto generado, referencia)` y `F2` es la distancia coseno promedio contra las demás soluciones.
6. Marca las soluciones no dominadas usando dominancia de Pareto sobre maximización de `F1` y `F2`.

Las estimaciones comparativas de costo usan el tiempo promedio local medido en la prueba. Para EVOLMD-MO se estima Init Agent + Data Agent por solución y mutación esperada con `Pm = 0.05`. Para MESAP se estima la inicialización descrita en la memoria con 3 llamadas por solución y un sobrecosto temporal de 18%.

## Comparador de propuestas

La sección `Comparador de propuestas` integra baselines clonados como submodules:

- `baselines/external/evolmd`
- `baselines/external/evolmd-mo`
- `C:\Users\Admin\Desktop\Implementación\Binary MOPSO-CD`

Los submodules EVOLMD y EVOLMD-MO deben apuntar a los forks `escmHEX/MDPI-EVOLMD` y `escmHEX/EVOLMD-MO`.

La configuración editable del comparador vive en `baselines/comparator_config.json`. Ahí se definen rutas de repositorios, Python por propuesta, rutas extra de `PYTHONPATH`, módulos requeridos, defaults comunes y modelo SBERT post-hoc. Los flags propios de cada propuesta se declaran en su adaptador y se configuran desde `proposalConfigs[proposalId].cliValues`. `COMPARATOR_CONFIG_PATH` permite usar otro archivo de configuración sin modificar código.

Binary MOPSO-CD se ejecuta desde su repositorio real y lee `pareto_front.json`, `final_selection_hybrid.json`, `evolucion_metricas.csv` y `llm_calls.jsonl`, normalizando F.O. como `[objectives.f1, objectives.f2]`.

`selectedProposalIds` permite ejecutar una, varias o todas las propuestas. La interfaz visual construye `proposalConfigs[proposalId].cliValues` con inputs, selects y checkboxes por flag configurable; el backend traduce esos valores a argumentos CLI. `extraArgs` se conserva solo como compatibilidad interna. Salida, texto de referencia, semilla, repeticiones y flags globales siguen gestionados por el comparador para mantener aislamiento y comparabilidad.

Los costos se reportan separados en algoritmo Python, llamadas LLM, post-procesamiento externo de seleccion y extraccion/preparacion de metricas para visualizacion.

La web llama a la API local de `server.py`:

- `GET /api/comparator/proposals`
- `POST /api/comparator/runs`
- `GET /api/comparator/runs/{runId}`
- `POST /api/comparator/runs/{runId}/cancel`

El backend ejecuta los `main.py` originales con Ollama local, usando `http://127.0.0.1:11434` tal como esperan los clones. Cada corrida se guarda en `runs/comparator/<runId>/`, junto con un `summary.json` normalizado.

Los adaptadores propios están fuera de los submodules, en `baselines/comparator.py`. EVOLMD lee `data_final_evaluada.json` y normaliza F.O. como `[fitness]`. EVOLMD-MO lee `pareto_front.json` y normaliza F.O. como `[fidelity_sbert, diversity_individual]`.

Para EVOLMD, el adaptador calcula diversidad semántica post-hoc sobre los textos finales con SBERT `all-MiniLM-L6-v2`. Ese cálculo no cambia selección, cruce, mutación ni supervivencia de EVOLMD. Se reporta como vector diagnóstico `[fitness, semantic_diversity_posthoc]` y permite mostrar HV/spread post-hoc separados del objetivo original single-objective.

El lanzamiento pasa por `baselines/bootstrap.py`, que precarga los módulos declarados por cada adaptador antes de ejecutar el `main.py` original. Esto evita problemas de orden de carga de PyTorch en Windows sin modificar los clones.

La API devuelve progreso estructurado en `progress` y `proposalStates`. La web muestra porcentaje, etapa activa, tiempo transcurrido, tiempo restante estimado y estado por propuesta. La cancelación termina el árbol de procesos activo en Windows para cortar llamadas largas sin esperar nuevos logs.

Los costos de ejecución se devuelven en `costSummary` a nivel de corrida y en `proposal.cost` por baseline. Se mide wall-clock del proceso, runtime reportado por `runtime.txt`, llamadas reales a Ollama interceptadas desde `baselines/bootstrap.py`, tiempo cliente acumulado de esas llamadas, duración reportada por Ollama cuando está disponible, tokens `prompt_eval_count`/`eval_count`, promedio por llamada y llamadas fallidas.

`proposalParallelism` controla cuántas propuestas se ejecutan al mismo tiempo desde el adaptador. No modifica el paralelismo interno de EVOLMD ni EVOLMD-MO, porque ese comportamiento queda dentro de los clones upstream. `timeoutMinutes` limita la duración máxima por propuesta.

Para EVOLMD-MO se usa maximización, fidelidad normalizada con `(fidelity + 1) / 2`, diversidad acotada a `[0, 1]`, punto de referencia HV `[0, 0]` y spread como desviación normalizada entre distancias consecutivas del frente no dominado. Para EVOLMD, HV/spread son diagnósticos post-hoc sobre `[fitness, semantic_diversity_posthoc]`, con ambos valores acotados a `[0, 1]`.

Antes de ejecutar una comparación real, configura en `baselines/comparator_config.json` el `pythonExecutable` correcto para cada propuesta o sus `pythonPathEntries`. La API `GET /api/comparator/proposals` reporta dependencias faltantes por propuesta antes de permitir seleccionarlas. Mantén Ollama corriendo con el modelo elegido.

## Simulación de iteración PSO

La sección `Componente semántico` incluye el switch `Simular iteración PSO`.

Cuando está activo:

1. Carga `LLM/data/pso-individuals.json`, con 200 individuos.
2. Toma los primeros `N individuos` configurados.
3. Recorre `role`, `topic` y `action` de cada individuo.
4. Sortea cada componente y la selecciona para cambio si `random > umbral sorteo cambio`.
5. Aplica el operador semántico elegido: influencia hacia `pbest`/`lider` o turbulencia.
6. Renderiza la plantilla usando el componente actual, los otros componentes del individuo, el texto de referencia fijo y, solo para influencia, el objetivo elegido.
7. Solicita `num_candidates` candidatos al LLM.
8. Aplica un candidato válido según el criterio del operador.

### Operador influencia pbest/gbest

Para cada componente sorteada, el objetivo se elige al azar entre `pbest` y `lider`.

Un candidato es válido si:

- Respeta largo mínimo y máximo.
- No copia literalmente `current` ni `target`.
- No está duplicado en la misma respuesta.
- Cumple `cos(candidate, target) > cos(current, target) + eta`.
- Cumple `cos(candidate, target) < tau_target_copy`.

Entre los candidatos válidos, la herramienta aplica el de mayor similitud con el objetivo.

### Operador turbulencia

Este operador no usa `target`. Usa `current` como ancla semántica para exigir exploración controlada.

Un candidato es válido si:

- Respeta largo mínimo y máximo.
- No copia literalmente `current`.
- No está duplicado en la misma respuesta.
- Cumple `turbulence_min <= cos(candidate, current) <= turbulence_max`.

Por defecto, el rango de turbulencia queda en `[0.55, 0.9]`. Bajo `0.55` se considera demasiado lejano al tópico; sobre `0.9` se considera demasiado parecido al componente actual para aportar diversidad.

Entre los candidatos válidos, la herramienta aplica uno al azar para preservar el carácter exploratorio del operador.

Los componentes de un mismo individuo se procesan en orden. Si `role` cambia, los prompts posteriores de `topic` y `action` usan ese `role` actualizado dentro de `Otros componentes`.

El peor caso realiza `3 * N` llamadas al LLM. Con `N = 200`, eso puede llegar a 600 llamadas.

Las métricas específicas del modo PSO reportan componentes sorteadas, componentes que no pudieron cambiar, tasa de no cambio, cambios aplicados, individuos modificados, llamadas LLM, tiempo promedio por llamada, tiempo total LLM, candidatos válidos, fallos por candidatos no aceptables y errores técnicos de ejecución. Los tiempos son wall-clock por request; el tiempo total LLM es la suma de llamadas y puede ser mayor que la duración real de la corrida si hay paralelismo. En la tabla de candidatos, `Referencia` es el target para influencia y el current para turbulencia.

## Embeddings

Los embeddings de las secciones propias del portal usan Python `sentence-transformers` con `normalize_embeddings=True`, centralizado en `sbert_service.py`. La calculadora usa `POST /api/sbert/pair`; los flujos que necesitan vectores usan `POST /api/sbert/embeddings`. La comparación de turbulencia, la evaluación de plantillas, la prueba de soluciones y las métricas comunes backend quedan sobre el mismo runtime.

Modelos del runtime compartido:

- `all-MiniLM-L6-v2` (`sentence-transformers/all-MiniLM-L6-v2`)
- `thenlper/gte-small`

Los alias antiguos `Xenova/all-MiniLM-L6-v2` y `Xenova/gte-small` se aceptan solo por compatibilidad y se redirigen al runtime Python equivalente.
