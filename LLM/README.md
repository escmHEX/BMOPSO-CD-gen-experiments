# Portal de herramientas LLM

Portal web estático para apoyar pruebas de implementación de la tesis. Incluye:

- Calculadora de embeddings y similitud coseno.
- Evaluador de plantillas LLM para operadores semánticos de movimiento hacia `pbest`, `gbest` o un target fijo.

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

El portal se sirve con `server.py`, que además expone un proxy local en `/lmstudio/...`. Esto evita errores CORS del navegador al llamar a LM Studio desde otro puerto.

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

La herramienta reporta tasa de éxito válida, mejor similitud promedio, mejora promedio, candidatos válidos y resultados por ejecución.

El umbral `tau_target_copy` queda en `0.94` por defecto. Si `cos(candidate, target) >= tau_target_copy`, el candidato se etiqueta como `semantic_copy_target` y no cuenta como éxito.

## Simulación de iteración PSO

La sección `Componente semántico` incluye el switch `Simular iteración PSO`.

Cuando está activo:

1. Carga `LLM/data/pso-individuals.json`, con 200 individuos.
2. Toma los primeros `N individuos` configurados.
3. Recorre `role`, `topic` y `action` de cada individuo.
4. Sortea cada componente y la selecciona para cambio si `random > umbral sorteo cambio`.
5. Si la componente fue sorteada, el objetivo se elige al azar entre `pbest` y `lider`.
6. Renderiza la plantilla usando el componente actual, el objetivo elegido, los otros componentes del individuo y el texto de referencia fijo.
7. Solicita `num_candidates` candidatos al LLM.
8. Aplica el primer candidato válido que mejora semanticamente hacia el objetivo y no copia el target.

El peor caso realiza `3 * N` llamadas al LLM. Con `N = 200`, eso puede llegar a 600 llamadas.

Las métricas específicas del modo PSO reportan componentes sorteadas, componentes que no pudieron cambiar, tasa de no cambio, cambios aplicados, individuos modificados, llamadas LLM, candidatos válidos, fallos por candidatos no aceptables y errores técnicos de ejecución.

## Embeddings

Los embeddings se calculan en el navegador mediante Transformers.js con:

- `feature-extraction`
- `pooling: "mean"`
- `normalize: true`

Modelos incluidos:

- `Xenova/all-MiniLM-L6-v2`
- `Xenova/gte-small`
