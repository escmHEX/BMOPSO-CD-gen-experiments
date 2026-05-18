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

La herramienta reporta tasa de éxito válida, mejor similitud promedio, mejora promedio, candidatos válidos y resultados por ejecución. Una ejecución sin candidatos parseables cuenta como ejecución completada sin éxito, no se excluye del denominador.

El umbral `tau_target_copy` queda en `0.94` por defecto. Si `cos(candidate, target) >= tau_target_copy`, el candidato se etiqueta como `semantic_copy_target` y no cuenta como éxito.

Las líneas duplicadas no se descartan silenciosamente: se conservan en la tabla y se marcan como `duplicate_output`.

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

Los embeddings se calculan en el navegador mediante Transformers.js con:

- `feature-extraction`
- `pooling: "mean"`
- `normalize: true`

Para que la calculadora y el evaluador reporten el mismo valor al copiar exactamente los mismos textos, cada string se embebe de forma individual y se cachea por `modelo + texto exacto`. Esto evita que el resultado dependa del lote de textos evaluados en una corrida.

Modelos incluidos:

- `Xenova/all-MiniLM-L6-v2`
- `Xenova/gte-small`
