# Calculadora SBERT

Aplicación web estática para generar embeddings de dos textos y calcular similitud coseno y distancia coseno.

## Modelos

La aplicación permite seleccionar:

- `Xenova/all-MiniLM-L6-v2`, variante ONNX preparada para Transformers.js del modelo `sentence-transformers/all-MiniLM-L6-v2`.
- `Xenova/gte-small`, variante ONNX preparada para Transformers.js del modelo `thenlper/gte-small`.

Ambos generan embeddings densos de 384 dimensiones. En la aplicación, el pipeline se ejecuta con `feature-extraction`, `pooling: "mean"` y `normalize: true`.

## Ejecución local

Desde esta carpeta en Windows:

```powershell
py -m http.server 4173
```

Luego abrir:

```text
http://localhost:4173
```

En entornos donde `python` esté en el `PATH`, el comando equivalente es `python -m http.server 4173`.

La primera ejecución descarga Transformers.js desde jsDelivr y el modelo seleccionado desde Hugging Face, por lo que requiere conexión a internet. Después, el navegador puede reutilizar su caché. Cada modelo se cachea por separado dentro del worker mientras la página sigue abierta.

## Cálculos

- Similitud coseno: producto punto dividido por el producto de las normas de los embeddings.
- Distancia coseno: `1 - similitud coseno`.

El cálculo se ejecuta en un Web Worker para mantener la interfaz responsiva mientras se carga el modelo o se generan los embeddings.

La página también muestra las fórmulas, la configuración de inferencia y las características principales del modelo seleccionado, incluyendo MTEB Average, MTEB Retrieval, dimensión y costo aproximado de descarga.
