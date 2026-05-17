# Calculadora SBERT

Aplicación web estática para generar embeddings de dos textos y calcular similitud coseno y distancia coseno.

## Modelo

La aplicación usa `Xenova/all-MiniLM-L6-v2`, la variante ONNX preparada para Transformers.js del modelo `sentence-transformers/all-MiniLM-L6-v2`.

El modelo base genera embeddings densos de 384 dimensiones para oraciones y párrafos. En la aplicación, el pipeline se ejecuta con `feature-extraction`, `pooling: "mean"` y `normalize: true`.

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

La primera ejecución descarga Transformers.js desde jsDelivr y el modelo desde Hugging Face, por lo que requiere conexión a internet. Después, el navegador puede reutilizar su caché.

## Cálculos

- Similitud coseno: producto punto dividido por el producto de las normas de los embeddings.
- Distancia coseno: `1 - similitud coseno`.

El cálculo se ejecuta en un Web Worker para mantener la interfaz responsiva mientras se carga el modelo o se generan los embeddings.
