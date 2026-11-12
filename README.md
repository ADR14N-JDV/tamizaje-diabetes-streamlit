# Tamizaje de riesgo de diabetes tipo 2 · Streamlit

Aplicación web que despliega el modelo de clasificación del proyecto de aula
**«Predicción temprana del riesgo de diabetes mellitus tipo 2»** (Minería de Datos,
Universidad de la Costa, 2026). Corresponde a la fase de **despliegue** de CRISP-DM.

- **Aplicación pública:** <https://tamizaje-diabetes-cuc.streamlit.app>
- **Integrantes:** Juan Retamozo Morán y Adrian Del Valle Quintero
- **Docente:** José Gutiérrez

## ¿Qué hace la aplicación?

1. Explica su propósito: identificar a personas con un perfil de riesgo elevado de
   prediabetes o diabetes usando solo indicadores de fácil obtención, sin pruebas de laboratorio.
2. Solicita los 21 indicadores en un cuestionario de cinco bloques, con preguntas en lenguaje
   cotidiano (el IMC se calcula a partir del peso y la estatura).
3. Presenta el resultado como **riesgo elevado** o **riesgo bajo**, con el puntaje del modelo,
   una recomendación de acción, el IMC calculado y las respuestas que más influyeron.
4. Documenta el modelo: métricas en el conjunto de prueba, matriz de confusión, comparación con el
   árbol de decisión y la línea base, coeficientes y limitaciones.

> Herramienta académica de tamizaje; no reemplaza un diagnóstico médico.

## Modelo

| Elemento | Detalle |
|---|---|
| Datos | CDC Diabetes Health Indicators (UCI, id 891): 253 680 registros, 21 predictoras |
| Objetivo | `Diabetes_binary` (0 = sin diabetes, 1 = prediabetes o diabetes) |
| Partición | 80 % entrenamiento / 20 % prueba, estratificada, semilla 42 |
| Algoritmo | `Pipeline(StandardScaler, LogisticRegression(class_weight="balanced"))` |
| Umbral | 0,5 |

Métricas en el conjunto de prueba (n = 50 736):

| Recall | Especificidad | ROC-AUC | Accuracy | Precisión | F1 |
|---|---|---|---|---|---|
| 0,761 | 0,727 | 0,820 | 0,732 | 0,311 | 0,441 |

## Estructura del repositorio

```
├── app.py                      # aplicación de Streamlit
├── entrenar_modelo.py          # reentrena el modelo y genera los archivos de modelo/
├── requirements.txt            # dependencias con versiones fijas
├── .streamlit/config.toml      # tema visual
└── modelo/
    ├── modelo_regresion_logistica.joblib   # pipeline entrenado
    └── metadatos.json                      # métricas, coeficientes y tablas de apoyo
```

## Ejecución local

```bash
python -m venv .venv
source .venv/bin/activate        # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
python entrenar_modelo.py        # opcional: regenera el modelo
streamlit run app.py
```

La aplicación solo necesita los archivos de `modelo/`. El script `entrenar_modelo.py` busca el CSV
oficial en `data/cdc_diabetes_health_indicators.csv` y, si no está, lo descarga desde el UCI Machine
Learning Repository con `ucimlrepo` y lo guarda en esa ruta.

## Despliegue en Streamlit Community Cloud

1. Subir este repositorio a GitHub (público).
2. Entrar a <https://share.streamlit.io> con la cuenta de GitHub y elegir **Create app**.
3. Seleccionar el repositorio, la rama `main` y el archivo principal `app.py`.
4. En **Advanced settings**, elegir Python 3.12.
5. Pulsar **Deploy** y copiar el enlace público generado.
