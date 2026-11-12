"""
Entrenamiento y exportación del modelo para la aplicación de Streamlit.

Reproduce exactamente el modelo seleccionado en el segundo avance del proyecto
(cuaderno Avance_2_Mineria_de_Datos.ipynb):

  * Conjunto de datos: CDC Diabetes Health Indicators (UCI, id 891), 253 680 registros.
  * 21 variables predictoras, variable objetivo Diabetes_binary.
  * Partición estratificada 80/20 con semilla 42.
  * Pipeline: StandardScaler + LogisticRegression(class_weight="balanced").

Genera dos archivos en la carpeta modelo/:
  * modelo_regresion_logistica.joblib -> pipeline entrenado que usa la app.
  * metadatos.json -> métricas de prueba, matriz de confusión, coeficientes y
    tablas de prevalencia que la app muestra sin necesidad de cargar el CSV.

Uso:
    python entrenar_modelo.py
"""

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEMILLA = 42
UMBRAL = 0.5

RAIZ = Path(__file__).resolve().parent
ARCHIVO_DATOS = RAIZ / "data" / "cdc_diabetes_health_indicators.csv"
CARPETA_MODELO = RAIZ / "modelo"

BINARIAS = ["HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", "HeartDiseaseorAttack",
            "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare",
            "NoDocbcCost", "DiffWalk", "Sex"]
ORDINALES = ["GenHlth", "Age", "Education", "Income"]
NUMERICAS = ["BMI", "MentHlth", "PhysHlth"]
VARIABLES = BINARIAS + ORDINALES + NUMERICAS
OBJETIVO = "Diabetes_binary"


def cargar_datos() -> pd.DataFrame:
    """Lee la copia local del CSV oficial; si no existe, la descarga desde UCI."""
    if not ARCHIVO_DATOS.exists():
        try:
            from ucimlrepo import fetch_ucirepo
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ucimlrepo"])
            from ucimlrepo import fetch_ucirepo
        cdc = fetch_ucirepo(id=891)
        datos = pd.concat([cdc.data.ids, cdc.data.targets, cdc.data.features], axis=1)
        ARCHIVO_DATOS.parent.mkdir(parents=True, exist_ok=True)
        datos.to_csv(ARCHIVO_DATOS, index=False)
    return pd.read_csv(ARCHIVO_DATOS).drop(columns="ID")


def main() -> None:
    df = cargar_datos()
    X, y = df[VARIABLES], df[OBJETIVO]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEMILLA
    )

    modelo = Pipeline([
        ("escalamiento", StandardScaler()),
        ("modelo", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEMILLA)),
    ])
    modelo.fit(X_train, y_train)

    prob = modelo.predict_proba(X_test)[:, 1]
    pred = (prob >= UMBRAL).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

    metricas = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precisión": precision_score(y_test, pred, zero_division=0),
        "Recall (sensibilidad)": recall_score(y_test, pred),
        "Especificidad": recall_score(y_test, pred, pos_label=0),
        "F1": f1_score(y_test, pred),
        "F1 macro": f1_score(y_test, pred, average="macro"),
        "ROC-AUC": roc_auc_score(y_test, prob),
    }

    escalador = modelo.named_steps["escalamiento"]
    logistica = modelo.named_steps["modelo"]
    coeficientes = pd.Series(logistica.coef_[0], index=VARIABLES)

    # Prevalencia observada por grupo (contexto para el usuario en la app)
    def prevalencia(columna):
        tabla = df.groupby(columna)[OBJETIVO].agg(["mean", "size"])
        return {str(int(k)): {"prevalencia": round(float(v["mean"]) * 100, 1), "n": int(v["size"])}
                for k, v in tabla.iterrows()}

    metadatos = {
        "fecha_entrenamiento": date.today().isoformat(),
        "versiones": {"scikit-learn": sklearn.__version__, "pandas": pd.__version__,
                      "numpy": np.__version__},
        "semilla": SEMILLA,
        "umbral": UMBRAL,
        "variables": VARIABLES,
        "registros": {"total": int(len(df)), "entrenamiento": int(len(y_train)),
                      "prueba": int(len(y_test)),
                      "positivos_prueba": int(y_test.sum()),
                      "prevalencia_general": round(float(y.mean()) * 100, 2)},
        "metricas_prueba": {k: round(float(v), 3) for k, v in metricas.items()},
        "matriz_confusion": {"VN": int(tn), "FP": int(fp), "FN": int(fn), "VP": int(tp)},
        "coeficientes": {k: round(float(v), 4) for k, v in coeficientes.items()},
        "intercepto": round(float(logistica.intercept_[0]), 4),
        "medias": {k: float(v) for k, v in zip(VARIABLES, escalador.mean_)},
        "desviaciones": {k: float(v) for k, v in zip(VARIABLES, escalador.scale_)},
        "prevalencia_por_grupo": {c: prevalencia(c) for c in ["Age", "GenHlth", "HighBP", "Income"]},
        # Referencia del segundo avance (Tabla 5 del informe) para comparar
        "comparacion_modelos_prueba": {
            "Línea base": {"Accuracy": 0.861, "Recall": 0.000, "Especificidad": 1.000,
                           "F1": 0.000, "ROC-AUC": 0.500},
            "Regresión logística": {"Accuracy": 0.732, "Recall": 0.761, "Especificidad": 0.727,
                                    "F1": 0.441, "ROC-AUC": 0.820},
            "Árbol de decisión": {"Accuracy": 0.698, "Recall": 0.770, "Especificidad": 0.686,
                                  "F1": 0.415, "ROC-AUC": 0.801},
        },
        "validacion_cruzada_rl": {"Accuracy": "0,732 ± 0,001", "Recall": "0,765 ± 0,009",
                                  "F1": "0,443 ± 0,004", "ROC-AUC": "0,822 ± 0,003"},
    }

    CARPETA_MODELO.mkdir(exist_ok=True)
    joblib.dump(modelo, CARPETA_MODELO / "modelo_regresion_logistica.joblib")
    with open(CARPETA_MODELO / "metadatos.json", "w", encoding="utf-8") as f:
        json.dump(metadatos, f, ensure_ascii=False, indent=2)

    print("Modelo guardado en", CARPETA_MODELO)
    print(pd.Series(metricas).round(3).to_string())
    print(f"Matriz de confusión: VN={tn} FP={fp} FN={fn} VP={tp}")


if __name__ == "__main__":
    main()
