"""
Aplicación de tamizaje del riesgo de prediabetes o diabetes tipo 2.

Proyecto de aula · Minería de Datos · Universidad de la Costa (CUC)
Fase de despliegue (CRISP-DM) del modelo seleccionado en el segundo avance:
regresión logística con pesos balanceados por clase, entrenada con el
CDC Diabetes Health Indicators (UCI, id 891).

Ejecución local:
    streamlit run app.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Configuración general
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent
RUTA_MODELO = RAIZ / "modelo" / "modelo_regresion_logistica.joblib"
RUTA_METADATOS = RAIZ / "modelo" / "metadatos.json"

# Enlace del repositorio que se muestra en la pestaña «Datos y proyecto».
URL_GITHUB = "https://github.com/ADR14N-JDV/tamizaje-diabetes-streamlit"
URL_DATASET = "https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators"

ROJO = "#B71C1C"
AZUL = "#1F5F8B"
VERDE = "#2E7D32"
GRIS = "#6B6B6B"

st.set_page_config(
    page_title="Tamizaje de riesgo de diabetes · CUC",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 3.5rem; padding-bottom: 3rem; max-width: 1200px;}
      .etiqueta {font-size: 0.78rem; letter-spacing: .12em; text-transform: uppercase;
                 color: #B71C1C; font-weight: 600; margin-bottom: .2rem;}
      .tarjeta {border-radius: 14px; padding: 1.2rem 1.4rem; margin-bottom: 1rem;
                border: 1px solid rgba(0,0,0,.08);}
      .tarjeta h3 {margin: 0 0 .4rem 0; font-size: 1.45rem;}
      .tarjeta p {margin: 0; font-size: 1rem; line-height: 1.5; color: #222;}
      .alto {background: #FDECEA; border-left: 6px solid #B71C1C;}
      .alto h3 {color: #8E1414;}
      .bajo {background: #EAF5EC; border-left: 6px solid #2E7D32;}
      .bajo h3 {color: #1E5E22;}
      .nota {font-size: .86rem; color: #555;}
      div[data-testid="stMetricValue"] {font-size: 1.6rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Diccionarios para traducir la codificación del conjunto de datos
# ---------------------------------------------------------------------------
NOMBRES = {
    "HighBP": "Presión arterial alta",
    "HighChol": "Colesterol alto",
    "CholCheck": "Control de colesterol en 5 años",
    "Smoker": "Ha fumado 100 cigarrillos o más",
    "Stroke": "Accidente cerebrovascular",
    "HeartDiseaseorAttack": "Enfermedad coronaria o infarto",
    "PhysActivity": "Actividad física reciente",
    "Fruits": "Fruta a diario",
    "Veggies": "Verduras a diario",
    "HvyAlcoholConsump": "Consumo alto de alcohol",
    "AnyHealthcare": "Cobertura de salud",
    "NoDocbcCost": "No fue al médico por costo",
    "DiffWalk": "Dificultad para caminar",
    "Sex": "Sexo",
    "GenHlth": "Salud general autopercibida",
    "Age": "Grupo de edad",
    "Education": "Nivel educativo",
    "Income": "Nivel de ingreso",
    "BMI": "Índice de masa corporal",
    "MentHlth": "Días de mala salud mental",
    "PhysHlth": "Días de mala salud física",
}

EDADES = {1: "18 a 24", 2: "25 a 29", 3: "30 a 34", 4: "35 a 39", 5: "40 a 44",
          6: "45 a 49", 7: "50 a 54", 8: "55 a 59", 9: "60 a 64", 10: "65 a 69",
          11: "70 a 74", 12: "75 a 79", 13: "80 o más"}
SALUD = {1: "Excelente", 2: "Muy buena", 3: "Buena", 4: "Regular", 5: "Mala"}
EDUCACION = {1: "Nunca asistió o solo preescolar",
             2: "Primaria (grados 1 a 8)",
             3: "Secundaria incompleta (grados 9 a 11)",
             4: "Bachiller (grado 12 o equivalente)",
             5: "Técnico o universitario incompleto",
             6: "Universitario completo o más"}
INGRESOS = {1: "Menos de 10 000 USD", 2: "10 000 a 14 999 USD", 3: "15 000 a 19 999 USD",
            4: "20 000 a 24 999 USD", 5: "25 000 a 34 999 USD", 6: "35 000 a 49 999 USD",
            7: "50 000 a 74 999 USD", 8: "75 000 USD o más"}


@st.cache_resource
def cargar_modelo():
    return joblib.load(RUTA_MODELO)


@st.cache_data
def cargar_metadatos():
    with open(RUTA_METADATOS, encoding="utf-8") as f:
        return json.load(f)


modelo = cargar_modelo()
meta = cargar_metadatos()
VARIABLES = meta["variables"]
UMBRAL = meta["umbral"]
M = meta["metricas_prueba"]


def pct(valor, decimales=1):
    return f"{valor * 100:.{decimales}f} %".replace(".", ",")


def coma(valor, decimales=3):
    return f"{valor:.{decimales}f}".replace(".", ",")


def miles(valor):
    return f"{valor:,}".replace(",", " ")


def categoria_imc(imc):
    if imc < 18.5:
        return "Bajo peso"
    if imc < 25:
        return "Peso adecuado"
    if imc < 30:
        return "Sobrepeso"
    return "Obesidad"


def valor_legible(variable, valor):
    """Texto corto con la respuesta del usuario, para las etiquetas del gráfico de aportes."""
    if variable == "Sex":
        return "Hombre" if valor == 1 else "Mujer"
    if variable == "Age":
        return f"{EDADES[valor]} años"
    if variable == "GenHlth":
        return SALUD[valor]
    if variable == "Education":
        return f"nivel {valor} de 6"
    if variable == "Income":
        return f"nivel {valor} de 8"
    if variable == "BMI":
        return str(valor)
    if variable in ("MentHlth", "PhysHlth"):
        return f"{valor} días"
    return "Sí" if valor == 1 else "No"


def si_no(pregunta, clave, defecto="No", ayuda=None):
    """Pregunta binaria Sí/No codificada como 1/0, igual que en el conjunto de datos."""
    respuesta = st.radio(pregunta, ["No", "Sí"], index=["No", "Sí"].index(defecto),
                         horizontal=True, key=clave, help=ayuda)
    return int(respuesta == "Sí")


# ---------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------
st.markdown('<div class="etiqueta">Proyecto de aula · Minería de Datos · Universidad de la Costa</div>',
            unsafe_allow_html=True)
st.title("🩺 Tamizaje de riesgo de diabetes tipo 2")
st.markdown(
    "Esta aplicación estima si una persona tiene un **perfil de riesgo elevado de prediabetes o "
    "diabetes** a partir de 21 indicadores de salud y estilo de vida **que no requieren pruebas de "
    "laboratorio**. Usa el modelo de regresión logística desarrollado en el proyecto, entrenado con "
    f"{miles(meta['registros']['total'])} respuestas de la encuesta BRFSS de los CDC de Estados Unidos."
)
st.info(
    "**Herramienta académica de tamizaje, no de diagnóstico.** Un resultado de riesgo elevado indica "
    "que conviene realizar una prueba confirmatoria (glucosa en ayunas o hemoglobina glicosilada) y "
    "consultar a un profesional de la salud. Las respuestas no se almacenan.",
    icon="ℹ️",
)

tab_evaluar, tab_modelo, tab_datos = st.tabs(
    ["Evaluar riesgo", "Cómo funciona el modelo", "Datos y proyecto"]
)

# ---------------------------------------------------------------------------
# Pestaña 1: cuestionario y resultado
# ---------------------------------------------------------------------------
with tab_evaluar:
    st.markdown("#### Responda el cuestionario")
    st.caption("Son 21 preguntas agrupadas en cinco bloques; toma unos dos minutos. Los valores "
               "iniciales corresponden a la respuesta más frecuente en la encuesta.")

    with st.form("cuestionario"):
        col1, col2 = st.columns(2, gap="large")

        with col1:
            with st.container(border=True):
                st.markdown("**1. Datos personales**")
                sexo = st.radio("Sexo", ["Mujer", "Hombre"], horizontal=True)
                edad = st.select_slider("Edad (años)", options=list(EDADES),
                                        value=8, format_func=lambda k: EDADES[k])
                educacion = st.selectbox("Máximo nivel educativo alcanzado", list(EDUCACION),
                                         index=5, format_func=lambda k: EDUCACION[k])
                ingreso = st.selectbox(
                    "Ingreso anual del hogar", list(INGRESOS), index=7,
                    format_func=lambda k: INGRESOS[k],
                    help="Se conservan las categorías en dólares de la encuesta original, "
                         "porque así fue entrenado el modelo.",
                )

            with st.container(border=True):
                st.markdown("**2. Medidas corporales**")
                c_peso, c_talla = st.columns(2)
                peso = c_peso.number_input("Peso (kg)", min_value=30.0, max_value=250.0,
                                           value=70.0, step=0.5, format="%.1f")
                talla = c_talla.number_input("Estatura (cm)", min_value=120.0, max_value=220.0,
                                             value=165.0, step=1.0, format="%.0f")
                st.caption("Con estos datos se calcula el índice de masa corporal (IMC = peso / estatura²).")

        with col2:
            with st.container(border=True):
                st.markdown("**3. Diagnósticos y antecedentes**")
                st.caption("Responda «Sí» solo si un médico u otro profesional de la salud se lo ha dicho.")
                high_bp = si_no("¿Le han dicho que tiene la presión arterial alta?", "HighBP")
                high_chol = si_no("¿Le han dicho que tiene el colesterol alto?", "HighChol")
                chol_check = si_no("¿Se ha revisado el colesterol en los últimos 5 años?", "CholCheck", "Sí")
                stroke = si_no("¿Ha tenido un accidente cerebrovascular (derrame o ACV)?", "Stroke")
                heart = si_no("¿Ha tenido enfermedad coronaria o un infarto?", "HeartDiseaseorAttack")
                diff_walk = si_no("¿Tiene dificultad seria para caminar o subir escaleras?", "DiffWalk")

        col3, col4 = st.columns(2, gap="large")
        with col3:
            with st.container(border=True):
                st.markdown("**4. Percepción de su salud**")
                gen_hlth = st.select_slider("En general, ¿cómo diría que es su salud?",
                                            options=list(SALUD), value=2,
                                            format_func=lambda k: SALUD[k])
                ment_hlth = st.slider("En los últimos 30 días, ¿cuántos días tuvo mala salud mental "
                                      "(estrés, depresión, problemas emocionales)?", 0, 30, 0)
                phys_hlth = st.slider("En los últimos 30 días, ¿cuántos días tuvo mala salud física "
                                      "(enfermedad o lesión)?", 0, 30, 0)

        with col4:
            with st.container(border=True):
                st.markdown("**5. Hábitos y acceso a servicios de salud**")
                h1, h2 = st.columns(2, gap="medium")
                with h1:
                    phys_act = si_no("¿Hizo actividad física en los últimos 30 días, sin contar su "
                                     "trabajo?", "PhysActivity", "Sí")
                    fruits = si_no("¿Come fruta al menos una vez al día?", "Fruits", "Sí")
                    veggies = si_no("¿Come verduras al menos una vez al día?", "Veggies", "Sí")
                    alcohol = si_no(
                        "¿Su consumo de alcohol es alto?", "HvyAlcoholConsump",
                        ayuda="Más de 14 tragos por semana en hombres o más de 7 en mujeres.",
                    )
                with h2:
                    smoker = si_no("¿Ha fumado al menos 100 cigarrillos (5 cajetillas) en su vida?",
                                   "Smoker")
                    healthcare = si_no("¿Tiene algún tipo de cobertura o seguro de salud?",
                                       "AnyHealthcare", "Sí")
                    no_doc = si_no("En el último año, ¿dejó de ir al médico porque no podía pagarlo?",
                                   "NoDocbcCost")

        enviado = st.form_submit_button("Calcular riesgo", type="primary", width="stretch")

    if enviado:
        imc = peso / (talla / 100) ** 2
        imc_modelo = int(np.clip(round(imc), 12, 98))  # el conjunto de datos registra el IMC entero
        respuestas = {
            "HighBP": high_bp, "HighChol": high_chol, "CholCheck": chol_check, "Smoker": smoker,
            "Stroke": stroke, "HeartDiseaseorAttack": heart, "PhysActivity": phys_act,
            "Fruits": fruits, "Veggies": veggies, "HvyAlcoholConsump": alcohol,
            "AnyHealthcare": healthcare, "NoDocbcCost": no_doc, "DiffWalk": diff_walk,
            "Sex": int(sexo == "Hombre"), "GenHlth": gen_hlth, "Age": edad,
            "Education": educacion, "Income": ingreso, "BMI": imc_modelo,
            "MentHlth": ment_hlth, "PhysHlth": phys_hlth,
        }
        st.session_state["resultado"] = {"respuestas": respuestas, "imc": imc}

    if "resultado" in st.session_state:
        respuestas = st.session_state["resultado"]["respuestas"]
        imc = st.session_state["resultado"]["imc"]
        entrada = pd.DataFrame([respuestas])[VARIABLES]
        puntaje = float(modelo.predict_proba(entrada)[0, 1])
        riesgo_alto = puntaje >= UMBRAL

        st.divider()
        st.markdown("#### Resultado")
        r1, r2 = st.columns([1.15, 1], gap="large")

        with r1:
            if riesgo_alto:
                st.markdown(
                    '<div class="tarjeta alto"><h3>Riesgo elevado</h3><p>El perfil ingresado se '
                    'parece al de las personas con prediabetes o diabetes en la encuesta. Se recomienda '
                    '<b>realizar una prueba confirmatoria</b> (glucosa en ayunas o HbA1c) y consultar '
                    'con un profesional de la salud.</p></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="tarjeta bajo"><h3>Riesgo bajo</h3><p>El modelo no identifica un '
                    'perfil de riesgo elevado. Aun así, conviene mantener hábitos saludables y '
                    'los controles médicos periódicos, sobre todo si hay antecedentes familiares.</p></div>',
                    unsafe_allow_html=True,
                )

            m1, m2, m3 = st.columns(3)
            m1.metric("Puntaje del modelo", f"{puntaje * 100:.0f} / 100",
                      help="Umbral de alerta: 50. No es la probabilidad de tener diabetes "
                           "(ver la pestaña «Cómo funciona el modelo»).")
            m2.metric("IMC calculado", coma(imc, 1), categoria_imc(imc), delta_color="off",
                      delta_arrow="off")
            prev_edad = meta["prevalencia_por_grupo"]["Age"][str(respuestas["Age"])]["prevalencia"]
            m3.metric("Prevalencia (su edad)", f"{coma(prev_edad, 1)} %",
                      help="Porcentaje de personas con prediabetes o diabetes en la encuesta, "
                           f"entre quienes tienen {EDADES[respuestas['Age']]} años.")

        with r2:
            indicador = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(puntaje * 100),
                number={"suffix": " / 100", "font": {"size": 34}},
                gauge={
                    "axis": {"range": [0, 100], "tickvals": [0, 25, 50, 75, 100]},
                    "bar": {"color": ROJO if riesgo_alto else VERDE, "thickness": 0.3},
                    "steps": [{"range": [0, 50], "color": "#EAF5EC"},
                              {"range": [50, 100], "color": "#FDECEA"}],
                    "threshold": {"line": {"color": "#222", "width": 3}, "thickness": 0.85,
                                  "value": UMBRAL * 100},
                },
                title={"text": "Puntaje de riesgo (umbral de alerta = 50)", "font": {"size": 15}},
            ))
            indicador.update_layout(height=260, margin=dict(l=30, r=30, t=60, b=10))
            st.plotly_chart(indicador)

        # Contribución de cada respuesta frente al perfil promedio de entrenamiento
        st.markdown("#### ¿Qué respuestas influyeron más?")
        coef = pd.Series(meta["coeficientes"])
        medias = pd.Series(meta["medias"])
        desv = pd.Series(meta["desviaciones"])
        aporte = coef * (entrada.iloc[0][VARIABLES] - medias) / desv
        principales = aporte.reindex(aporte.abs().sort_values(ascending=False).index).head(8)
        principales = principales[principales.abs() > 0.02].iloc[::-1]

        if principales.empty:
            st.write("Sus respuestas están muy cerca del perfil promedio de la encuesta.")
        else:
            fig_aporte = go.Figure(go.Bar(
                x=principales.values,
                y=[f"{NOMBRES[v]}: {valor_legible(v, respuestas[v])}" for v in principales.index],
                orientation="h",
                marker_color=[ROJO if v > 0 else AZUL for v in principales.values],
                hovertemplate="%{y}: %{x:.2f}<extra></extra>",
            ))
            fig_aporte.add_vline(x=0, line_color="#333", line_width=1)
            fig_aporte.update_layout(
                height=60 + 38 * len(principales), margin=dict(l=10, r=10, t=10, b=40),
                xaxis_title="← disminuye el riesgo        aumenta el riesgo →",
                plot_bgcolor="white",
            )
            fig_aporte.update_xaxes(showgrid=True, gridcolor="#EEE", zeroline=False)
            st.plotly_chart(fig_aporte)
            st.caption(
                "Cada barra compara su respuesta con el promedio de las personas encuestadas y muestra "
                "cuánto mueve el puntaje del modelo (en la escala de la regresión logística). Rojo: "
                "aumenta el riesgo; azul: lo disminuye. Son asociaciones estadísticas, no causas."
            )

        with st.expander("¿Cómo interpretar este resultado?"):
            st.markdown(
                f"""
- El modelo marca **riesgo elevado** cuando el puntaje llega a {UMBRAL * 100:.0f}. En el conjunto de
  prueba detectó a **{pct(M['Recall (sensibilidad)'], 0)}** de las personas con prediabetes o diabetes
  (recall) y descartó correctamente a **{pct(M['Especificidad'], 0)}** de las personas sanas
  (especificidad).
- Como es una herramienta de tamizaje, se priorizó no dejar casos sin detectar. A cambio, cerca de
  **7 de cada 10 alertas son falsas** (precisión de {pct(M['Precisión'], 0)}): un resultado elevado
  **no significa que tenga diabetes**, sino que vale la pena confirmarlo con una prueba.
- Un resultado bajo tampoco descarta la enfermedad: alrededor de 1 de cada 4 casos no es detectado.
- El puntaje no es una probabilidad: el modelo se entrenó con pesos balanceados por clase para
  compensar el desbalance de la encuesta (13,9 % de casos positivos).
                """
            )

        resumen = pd.DataFrame({
            "Variable": [NOMBRES[v] for v in VARIABLES] + ["Puntaje del modelo", "Resultado"],
            "Valor": [respuestas[v] for v in VARIABLES]
                     + [round(puntaje * 100, 1), "Riesgo elevado" if riesgo_alto else "Riesgo bajo"],
        })
        st.download_button("Descargar resumen (CSV)", resumen.to_csv(index=False).encode("utf-8-sig"),
                           file_name="resultado_tamizaje_diabetes.csv", mime="text/csv")

# ---------------------------------------------------------------------------
# Pestaña 2: descripción del modelo y métricas
# ---------------------------------------------------------------------------
with tab_modelo:
    st.markdown("#### Modelo desplegado")
    st.markdown(
        f"""
**Regresión logística con pesos balanceados por clase**, dentro de un *Pipeline* de scikit-learn
que primero estandariza las 21 variables (`StandardScaler`). Se entrenó con
{miles(meta['registros']['entrenamiento'])} registros (80 %, partición estratificada, semilla 42) y se
evaluó con {miles(meta['registros']['prueba'])} registros que el modelo nunca vio. Se eligió frente a un
árbol de decisión y a una línea base porque ofrece el mejor equilibrio entre detectar casos y evitar
falsas alarmas, con resultados estables en validación cruzada de 5 particiones.
        """
    )

    st.markdown("##### Desempeño en el conjunto de prueba")
    k = st.columns(6)
    k[0].metric("Recall (sensibilidad)", pct(M["Recall (sensibilidad)"]),
                help="Proporción de personas con prediabetes o diabetes que el modelo detecta.")
    k[1].metric("Especificidad", pct(M["Especificidad"]),
                help="Proporción de personas sin diabetes que el modelo clasifica correctamente.")
    k[2].metric("ROC-AUC", coma(M["ROC-AUC"]),
                help="Capacidad de ordenar a las personas por riesgo, sin depender del umbral.")
    k[3].metric("Accuracy", pct(M["Accuracy"]))
    k[4].metric("Precisión", pct(M["Precisión"]),
                help="De las alertas de riesgo elevado, proporción que corresponde a casos reales.")
    k[5].metric("F1 (clase positiva)", coma(M["F1"]))

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("##### Matriz de confusión")
        mc = meta["matriz_confusion"]
        matriz = np.array([[mc["VN"], mc["FP"]], [mc["FN"], mc["VP"]]])
        fig_mc = px.imshow(
            matriz, color_continuous_scale="Blues", aspect="auto",
            x=["Sin diabetes", "Prediabetes o diabetes"],
            y=["Sin diabetes", "Prediabetes o diabetes"],
            labels={"x": "Predicción del modelo", "y": "Diagnóstico real", "color": "Personas"},
        )
        etiquetas_mc = [[f"VN<br>{miles(mc['VN'])}", f"FP<br>{miles(mc['FP'])}"],
                        [f"FN<br>{miles(mc['FN'])}", f"VP<br>{miles(mc['VP'])}"]]
        fig_mc.update_traces(text=etiquetas_mc, texttemplate="%{text}", textfont_size=16,
                             hovertemplate="Real: %{y}<br>Predicción: %{x}<br>%{z} personas<extra></extra>")
        fig_mc.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig_mc)
        st.caption(
            f"De {miles(mc['FN'] + mc['VP'])} personas con prediabetes o diabetes, el modelo detecta "
            f"{miles(mc['VP'])} y deja sin detectar {miles(mc['FN'])} (falsos negativos)."
        )

    with c2:
        st.markdown("##### Comparación con los otros modelos")
        comparacion = pd.DataFrame(meta["comparacion_modelos_prueba"]).T
        st.dataframe(
            comparacion.style.format(lambda v: coma(v)).highlight_max(
                axis=0, subset=["Recall", "F1", "ROC-AUC"], color="#FDECEA"),
        )
        st.caption(
            "La línea base tiene el mayor accuracy (86,1 %) sin detectar a nadie, por eso el accuracy "
            "no es el criterio de selección. El árbol tiene un recall algo mayor, pero con 1 770 falsas "
            "alarmas más y menor ROC-AUC."
        )
        cv = meta["validacion_cruzada_rl"]
        st.markdown(
            "**Validación cruzada (5 particiones) de la regresión logística:** "
            + " · ".join(f"{k} {v}" for k, v in cv.items())
        )

    st.markdown("##### Variables con mayor peso en el modelo")
    coef = pd.Series(meta["coeficientes"]).sort_values()
    razones = np.exp(coef.values)
    fig_coef = go.Figure(go.Bar(
        x=razones - 1, base=1, y=[NOMBRES[v] for v in coef.index], orientation="h",
        marker_color=[ROJO if v > 0 else AZUL for v in coef.values],
        customdata=razones, hovertemplate="%{y}: OR = %{customdata:.2f}<extra></extra>",
    ))
    fig_coef.add_vline(x=1, line_color="#333", line_width=1)
    fig_coef.update_layout(height=620, margin=dict(l=10, r=10, t=10, b=40), plot_bgcolor="white",
                           xaxis_title="Odds ratio por cada desviación estándar (1 = sin efecto)")
    fig_coef.update_xaxes(range=[0.75, 1.95], showgrid=True, gridcolor="#EEE")
    st.plotly_chart(fig_coef)
    st.caption(
        "La salud general autopercibida, el IMC, la edad, la presión alta y el colesterol alto son las "
        "variables con mayor asociación positiva. Las asociaciones negativas del consumo alto de alcohol "
        "y positivas del control de colesterol probablemente reflejan cambios de comportamiento "
        "posteriores al diagnóstico, porque la encuesta es transversal."
    )

    with st.expander("Limitaciones del modelo"):
        st.markdown(
            """
- **Datos autorreportados y transversales:** no hay confirmación clínica y algunas respuestas pueden
  ser consecuencia del diagnóstico, no factores previos a él.
- **Población de Estados Unidos (BRFSS 2015):** la generalización al Caribe colombiano no está
  garantizada; las categorías de ingreso están en dólares.
- **Precisión baja (31,1 %):** con solo 13,9 % de casos positivos, incluso una tasa moderada de
  falsos positivos supera a los verdaderos positivos.
- **Clase positiva combinada:** agrupa prediabetes y diabetes sin distinguir el tipo.
- El umbral de decisión se mantiene en 0,5, tal como se evaluó en el segundo avance.
            """
        )

# ---------------------------------------------------------------------------
# Pestaña 3: datos y proyecto
# ---------------------------------------------------------------------------
with tab_datos:
    d1, d2 = st.columns([1.1, 1], gap="large")
    with d1:
        st.markdown("#### Conjunto de datos")
        st.markdown(
            f"""
[CDC Diabetes Health Indicators]({URL_DATASET}) (UCI Machine Learning Repository, id 891), derivado
de la encuesta telefónica *Behavioral Risk Factor Surveillance System* (BRFSS) de los Centers for
Disease Control and Prevention de Estados Unidos.

- **{miles(meta['registros']['total'])} registros** y **21 variables predictoras**: 14 binarias,
  4 ordinales y 3 numéricas, sin valores faltantes.
- **Variable objetivo:** `Diabetes_binary` (0 = sin diabetes, 1 = prediabetes o diabetes).
- **Desbalance:** {coma(meta['registros']['prevalencia_general'], 1)} % de casos positivos
  (6,18 negativos por cada positivo).
            """
        )
        st.markdown("#### Proyecto")
        st.markdown(
            f"""
**Predicción temprana del riesgo de diabetes mellitus tipo 2** · Proyecto de aula de la asignatura
Minería de Datos, Facultad de Ingeniería, Universidad de la Costa (Barranquilla, 2026).

- **Integrantes:** Juan Retamozo Morán y Adrian Del Valle Quintero
- **Docente:** José Gutiérrez
- **Metodología:** CRISP-DM; esta aplicación corresponde a la fase de **despliegue**.
            """
        )
        if "USUARIO" not in URL_GITHUB:
            st.markdown(f"- **Código fuente:** [repositorio en GitHub]({URL_GITHUB})")

    with d2:
        st.markdown("#### Prevalencia observada en la encuesta")
        prev_edad = pd.DataFrame(meta["prevalencia_por_grupo"]["Age"]).T
        prev_edad.index = [EDADES[int(i)] for i in prev_edad.index]
        fig_edad = px.bar(prev_edad, x=prev_edad.index, y="prevalencia",
                          labels={"x": "Grupo de edad (años)", "prevalencia": "Prediabetes o diabetes (%)"},
                          color_discrete_sequence=[ROJO])
        fig_edad.add_hline(y=meta["registros"]["prevalencia_general"], line_dash="dash", line_color="#333",
                           annotation_text="Prevalencia general", annotation_position="top left")
        fig_edad.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10), plot_bgcolor="white",
                               xaxis_title="Grupo de edad (años)")
        st.plotly_chart(fig_edad)

        prev_salud = pd.DataFrame(meta["prevalencia_por_grupo"]["GenHlth"]).T
        prev_salud.index = [SALUD[int(i)] for i in prev_salud.index]
        fig_salud = px.bar(prev_salud, x=prev_salud.index, y="prevalencia",
                           labels={"x": "Salud general autopercibida",
                                   "prevalencia": "Prediabetes o diabetes (%)"},
                           color_discrete_sequence=[ROJO])
        fig_salud.add_hline(y=meta["registros"]["prevalencia_general"], line_dash="dash", line_color="#333")
        fig_salud.update_layout(height=280, margin=dict(l=10, r=10, t=20, b=10), plot_bgcolor="white",
                                xaxis_title="Salud general autopercibida")
        st.plotly_chart(fig_salud)

    st.markdown("#### Diccionario de variables usadas por el modelo")
    diccionario = pd.DataFrame({
        "Variable": VARIABLES,
        "Descripción": [NOMBRES[v] for v in VARIABLES],
        "Tipo": ["Binaria (0 = no, 1 = sí)"] * 13 + ["Binaria (0 = mujer, 1 = hombre)",
                 "Ordinal (1 = excelente … 5 = mala)", "Ordinal (1 = 18–24 … 13 = 80 o más)",
                 "Ordinal (1 a 6)", "Ordinal (1 a 8)", "Numérica (kg/m²)",
                 "Numérica (0 a 30 días)", "Numérica (0 a 30 días)"],
    })
    st.dataframe(diccionario, hide_index=True, height=35 * len(diccionario) + 40)

st.markdown(
    '<p class="nota" style="text-align:center;margin-top:2rem;">Universidad de la Costa · '
    'Minería de Datos · 2026 · Uso académico</p>',
    unsafe_allow_html=True,
)
