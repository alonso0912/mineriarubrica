import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import io

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


# ======================================================
# 1. CONFIGURACIÓN STREAMLIT
# ======================================================

st.set_page_config(page_title="Iris Species Classification", layout="wide")
st.title("🌸 Iris Species Classification — Dashboard Completo")


# ======================================================
# 2. CARGA DEL DATASET
# ======================================================

st.header("1) Cargar Dataset")

CSV_PATHS = [
    "Iris.csv",
    "iris.csv",
    "/mnt/data/Iris.csv"
]

df = None

for path in CSV_PATHS:
    if os.path.exists(path):
        df = pd.read_csv(path)
        st.success(f"Archivo cargado desde: {path}")
        break

if df is None:
    st.error("No se encontró Iris.csv. Súbelo manualmente:")
    uploaded = st.file_uploader("Sube Iris.csv")
    if uploaded:
        df = pd.read_csv(uploaded)
    else:
        st.stop()


# Renombrar columnas automáticamente
df = df.rename(columns={
    df.columns[1]: "sepal_length",
    df.columns[2]: "sepal_width",
    df.columns[3]: "petal_length",
    df.columns[4]: "petal_width",
    df.columns[5]: "species"
})

st.dataframe(df.head())


# ======================================================
# 3. INFORMACIÓN DEL DATASET
# ======================================================

st.header("2) Información del Dataset")

buf = io.StringIO()
df.info(buf=buf)
info_str = buf.getvalue()

st.text(info_str)

st.subheader("Descripción Estadística")
st.write(df.describe())

st.subheader("Clases Presentes")
st.write(df["species"].value_counts())


# ======================================================
# 4. EDA (VISUALIZACIONES)
# ======================================================

st.header("3) Análisis Exploratorio (EDA)")

st.subheader("Histogramas")
df_melt = df.melt(id_vars="species")
fig = px.histogram(df_melt, x="value", color="species",
                   facet_col="variable", facet_col_wrap=2)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Scatter Matrix")
fig_scatter = px.scatter_matrix(
    df,
    dimensions=["sepal_length","sepal_width","petal_length","petal_width"],
    color="species"
)
fig_scatter.update_traces(diagonal_visible=False)
st.plotly_chart(fig_scatter, use_container_width=True)

st.subheader("Matriz de Correlación")

numeric_df = df.select_dtypes(include=[np.number])

fig, ax = plt.subplots(figsize=(6,4))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)



# ======================================================
# 5. PREPROCESAMIENTO
# ======================================================

st.header("4) Preprocesamiento")

X = df[["sepal_length","sepal_width","petal_length","petal_width"]].values
y = df["species"].values

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

st.success("Preprocesamiento completado.")


# ======================================================
# 6. ENTRENAMIENTO DEL MODELO
# ======================================================

st.header("5) Entrenamiento del Modelo (Random Forest)")

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5]
}

rf = RandomForestClassifier(random_state=42)
grid = GridSearchCV(rf, param_grid, cv=5, scoring="accuracy", n_jobs=-1)

grid.fit(X_train_s, y_train)
best = grid.best_estimator_

st.success(f"Mejores parámetros encontrados: {grid.best_params_}")

# Guardar modelo
joblib.dump({
    "model": best,
    "scaler": scaler,
    "le": le
}, "model_iris_rf.joblib")

st.success("Modelo guardado como model_iris_rf.joblib")


# ======================================================
# 7. EVALUACIÓN
# ======================================================

st.header("6) Evaluación del Modelo")

y_pred = best.predict(X_test_s)

st.write("### Accuracy:", accuracy_score(y_test, y_pred))
st.write("### Precision:", precision_score(y_test, y_pred, average="weighted"))
st.write("### Recall:", recall_score(y_test, y_pred, average="weighted"))
st.write("### F1-score:", f1_score(y_test, y_pred, average="weighted"))

st.subheader("Reporte de Clasificación")
report = classification_report(
    y_test, y_pred, target_names=le.inverse_transform([0,1,2])
)
st.text(report)

st.subheader("Matriz de Confusión")
cm = confusion_matrix(y_test, y_pred)
fig_cm = px.imshow(cm,
                   text_auto=True,
                   labels={"x": "Predicción", "y": "Real"},
                   x=le.inverse_transform([0,1,2]),
                   y=le.inverse_transform([0,1,2]))
st.plotly_chart(fig_cm)


# ======================================================
# 8. PREDECIR NUEVAS MUESTRAS
# ======================================================

st.header("7) Predicción en Tiempo Real")

col1, col2, col3, col4 = st.columns(4)

sepal_length = col1.number_input("Sepal length", value=5.1)
sepal_width  = col2.number_input("Sepal width",  value=3.5)
petal_length = col3.number_input("Petal length", value=1.4)
petal_width  = col4.number_input("Petal width",  value=0.2)

if st.button("Predecir especie"):
    new = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    new_s = scaler.transform(new)
    pred = le.inverse_transform(best.predict(new_s))[0]
    st.success(f"🌼 Especie predicha: **{pred}**")


