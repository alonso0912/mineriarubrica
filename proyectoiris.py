# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import os

# --- SECCIÓN PARA ESCOLLER EL CSV EN STREAMLIT ---
st.title("Minería de Rubrica - Iris Dataset")

# Subidor de archivos
uploaded_file = st.file_uploader("Selecciona el archivo CSV de Iris", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"Dataset cargado — filas: {df.shape[0]}, columnas: {df.shape[1]}")
    st.dataframe(df.head())
    # El resto de tu pipeline va aquí...
else:
    st.warning("Por favor, selecciona un archivo CSV para continuar.")

# El resto de tu código original (EDA, preprocesamiento, modelo, etc.) debería ir aquí, 
# validando siempre que df exista antes de operar.

# Estandarizar nombres de columnas si es necesario
def standardize_columns(df):
    col_map = {}
    for c in df.columns:
        lc = c.lower().strip()
        if "sepal" in lc and "length" in lc:
            col_map[c] = "sepal_length"
        if "sepal" in lc and "width" in lc:
            col_map[c] = "sepal_width"
        if "petal" in lc and "length" in lc:
            col_map[c] = "petal_length"
        if "petal" in lc and "width" in lc:
            col_map[c] = "petal_width"
        if "species" in lc or "class" in lc:
            col_map[c] = "species"
    df = df.rename(columns=col_map)
    return df

if 'df' in globals():
    df = standardize_columns(df)
    print("Columnas actuales:", df.columns.tolist())
else:
    print("DataFrame no cargado todavía.")

"""## 4) Información básica y valores faltantes"""

if 'df' in globals():
    st.text(df.info())
    st.text(df.describe())
    print('\nValores faltantes por columna:')
    print(df.isnull().sum())
else:
    print("Carga el dataset primero.")

"""## 5) Exploratory Data Analysis (EDA)"""

if 'df' in globals():
    # Histogramas
    df_melt = df.melt(id_vars=['species']) if 'species' in df.columns else df.melt()
    fig = px.histogram(df_melt, x='value', color='species' if 'species' in df.columns else None, facet_col='variable', facet_col_wrap=2, title='Histogramas por variable')
    fig.show()
    # Scatter matrix
    fig2 = px.scatter_matrix(df, dimensions=['sepal_length','sepal_width','petal_length','petal_width'], color='species' if 'species' in df.columns else None, title='Scatter matrix')
    fig2.update_traces(diagonal_visible=False)
    fig2.show()
    # Heatmap correlation
    plt.figure(figsize=(6,4))
    sns.heatmap(df[['sepal_length','sepal_width','petal_length','petal_width']].corr(), annot=True, cmap='coolwarm')
    plt.title('Matriz de correlación')
    plt.show()
else:
    print('Carga el dataset primero.')

"""## 6) Preprocesamiento
Codificar etiquetas, train/test split y escalado.
"""

if 'df' in globals():
    X = df[['sepal_length','sepal_width','petal_length','petal_width']].values
    y = df['species'].values
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    print('Preprocesamiento completado. Tamaños — Train:', X_train.shape, 'Test:', X_test.shape)
else:
    print('Carga el dataset primero.')

"""## 7) Entrenamiento del modelo
Entrenamos RandomForest y probamos con GridSearchCV (opcional).
"""

if 'X_train_s' in globals():
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5]
    }
    grid = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train_s, y_train)
    best = grid.best_estimator_
    print('Best params:', grid.best_params_)
    # Fit best on entire train set
    best.fit(X_train_s, y_train)
    # Save model, scaler, label encoder
    joblib.dump({'model': best, 'scaler': scaler, 'le': le}, 'model_iris_rf.joblib')
    print('Modelo entrenado y guardado: model_iris_rf.joblib')
else:
    print('Ejecuta las celdas de preprocesamiento antes.')

"""## 8) Evaluación del modelo"""

if 'best' in globals():
    y_pred = best.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    print(f'Accuracy: {acc:.4f}')
    print(f'Precision: {prec:.4f}')
    print(f'Recall: {rec:.4f}')
    print(f'F1-score: {f1:.4f}')
    print('\nClassification report:\n')
    print(classification_report(y_test, y_pred, target_names=le.inverse_transform([0,1,2])))
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(cm, labels=dict(x='Predicho', y='Real'), x=le.inverse_transform([0,1,2]), y=le.inverse_transform([0,1,2]), text_auto=True)
    fig.update_layout(title='Matriz de confusión')
    fig.show()
else:
    print('Entrena el modelo primero.')

"""## 9) Predicción de nuevas muestras"""

# Ejemplo: predecir una nueva muestra
if 'best' in globals():
    sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # cambiar si quieres
    sample_s = scaler.transform(sample)
    pred = best.predict(sample_s)[0]
    probs = best.predict_proba(sample_s)[0]
    print('Predicción:', le.inverse_transform([pred])[0])
    print('Probabilidades por clase:')
    for name, p in zip(le.inverse_transform([0,1,2]), probs):
        print(f'{name}: {p:.3f}')
else:
    print('Entrena el modelo primero.')
