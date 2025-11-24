1. Propósito del Proyecto

El propósito de este proyecto es desarrollar un modelo de aprendizaje supervisado capaz de clasificar correctamente las especies de flores Iris a partir de sus características morfológicas: longitud y ancho del sépalo, y longitud y ancho del pétalo.

Además, se implementa un dashboard interactivo con Streamlit que permite:

Visualizar el análisis exploratorio del dataset.

Consultar las métricas del modelo entrenado.

Realizar predicciones en tiempo real ingresando nuevos valores.

Este proyecto aplica el flujo completo de un proceso de minería de datos, desde la carga del dataset hasta la presentación de resultados.

2. Estructura del Proyecto

El repositorio contiene los siguientes archivos:

│── Iris.csv               # Dataset proporcionado por el profesor
│── proyectoiris.py        # Dashboard principal desarrollado en Streamlit
│── model_iris_rf.joblib   # Modelo Random Forest entrenado y guardado
│── requirements.txt        # Dependencias necesarias para ejecutar el proyecto
└── README.md               # Descripción y guía de uso del proyecto

3. Metodología de Desarrollo

El desarrollo del proyecto se basó en un ciclo de trabajo inspirado en CRISP-DM, compuesto por:

3.1 Comprensión del Problema

Identificación del objetivo de clasificar tres especies de Iris: Setosa, Versicolor y Virginica.

3.2 Comprensión de los Datos

Se analizó el dataset Iris.csv:

Estructura y tipos de datos

Distribución de clases

Estadísticos descriptivos

3.3 Análisis Exploratorio (EDA)

Incluye:

Histogramas por variable

Scatter Matrix con separación entre especies

Matriz de correlación entre variables numéricas

3.4 Preprocesamiento

Codificación de etiquetas con LabelEncoder

División del dataset en entrenamiento (80%) y prueba (20%)

Escalado de variables con StandardScaler

3.5 Entrenamiento del Modelo

Se utilizó un RandomForestClassifier, seleccionando los mejores hiperparámetros mediante GridSearchCV.

3.6 Evaluación

Se calcularon las métricas:

Accuracy

Precision

Recall

F1-score

Matriz de confusión

El modelo final obtuvo un desempeño superior al 90%.

3.7 Implementación del Dashboard

El dashboard fue desarrollado con Streamlit, integrando:

Visualizaciones EDA

Métricas del modelo

Predicciones interactivas

4. Instrucciones de Ejecución
4.1 Requerimientos Previos

Asegúrese de tener instalado:

Python 3.9 o superior

Pip actualizado

4.2 Instalación de Dependencias

Ejecutar en consola:

pip install -r requirements.txt


Contenido del archivo:

streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
plotly
joblib

4.3 Ejecución del Dashboard

Ejecute el siguiente comando en la carpeta del proyecto:

streamlit run proyectoiris.py


Al ejecutarse, Streamlit generará un enlace local donde podrá visualizar el dashboard.

5. Funcionalidades Principales del Dashboard
5.1 Visualización del Dataset

Permite revisar las características y primeras filas del archivo Iris.csv.

5.2 Análisis Exploratorio

Incluye gráficos interactivos como:

Histogramas

Matriz de dispersión

Matriz de correlación

5.3 Resultados del Modelo

Muestra las métricas del Random Forest optimizado.

5.4 Predicción en Tiempo Real

El usuario puede ingresar las medidas de una flor y recibir la especie predicha.

6. Objetivo Académico

Este proyecto fue desarrollado como parte de la asignatura Minería de Datos, con el objetivo de aplicar conceptos fundamentales de procesamiento de datos, análisis exploratorio, modelado supervisado y despliegue de herramientas interactivas.
