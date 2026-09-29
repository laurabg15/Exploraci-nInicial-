# Predicción del Precio de Viviendas – House Prices (Ames, Iowa)

## Integrantes del equipo
- Laura Vanessa Botero Gil
- Duban Lopez Sanchez

## Descripción del problema
Se busca estimar el precio de venta de viviendas residenciales en Ames, Iowa (EE. UU.) a partir de sus características físicas, de calidad y de ubicación. Es un problema de **aprendizaje supervisado de regresión**, cuya variable objetivo es `SalePrice` (precio de venta en USD).

## Fuente del conjunto de datos
Competencia de Kaggle *House Prices – Advanced Regression Techniques*:
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

- `train.csv`: 1.460 viviendas, 79 variables explicativas + `SalePrice`.
- `test.csv`: 1.459 viviendas sin `SalePrice` (conjunto de validación de la competencia).

## Objetivo del modelo
Predecir `SalePrice` con el menor error posible en viviendas no vistas, usando un modelo simple e interpretable.

## Algoritmo utilizado
**Regresión Lineal** (`LinearRegression` de scikit-learn).

Se compararon un modelo base (mediana), Regresión Lineal, Ridge, Lasso, Random Forest y XGBoost. Ridge y Lasso obtuvieron resultados prácticamente iguales a los de la Regresión Lineal. Random Forest y XGBoost mostraron sobreajuste: ajuste casi perfecto en entrenamiento y peor desempeño en prueba. Por eso se eligió la Regresión Lineal, que ofrece simplicidad, interpretabilidad y buena generalización.

Variables finales (13 tras la codificación): `OverallQual`, `ExterQual`, `BsmtQual`, `KitchenQual`, `TotalBsmtSF`, `1stFlrSF`, `GrLivArea`, `GarageArea`, `GarageFinish` (one-hot) y `CentralAir` (one-hot).

## Métrica empleada
- **MAE** (error absoluto medio): métrica principal, en dólares.
- **RMSE** (raíz del error cuadrático medio): penaliza más los errores grandes.
- **R²** (coeficiente de determinación): proporción de la variabilidad del precio que explica el modelo.

## Principales resultados obtenidos
Partición 80/20 del `train.csv` (`random_state=80`):

| Modelo                  | MAE Test  | RMSE Test | R² Test |
|-------------------------|-----------|-----------|---------|
| Modelo Base (Mediana)   | 54.045    | 74.327    | -0.05   |
| **Regresión Lineal**    | **20.564**| **27.091**| **0.86**|
| Ridge (alpha=1)         | 20.558    | 27.083    | 0.86    |
| Random Forest           | 18.759    | 28.551    | 0.85    |
| XGBoost                 | 20.132    | 31.239    | 0.82    |

- El modelo se equivoca en promedio unos **$20.500** por vivienda, frente a unos $54.000 del modelo base.
- Explica alrededor del **86 %** de la variabilidad del precio en el conjunto de prueba.

## Instrucciones para ejecutar el notebook
1. Clonar el repositorio y entrar a la carpeta del proyecto.
2. Crear un entorno virtual (opcional) e instalar las dependencias:
   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib kaggle python-dotenv jupyter
   ```
3. Datos:
   - Si `train.csv` y `test.csv` ya están en la carpeta, no hay que hacer nada más.
   - Para descargarlos desde Kaggle, genere un token en https://www.kaggle.com/settings/api y guárdelo en un archivo `.env` en la raíz del proyecto:
     ```
     KAGGLE_API_TOKEN=su_token
     ```
     (Debe tener aceptadas las reglas de la competencia en Kaggle.)
4. Abrir el notebook y ejecutar todas las celdas en orden:
   ```bash
   jupyter notebook InitialExploration_Dataset.ipynb
   ```
   Si los CSV ya están descargados, puede omitir la primera celda de código (descarga desde Kaggle).

### Uso del modelo guardado
```python
import joblib, pandas as pd

transformador = joblib.load("pipeline_transformacion.joblib")
modelo = joblib.load("modelo_regresion_lineal.joblib")

datos = pd.read_csv("test.csv")
predicciones = modelo.predict(transformador.transform(datos))
```
