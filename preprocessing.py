# Importar las bibliotecas necesarias
import numpy as np  # Para realizar operaciones numéricas
import pandas as pd  # Para manejar datos en formato de DataFrame
from sklearn.preprocessing import StandardScaler  # Para escalar datos con Z-score
from sklearn.preprocessing import MinMaxScaler  # Para escalar datos en el rango 0-1
from sklearn.linear_model import LogisticRegression  # Para crear el modelo de regresión logística

# Cargar los datos
PATH = 'input/'  # Ruta donde se encuentran los datos
train_data = pd.read_csv(PATH + 'train.csv')  # Cargar datos de entrenamiento
test_data = pd.read_csv(PATH + 'test.csv')  # Cargar datos de prueba

# Función para preprocesar columnas como título, tamaño de la familia, etc.
def preprocess_data(df):
    # Crear la columna FamilySize (tamaño de la familia) sumando SibSp y Parch y sumando 1 para incluir al pasajero mismo
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # Convertir la columna Sex a valores numéricos (0 para male, 1 para female)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    
    # Extraer el título del pasajero de la columna Name y luego eliminar la columna Name
    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)  # Extraer títulos como Mr, Miss, etc.
    # Agrupar ciertos títulos menos comunes en la categoría 'Others'
    df['Title'] = df['Title'].replace(
        ['Dr', 'Rev', 'Col', 'Major', 'Countess', 'Sir', 'Jonkheer', 'Lady', 'Capt', 'Don'], 'Others'
    )
    # Agrupar títulos relacionados (Ms, Mlle, Mme) bajo la categoría 'Miss'
    df['Title'] = df['Title'].replace(['Ms', 'Mlle', 'Mme'], 'Miss')
    # Mapear títulos a valores numéricos
    df['Title'] = df['Title'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Others': 4})
    
    # Eliminar columnas innecesarias para el modelo
    df = df.drop(columns=['Ticket', 'PassengerId', 'Cabin', 'Name'])
    return df

# Aplicar el preprocesamiento tanto a los datos de entrenamiento como a los de prueba
train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

# Rellenar valores faltantes en Embarked con el valor de la clase más frecuente (aquí se usa el código 2 como supuesto de mayoría)
train_data['Embarked'] = train_data['Embarked'].fillna(2)
test_data['Embarked'] = test_data['Embarked'].fillna(2)

# Función para rellenar valores faltantes en Age con la mediana de pasajeros similares
def fill_age_by_similar_passengers(df, reference_df):
    # Encontrar los índices donde Age es nulo
    NaN_indexes = df['Age'][df['Age'].isnull()].index
    for i in NaN_indexes:
        # Calcular la mediana de Age para pasajeros con características similares (mismo Pclass, SibSp y Parch)
        pred_age = reference_df['Age'][((reference_df.SibSp == df.iloc[i]["SibSp"]) & 
                                        (reference_df.Parch == df.iloc[i]["Parch"]) & 
                                        (reference_df.Pclass == df.iloc[i]["Pclass"]))].median()
        # Si hay una mediana calculada, usarla; si no, usar la mediana general
        df.at[i, 'Age'] = pred_age if not np.isnan(pred_age) else reference_df['Age'].median()
    return df

# Aplicar la estrategia de imputación de Age tanto al conjunto de entrenamiento como al de prueba
train_data = fill_age_by_similar_passengers(train_data, train_data)
test_data = fill_age_by_similar_passengers(test_data, train_data)

# Rellenar valores faltantes en Fare en el conjunto de prueba con la media de Fare del conjunto de entrenamiento
test_data['Fare'] = test_data['Fare'].fillna(train_data['Fare'].mean())

# Separar las características (X) y el objetivo (y) para el entrenamiento
X_train = train_data.drop(columns='Survived')  # X_train contiene todas las columnas excepto Survived
y_train = train_data['Survived']  # y_train contiene únicamente la columna Survived
X_test = test_data.copy()  # Copiar X_test para los datos de prueba

# Combinar los datos de entrenamiento y prueba para hacer una codificación consistente
combined_data = pd.concat([X_train, X_test], keys=['train', 'test'])

# Aplicar codificación One-Hot para las columnas categóricas (Embarked y Title)
combined_data = pd.get_dummies(combined_data, columns=['Embarked', 'Title'], drop_first=True)

# Dividir de nuevo en X_train y X_test
X_train = combined_data.loc['train'].reset_index(drop=True)
X_test = combined_data.loc['test'].reset_index(drop=True)

# Escalar todas las características al rango 0-1 usando MinMaxScaler
scaler = MinMaxScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)  # Escalar y transformar X_train
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)  # Transformar X_test con el mismo escalador

# Inicializar y entrenar el modelo de regresión logística
model = LogisticRegression()
model.fit(X_train, y_train)  # Ajustar el modelo con los datos de entrenamiento

# Guardar los datos procesados en archivos CSV
X_train.to_csv('X_train_processed.csv', index=False)  # Guardar X_train procesado
y_train.to_csv('y_train.csv', index=False)  # Guardar y_train
X_test.to_csv('X_test_processed.csv', index=False)  # Guardar X_test procesado

# Mostrar una muestra de los datos procesados para verificar
print("X_train sample:")
print(X_train.head())
print("\nX_test sample:")
print(X_test.head())
