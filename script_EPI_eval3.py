import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import statistics
import statsmodels.api as sm


"""
1. Realice una regresión univariada entre los cambios de ambas tasas ( y ).
Interprete el coeficiente de regresión obtenido (tanto su nivel como su 
significancia), así como la bondad de ajuste de la recta estimada.
"""


# --- IMPORTAMOS LOS DATAFRAMES ---

#Tasa de Politica Monetaria (TPM).
df_TPM = pd.read_excel('C:/Users/HP/OneDrive - usach.cl/Escritorio/TPM_C1.xlsx', header=2)
#----------------------------------------------------

#Federal Funds Efective Rate (FFER).
df_FFER = pd.read_excel('C:/Users/HP/OneDrive - usach.cl/Escritorio/FRB_H15.xls', header=1)

#Nota: Si se encuentra con errores al importar el archivo ´FRB_H15´ intente cambiar el formato de '.xlm' a '.xls' manualmente.


# --- LIMPIEZA DE DATAFRAMES ---

#TPM:

#Modificacion de nombre de columnas.
df_TPM.rename(columns={'1. Tasa de política monetaria (TPM) (porcentaje)': 'TPM'}, inplace=True)

#De texto a fecha.
df_TPM['Periodo'] = pd.to_datetime(df_TPM.iloc[:, 0])

#de texto a numerico.
df_TPM['TPM'] = pd.to_numeric(df_TPM['TPM'], errors='coerce')


#FFER:

#Se elimina las primeras 5 filas.
df_FFER.drop([0, 1, 2, 3, 4], inplace=True)

#Modificacion de nombre de columnas
df_FFER.rename(columns={'Series Description': 'Periodo', 'Federal funds effective rate': 'FFER'}, inplace=True)

#Conservamos las columnas de interes referentes al periodo y su tasa, eliminando el resto.
df_FFER = df_FFER[['Periodo', 'FFER']]

#De texto a fecha.
df_FFER['Periodo'] = pd.to_datetime(df_FFER['Periodo'])


#De texto a numerico.
df_FFER['FFER'] = pd.to_numeric(df_FFER['FFER'], errors='coerce')
df_FFER.dropna(subset=['FFER'], inplace=True)


#---UNIFICAMOS LOS DATAFRAMES.---
df_reg = pd.merge(df_TPM, df_FFER, on='Periodo', how='inner')


# --- REGRESIÓN UNIVARIADA (TPM = alpha + beta * FFER) ---

Y = df_reg['TPM']
X = df_reg['FFER']
X = sm.add_constant(X) # Agregamos la constante (beta 0)

modelo1 = sm.OLS(Y, X).fit()

print("\nResultados de la Regresión:")
print(modelo1.summary())

# --- PREPARACIÓN PARA EL GRÁFICO ---

#Extraemos el Año de la columna fecha para usarlo como etiqueta.
df_reg['Año'] = df_reg['Periodo'].dt.year

#Configuramos el lienzo.
plt.figure(figsize=(12, 7))

sns.scatterplot(
    data=df_reg, 
    x='FFER', 
    y='TPM', 
    hue='Año', 
    palette='viridis', 
    s=100,
    edgecolor='black',
    alpha=0.8          
)

#Linea de dispersion 
#Nota: Usamos X porque ya tiene la constante agregada.
plt.plot(df_reg['FFER'], modelo1.predict(X), color='red', linewidth=2, label='Tendencia (Regresión)')

#Etiquetas
plt.title('Evolución TPM Chile vs Tasa FED (2020-2025)', fontsize=14)
plt.xlabel('Tasa FED (FFER) %', fontsize=12)
plt.ylabel('TPM Chile %', fontsize=12)
plt.legend(title='Año del dato')
plt.grid(True, alpha=0.5)

#Escribir el mes en cada punto.
for i in range(len(df_reg)):
    plt.text(
        df_reg['FFER'].iloc[i], 
        df_reg['TPM'].iloc[i] + 0.2,
        df_reg['Periodo'].iloc[i].strftime('%b'),
        fontsize=8, 
        alpha=0.7
    )

#Imprimir el grafico.
plt.show()

#----------------------------------------------------------------------------------------------------

"""
2. Calcule la desviación estándar del tipo de cambio para determinar si el 
régimen cambiario chileno corresponde a un peg de facto (baja volatilidad) 
o a un régimen flexible (flotante). Busque el criterio de evaluación en el 
artículo de Obstfeld, Shambaugh y Taylor.
"""

# --- IMPORTACION DEL DATAFRAME ---


#Tipo de Cambio Nominal 2020-2025 (TCN).
df_TCN = pd.read_excel('C:/Users/HP/OneDrive - usach.cl/Escritorio/TCN_2000_2025.xlsx', header=5)


#Modificacion de nombre de columnas.
df_TCN.rename(columns={'dólar observado diario': 'TCN'}, inplace=True)

#De texto a numerico.
df_TCN['TCN'] = pd.to_numeric(df_TCN['TCN'], errors='coerce')
df_TCN.dropna(subset=['TCN'], inplace=True)


#Desviacion estandar del Tipo de Cambio Nominal 2020-2025 (stdev_TCN).
stdev_TCN = statistics.stdev(df_TCN['TCN'])

print(stdev_TCN)
#82.77294435648197 CLP
