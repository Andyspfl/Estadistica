from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io
import base64

# Crear la aplicación Flask
app = Flask(__name__)

# Ruta principal para la página web
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para procesar los datos enviados desde el formulario
@app.route('/process', methods=['POST'])
def process():
    try:
        # Obtener los vectores x e y enviados desde el formulario
        vecx = request.form['vecx'].split()
        vecy = request.form['vecy'].split()

        # Verificar que los vectores tengan la misma longitud
        if len(vecx) != len(vecy):
            raise ValueError("Los vectores x e y deben tener la misma longitud.")
        
        # Convertir los vectores a arrays de numpy
        vecx = np.array(vecx, dtype=float)
        vecy = np.array(vecy, dtype=float)
        
        # Construir la matriz de frecuencia absoluta y relativa
        matrizF = construirMatrizFrecuenciaAbsoluta(vecx, vecy)
        matrizFR = construirMatrizFrecuenciaRelativa(matrizF)
        
        # Calcular las medidas de centralidad y dispersión de los vectores x e y
        medidas_x = calcular_medidas_centralidad_dispersion(vecx)
        medidas_y = calcular_medidas_centralidad_dispersion(vecy)
        
        # Calcular la covarianza y la recta de regresión
        covarianza = calcular_covarianza(vecx, vecy)
        recta_regresion = calcular_recta_regresion(vecx, vecy)
        
        # Calcular el valor de y para un valor de x dado
        x_input = float(request.form['x_input'])
        y_output = calcular_y(x_input, recta_regresion)
        
        # Generar gráficos y convertirlos a formato base64
        img_regresion = graficar_recta_regresion(x_input, y_output, recta_regresion, vecx, vecy)
        img_3d_abs = graficar_3d(matrizF, "Frecuencias Absolutas")
        img_3d_rel = graficar_3d(matrizFR, "Frecuencias Relativas")

        # Convertir los resultados a tipos de datos de Python
        response = {
            'matrizF': matrizF.tolist(),
            'matrizFR': matrizFR.tolist(),
            'medidas_x': {k: float(v) for k, v in medidas_x.items()},
            'medidas_y': {k: float(v) for k, v in medidas_y.items()},
            'covarianza': float(covarianza),
            'recta_regresion': {k: float(v) for k, v in recta_regresion.items()},
            'x_input': x_input,
            'y_output': y_output,
            'img_regresion': img_regresion,
            'img_3d_abs': img_3d_abs,
            'img_3d_rel': img_3d_rel
        }
        
        return jsonify(response)
    
    except Exception as e:
        return str(e)

# Función para construir la matriz de frecuencia absoluta
def construirMatrizFrecuenciaAbsoluta(vecx, vecy):
    numerosUnicosX = np.unique(vecx)
    numerosUnicosY = np.unique(vecy)
    matrizF = np.zeros((len(numerosUnicosX), len(numerosUnicosY)), dtype=int)
    
    for i, valorX in enumerate(numerosUnicosX):
        for j, valorY in enumerate(numerosUnicosY):
            contador = np.sum((vecx == valorX) & (vecy == valorY))
            matrizF[i, j] = contador
            
    return matrizF

# Función para construir la matriz de frecuencia relativa
def construirMatrizFrecuenciaRelativa(matrizF):
    total = np.sum(matrizF)
    return matrizF / total

# Función para calcular las medidas de centralidad y dispersión de un vector
def calcular_medidas_centralidad_dispersion(data):
    try:
        media_aritmetica = np.mean(data)
        moda = np.argmax(np.bincount(data.astype(int)))
        mediana = np.median(data)
        varianza = np.var(data)
        desviacion_estandar = np.std(data)
        coef_asimetria_1 = (media_aritmetica - moda) / desviacion_estandar
        coef_asimetria_2 = (3 * (media_aritmetica - mediana)) / desviacion_estandar
        coef_asimetria_3 = np.mean((data - media_aritmetica)**3) / np.mean((data - media_aritmetica)**2)**(3/2)

        return {
            "media_aritmetica": media_aritmetica,
            "moda": moda,
            "mediana": mediana,
            "varianza": varianza,
            "desviacion_estandar": desviacion_estandar,
            "coef_asimetria_1": coef_asimetria_1,
            "coef_asimetria_2": coef_asimetria_2,
            "coef_asimetria_3": coef_asimetria_3
        }
    except ValueError:
        # Si no se puede convertir a float, asumimos que es cualitativa
        return calcular_medidas_centralidad_dispersion_str(data)

# Función para calcular las medidas de centralidad y dispersión de una variable cualitativa
def calcular_medidas_centralidad_dispersion_str(data):
    moda = np.argmax(np.bincount(data.astype(int)))
    return {
        "moda": moda
    }

# Función para calcular la covarianza entre dos vectores
def calcular_covarianza(vecx, vecy):
    n = len(vecx)
    media_x = np.mean(vecx)
    media_y = np.mean(vecy)
    covarianza = np.sum((vecx - media_x) * (vecy - media_y)) / n
    return covarianza

# Función para calcular la recta de regresión lineal
def calcular_recta_regresion(vecx, vecy):
    n = len(vecx)
    media_x = np.mean(vecx)
    media_y = np.mean(vecy)
    covarianza_xy = calcular_covarianza(vecx, vecy)
    varianza_x = np.sum((vecx - media_x)**2) / n
    
    b = covarianza_xy / varianza_x
    a = media_y - b * media_x
    
    return {"a": a, "b": b}

# Función para calcular el valor de y dado un valor de x y la recta de regresión
def calcular_y(x, recta_regresion):
    a = recta_regresion["a"]
    b = recta_regresion["b"]
    return a + b * x

# Función para generar el gráfico de la recta de regresión
def graficar_recta_regresion(x_input, y_output, recta_regresion, vecx, vecy):
    plt.figure(figsize=(10, 6))
    plt.scatter(vecx, vecy, color='blue', label='Datos Originales')
    plt.plot(vecx, recta_regresion['a'] + recta_regresion['b'] * vecx, color='red', label='Recta de Regresión')
    plt.scatter(x_input, y_output, color='green', label=f'Para X = {x_input}, Y = {y_output:.2f}', marker='x', s=100)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Recta de Regresión y Datos Originales')
    plt.legend()
    plt.grid(True)
    
    # Convertir gráfica a imagen base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return f'data:image/png;base64,{plot_url}'

# Función para generar el gráfico 3D de la matriz de frecuencias
def graficar_3d(matriz, titulo):
    numerosUnicosX = np.arange(matriz.shape[0])
    numerosUnicosY = np.arange(matriz.shape[1])
    X, Y = np.meshgrid(numerosUnicosY, numerosUnicosX)
    Z = matriz
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Flatten arrays
    xpos = X.flatten()
    ypos = Y.flatten()
    zpos = np.zeros_like(xpos)
    
    # Dimensiones de las barras
    dx = dy = 0.5
    
    ax.bar3d(xpos, ypos, zpos, dx, dy, Z.flatten(), zsort='average')
    
    ax.set_xlabel('y')
    ax.set_ylabel('x')
    ax.set_zlabel('Frecuencia')
    ax.set_title(titulo)
    
    # Convertir gráfica a imagen base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return f'data:image/png;base64,{plot_url}'

if __name__ == "__main__":
    app.run(debug=True)
