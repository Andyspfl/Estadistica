from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from io import BytesIO
import base64

app = Flask(__name__)

def convertir_a_lista(datos):
    datos = datos.strip()
    lista_datos = datos.split()
    lista_datos = [int(d) if d.isdigit() else d for d in lista_datos]
    return lista_datos

def obtener_datos_unicos(lista):
    datos_unicos = list(set(lista))
    datos_unicos.sort()
    return datos_unicos

def FrecuenciaAbsoluta(lista, data):
    frec = []
    for i in lista:
        var = 0
        for j in data:
            if j == i:
                var += 1
        frec.append(var)
    return frec

def calcular_medidas_centralidad_dispersion(data):
    media_aritmetica = sum(data) / len(data)
    moda = max(set(data), key=data.count)
    mediana = sorted(data)[len(data) // 2] if len(data) % 2 != 0 else sum(sorted(data)[len(data) // 2 - 1:len(data) // 2 + 1]) / 2
    varianza = sum((x - media_aritmetica) ** 2 for x in data) / len(data)
    desviacion_estandar = math.sqrt(varianza)
    coef_asimetria_1 = (media_aritmetica - moda) / desviacion_estandar
    coef_asimetria_2 = (3 * (media_aritmetica - mediana)) / desviacion_estandar
    m3 = sum((x - media_aritmetica) ** 3 for x in data) / len(data)
    m2 = varianza
    coef_asimetria_3 = m3 / (m2 ** (3 / 2))

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

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        datos = request.form['datos']
        tipo_variable = request.form['tipo_variable']
        lista_numeros = convertir_a_lista(datos)

        # Lógica para analizar los datos según el tipo de variable
        if tipo_variable == 'Cualitativa':
            return analizar_cualitativa(lista_numeros)
        elif tipo_variable == 'Cuantitativa discreta':
            return analizar_discreta(lista_numeros)
        elif tipo_variable == 'Cuantitativa continua':
            return analizar_continua(lista_numeros)

    return render_template('index.html')

def analizar_cualitativa(datos):
    cualidades = obtener_datos_unicos(datos)
    frec_absoluta = FrecuenciaAbsoluta(cualidades, datos)
    frecuencia_relativa = [round(c / sum(frec_absoluta), 4) for c in frec_absoluta]
    frecuencia_relativa_porcentual = [round(c * 100, 2) for c in frecuencia_relativa]

    df = pd.DataFrame({
        'Cualidades': cualidades,
        'Frec. absoluta': frec_absoluta,
        'Frec. relativa': frecuencia_relativa,
        'Frec. relativa porcentual': frecuencia_relativa_porcentual,
    })

    plot_url = crear_graficos_cualitativa(df)

    return render_template('result.html', tables=[df.to_html(classes='data', index=False)], titles=df.columns.values, plot_url=plot_url)

def analizar_discreta(datos):
    cualidades = obtener_datos_unicos(datos)
    frec_absoluta = FrecuenciaAbsoluta(cualidades, datos)
    frecuencia_relativa = [round(c / sum(frec_absoluta), 4) for c in frec_absoluta]
    frecuencia_relativa_porcentual = [round(c * 100, 2) for c in frecuencia_relativa]
    frecuencia_acumulada_menor = [sum(frec_absoluta[:i+1]) for i in range(len(frec_absoluta))]
    frecuencia_acumulada_mayor = [sum(frec_absoluta[i:]) for i in range(len(frec_absoluta))]
    frecuencia_relativa_acumulada_menor = [round(sum(frecuencia_relativa[:i+1]), 4) for i in range(len(frecuencia_relativa))]
    frecuencia_relativa_acumulada_mayor = [round(sum(frecuencia_relativa[i:]), 4) for i in range(len(frecuencia_relativa))]

    df = pd.DataFrame({
        'Cualidades': cualidades,
        'Frec. absoluta': frec_absoluta,
        'Frec. absoluta acumulada menor que': frecuencia_acumulada_menor,
        'Frec. absoluta acumulada mayor que': frecuencia_acumulada_mayor,
        'Frec. relativa': frecuencia_relativa,
        'Frec. relativa porcentual': frecuencia_relativa_porcentual,
        'Frec. relativa acumulada menor que': frecuencia_relativa_acumulada_menor,
        'Frec. relativa acumulada mayor que': frecuencia_relativa_acumulada_mayor,
    })

    medidas = calcular_medidas_centralidad_dispersion(datos)
    medidas_df = pd.DataFrame(medidas.items(), columns=['Medida', 'Valor'])

    plot_url = crear_graficos_discreta(df)

    return render_template('result.html', tables=[df.to_html(classes='data', index=False), medidas_df.to_html(classes='data', index=False)], titles=df.columns.values, plot_url=plot_url)

def analizar_continua(datos):
    def intervalos(c, m, ymin):
        interval_list = []
        ini = ymin
        fin = c + ymin
        for i in range(m):
            interval_list.append(f"{ini}-{fin}")
            ini = fin
            fin += c
        return interval_list

    def MarcaDeClase(intervalos):
        marcas = []
        for intervalo in intervalos:
            x, y = map(int, intervalo.split('-'))
            marcas.append((x + y) / 2)
        return marcas

    def frecAbsoluta(intervalos, data):
        resultados = []
        for intervalo in intervalos:
            ini, fin = map(int, intervalo.split('-'))
            cantidad_numeros = sum(1 for num in data if ini <= num < fin)
            resultados.append(cantidad_numeros)
        return resultados

    def porcentualMayorMenor(relativa):
        n = len(relativa)
        mayor = []
        menor = []
        for i in range(n):
            mayor.append(relativa[i])
            menor.append(relativa[-1 * (i + 1)])
        return mayor, menor

    m = round(math.sqrt(len(datos)))
    c = round((max(datos) - min(datos)) / m)
    ymin = min(datos)
    intervalos = intervalos(c, m, ymin)
    marcaClase = MarcaDeClase(intervalos)
    frec_absoluta = frecAbsoluta(intervalos, datos)
    frecuencia_acumulada_menor = [sum(frec_absoluta[:i+1]) for i in range(len(frec_absoluta))]
    frecuencia_acumulada_mayor = [sum(frec_absoluta[i:]) for i in range(len(frec_absoluta))]
    frecuencia_relativa = [round(c / sum(frec_absoluta), 4) for c in frec_absoluta]
    frecuencia_relativa_acumulada_menor = [round(sum(frecuencia_relativa[:i+1]), 4) for i in range(len(frecuencia_relativa))]
    frecuencia_relativa_acumulada_mayor = [round(sum(frecuencia_relativa[i:]), 4) for i in range(len(frecuencia_relativa))]
    frecuencia_relativa_porcentual = [round(c * 100, 2) for c in frecuencia_relativa]
    frecuencia_relativa_porcentual_mayor, frecuencia_relativa_porcentual_menor = porcentualMayorMenor(frecuencia_relativa_porcentual)

    medidas = calcular_medidas_centralidad_dispersion(datos)

    df = pd.DataFrame({
        'Intervalos': intervalos,
        'Frec. absoluta': frec_absoluta,
        'Marca de clase': marcaClase,
        'Frec. absoluta acumulada menor que': frecuencia_acumulada_menor,
        'Frec. absoluta acumulada mayor que': frecuencia_acumulada_mayor,
        'Frec. relativa': frecuencia_relativa,
        'Frec. relativa acumulada menor que': frecuencia_relativa_acumulada_menor,
        'Frec. relativa acumulada mayor que': frecuencia_relativa_acumulada_mayor,
        'Frec. relativa porcentual': frecuencia_relativa_porcentual,
        'Frec. relativa porcentual acumulada menor que': frecuencia_relativa_porcentual_menor,
        'Frec. relativa porcentual acumulada mayor que': frecuencia_relativa_porcentual_mayor,
    })

    medidas_df = pd.DataFrame(medidas.items(), columns=['Medida', 'Valor'])

    plot_url = crear_graficos_continua(df)

    return render_template('result.html', tables=[df.to_html(classes='data', index=False), medidas_df.to_html(classes='data', index=False)], titles=df.columns.values, plot_url=plot_url)

def crear_graficos_cualitativa(df):
    plt.figure(figsize=(15, 6))

    # Diagrama Circular - Distribución de Frecuencia Relativa
    plt.subplot(1, 2, 1)
    plt.pie(df['Frec. relativa porcentual'], labels=df['Cualidades'], autopct='%1.1f%%', startangle=140, colors=sns.color_palette('viridis', len(df)))
    plt.title('Diagrama Circular - Distribución de Frecuencia Relativa')
    plt.axis('equal')

    # Diagrama de Barras - Frec. absoluta
    plt.subplot(1, 2, 2)
    sns.barplot(x='Cualidades', y='Frec. absoluta', data=df, palette='viridis')
    plt.title('Diagrama de Barras - Frec. absoluta')
    plt.xlabel('Cualidades')
    plt.ylabel('Frec. absoluta')

    plt.tight_layout()

    # Guardar el gráfico como un archivo temporal y obtener la URL para mostrarlo en la página
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return 'data:image/png;base64,{}'.format(plot_url)

def crear_graficos_discreta(df):
    plt.figure(figsize=(18, 6))

    # Polígono de Frecuencia
    plt.subplot(1, 4, 1)
    plt.plot(df['Cualidades'], df['Frec. absoluta'], marker='o', linestyle='-')
    plt.title('Polígono de Frecuencia')
    plt.xlabel('Cualidades')
    plt.ylabel('Frec. absoluta')

    # Ojiva Menor que
    plt.subplot(1, 4, 2)
    plt.plot(df['Cualidades'], df['Frec. absoluta acumulada menor que'], marker='o', linestyle='-')
    plt.title('Ojiva Menor que')
    plt.xlabel('Cualidades')
    plt.ylabel('Frec. absoluta acumulada')

    # Ojiva Mayor que
    plt.subplot(1, 4, 3)
    plt.plot(df['Cualidades'][::-1], df['Frec. absoluta acumulada mayor que'][::-1], marker='o', linestyle='-')
    plt.title('Ojiva Mayor que')
    plt.xlabel('Cualidades')
    plt.ylabel('Frec. absoluta acumulada')

    # Diagrama de Barras
    plt.subplot(1, 4, 4)
    sns.barplot(x='Cualidades', y='Frec. absoluta', data=df, palette='viridis')
    plt.title('Diagrama de Barras - Frec. absoluta')
    plt.xlabel('Cualidades')
    plt.ylabel('Frec. absoluta')

    plt.tight_layout()

    # Guardar el gráfico como un archivo temporal y obtener la URL para mostrarlo en la página
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return 'data:image/png;base64,{}'.format(plot_url)

def crear_graficos_continua(df):
    # Crear figura y establecer tamaño
    plt.figure(figsize=(18, 12))  # Aumentamos la altura para que quepan todas las gráficas
    
   # Crear el subplot para el histograma de frecuencia absoluta
    plt.subplot(2, 3, 1)
    plt.bar(df['Intervalos'], df['Frec. absoluta'], width=0.7, alpha=0.7, edgecolor='black')
    plt.title('Histograma de Frecuencia Absoluta')
    plt.xlabel('Intervalos')
    plt.ylabel('Frec. absoluta')
    
    # Subplot 2: Polígono de Frecuencia Absoluta
    plt.subplot(2, 3, 2)
    plt.plot(df['Intervalos'], df['Frec. absoluta'], marker='o', linestyle='-', color='orange')
    plt.title('Polígono de Frecuencia Absoluta')
    plt.xlabel('Intervalos')
    plt.ylabel('Frec. absoluta')
    
    # Subplot 3: Ojiva Menor que
    plt.subplot(2, 3, 3)
    plt.plot(df['Intervalos'], df['Frec. absoluta acumulada menor que'], marker='o', linestyle='-')
    plt.title('Ojiva Menor que')
    plt.xlabel('Intervalos')
    plt.ylabel('Frec. absoluta acumulada menor que')
    
    # Subplot 4: Ojiva Mayor que
    plt.subplot(2, 3, 4)
    plt.plot(df['Intervalos'], df['Frec. absoluta acumulada mayor que'], marker='o', linestyle='-')
    plt.title('Ojiva Mayor que')
    plt.xlabel('Intervalos')
    plt.ylabel('Frec. absoluta acumulada mayor que')
    
    # Subplot 5: Ojivas Menor que y Mayor que combinadas
    plt.subplot(2, 3, 5)
    plt.plot(df['Intervalos'], df['Frec. absoluta acumulada menor que'], marker='o', linestyle='-', label='Menor que')
    plt.plot(df['Intervalos'], df['Frec. absoluta acumulada mayor que'], marker='o', linestyle='-', label='Mayor que')
    plt.title('Ojivas Menor que y Mayor que')
    plt.xlabel('Intervalos')
    plt.ylabel('Frec. absoluta acumulada')
    plt.legend()
    
    # Ajustar el espaciado entre los subplots
    plt.tight_layout(pad=3.0)
    
    # Guardar el gráfico como un archivo temporal y obtener la URL para mostrarlo en la página
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()  # Cerrar la figura para liberar memoria
    
    # Devolver la URL del gráfico como formato de imagen base64
    return 'data:image/png;base64,{}'.format(plot_url)

if __name__ == '__main__':
    app.run(debug=True)
