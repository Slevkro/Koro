from cmath import log
from typing import Type
from numpy import var
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
from apyori import apriori
from scipy.spatial.distance import cdist
from sklearn.utils import shuffle
import seaborn as sns

from sklearn.tree import DecisionTreeRegressor

from Distancias import Distancias
from Cluster import Cluster
from Logistica import Logistica
from Pronostico import Pronostico
from Clasificacion import Clasificacion

# Configuraciones iniciales

st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden; }
    footer {visibility: hidden; }
    </style>
"""
st.markdown(hide_menu_style, unsafe_allow_html=True)

# NAVBAR prototipo
# Koro con gpu-card

# st.title("☺️ Koro")

selected = option_menu(
        menu_title=None,
        options=["KORO","Clasificacion", "Pronostico", "Regresion", "Asociacion", "Distancias", "Clustering"],
        icons=["option","grid-1x2", "cloud-drizzle", "graph-up arrow", "link", "people", "palette2"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
        "container": {"padding": "5!important", "background-color": "#10403B"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "center", "margin":"0px", "--hover-color": "#8AA6A3"},
        "nav-link-selected": {"background-color": "#8AA6A3"},
        }
    )

if selected == "KORO":
    #st.markdown("<h1 style='text-align: center;'>KORO SMART TOOL</h1>", unsafe_allow_html=True)
    #st.title(f"{selected} SMART TOOL")
    
    col1, col2 = st.columns(2)

    with col1:
        for i in range (0,20):
            st.write("")
        st.title(f"{selected} SMART TOOL")
        st.header("Inteligencia Artificial aplicada a su alcance")
        st.text("Seleccione algún algoritmo en la barra superior para poder comenzar con su análisis.")
    with col2:
        home_logo = Image.open('./Images/home-logo.png')
        st.image(home_logo)
    
    

if selected == "Asociacion":
    with st.expander("Resumen del Algoritmo"):
        st.title(f"REGLAS DE ASOCIACIÓN")
        st.markdown("Las reglas de asociación nos sirven para poder modelar relaciones entre un conjunto de transacciones del mismo tipo, por ejemplo, la lista de películas que un usuario visualiza en alguna plataforma de streaming o una lista de posibles compras a realizarse. Con base en esas relaciones el objetivo es analizar los patrones y hacer recomendaciones para generar interés.")
        st.markdown("El algoritmo Apriori nos servirá para reducir el numero de posibles combinaciones descartando transacciones para aumentar el rendimiento. Para ello es necesario seleccionar el soporte, la confianza y la elevación.")
    
    #Expander

    #st.text("Las reglas de asociación nos sirven para poder modelar relaciones entre un conjunto ")
    #st.text("de transacciones del mismo tipo, por ejemplo, la lista de películas que un usuario") 
    #st.text("visualiza en alguna plataforma de streaming o una lista de posibles compras a realizarse.") 
    #st.text("Con base en esas relaciones el objetivo es analizar los patrones y hacer recomendaciones para generar interés.")

    st.title(f"DATOS")

    #LECTURA DE LOS DATOS
    #uploaded_file = None
    uploaded_file = st.file_uploader("Choose a file", key = 'asociacion')
    if uploaded_file is not None:
        dataframe = pd.read_csv(uploaded_file)
        st.write(dataframe)
        #GENERACION DE LA TABLA DE FRECUENCIAS
        transacciones = dataframe.values.reshape(-1).tolist()
        tablaFrecuencias = pd.DataFrame(transacciones)
        tablaFrecuencias['Frecuencia'] = 0
        tablaFrecuencias = tablaFrecuencias.groupby(by=[0], as_index=False).count().sort_values(by=['Frecuencia'], ascending=False)
        tablaFrecuencias['Porcentaje'] = (tablaFrecuencias['Frecuencia'] / tablaFrecuencias['Frecuencia'].sum())
        tablaFrecuencias = tablaFrecuencias.rename(columns={0 : 'Item'})
        #IMPRESION DE LA TABLA DE FRECUENCIAS
        col1, col2 = st.columns(2)

        with col1:
            st.header("Resumen de valores leídos")
            st.markdown('A su derecha se muestra una tabla con los items mas frecuentes en la lista de transacciones proporcionada donde...')
            maximo = tablaFrecuencias.head(1)
            minimo = tablaFrecuencias.tail(1)
            st.markdown('La transaccion más frecuente corresponde al registro')
            st.write(maximo)
            st.markdown('La transaccion menos frecuente corresponde al registro')
            st.write(minimo)
            
            #st.dataframe(maximo)
        with col2:
            st.subheader('Tabla de frecuencias.')
            st.write(tablaFrecuencias)
        
        #GRAFICO DE FRECUENCIAS
        with st.expander("Grafico"):
            st.title(f"GRAFICO DE FRECUENCIAS.")
            grafica = plt.figure(figsize=(16,20), dpi=300) #Tamaño y resolucion
            plt.ylabel('Item')
            plt.xlabel('Frecuencia')
            plt.barh(tablaFrecuencias['Item'], width=tablaFrecuencias['Frecuencia'], color='blue')
            st.pyplot(grafica)
        
        #SELECCION DE PARAMETROS
        st.title(f"PARAMETROS")

        with st.form('parametros'):
            col1, col2, col3 = st.columns(3)
            with col1:
                soporte = st.number_input("Soporte", min_value=0.000, max_value=1.000, value=0.0000, step=0.001, format='%f')
            with col2:
                confianza = st.number_input("Confianza", min_value=0.0, max_value=1.0, value=0.0000, step=0.01, format='%f')  
            with col3:
                elevacion = st.number_input("Elevación", min_value=0.0, step=0.01, value=0.0000, format='%f')
            enviado = st.form_submit_button('Confirmar')
        
        #Aplicacion del algoritmo
        st.title(f"RESULTADOS")
        if enviado:
                listaTransacciones = dataframe.stack().groupby(level=0).apply(list).tolist()
                Reglas = apriori(listaTransacciones, 
                        min_support=soporte, 
                        min_confidence=confianza, 
                        min_lift=elevacion) 

                Resultados = list(Reglas)
                num_reglas = len(Resultados)
                
                tablaResultados = pd.DataFrame({
                    'Antecedente':[],
                    'Consecuente':[],
                    'Soporte': [], 
                    'Confianza': [],
                    'Elevacion': []
                })
                consecuente = ''
                for i in range (0, num_reglas):
                    aux = []
                    regla = list(Resultados[i][0])
                    for j in range (0, len(regla)):
                        if j == 0:
                            aux.append(regla[j])
                        else:
                            consecuente = consecuente + ',' + regla[j]
                    aux.append(consecuente[1:])
                    consecuente = ''
                    aux.append(Resultados[i][1])        #Soporte
                    aux.append(Resultados[i][2][0][2])  #Confianza
                    aux.append(Resultados[i][2][0][3])  #Elevacion
                    tablaResultados.loc[i] = aux
                    #st.write(aux)
                        #st.markdown(regla[j]);
                        #st.markdown(type(regla[j]));
                        #st.markdown('------------------------')
                    #st.markdown('+++++++++++++++++')
                #ResultadosD = pd.DataFrame(Resultados)
                #st.write(Resultados)
                tablaResultados['Soporte'] = tablaResultados['Soporte'] * 100
                tablaResultados['Confianza'] = tablaResultados['Confianza'] * 100
                
                c1, c2 = st.columns(2)
                with c1:
                    st.write(tablaResultados)
                with c2:
                    if num_reglas != 0:
                        st.subheader('Resumen')
                        st.markdown('Se han generado ' + str(num_reglas) + ' reglas')
                        st.markdown('Ejemplo con la ultima regla.')
                        st.markdown('En la ultima regla se relacionan los items: (' + str(aux[0]) + ') con (' + str(aux[1]) + ') teniendo un porcentaje de importancia dentro del conjunto de datos (Soporte) de ' + str(round(aux[2] * 100, 2)) + '%, un porcentaje de fiabilidad (Confianza) de ' + str(round(aux[3] * 100, 2)) + '% y por ultimo la regla representa un aumento en las posibilidades en '+str(round(aux[4], 2))+' veces (Elevacion) de que las personas que esten interesadas en (' + str(aux[0]) + ') tambien lo esten en (' +  str(aux[1]) + ').')
                    else:
                        st.subheader('No se ha generado ninguna regla')
                        st.markdown('Intente cambiando los parametros seleccionados')
                
            
            



        


if selected == "Distancias":
    with st.expander("Resumen del Algoritmo"):
        st.title(f"Distancias")
        st.markdown("Las métricas de distancia nos ayudaran a obtener índices que miden el nivel de similitud entre los objetos los cuales pueden ser, por ejemplo, usuarios, rutas, productos, etc. a su vez éstas luego servirán como la entrada para nuestros algoritmos de inteligencia artificial.")
        st.markdown("Debemos de tener en cuenta que para poder obtener estas distancias debemos de filtrar las características para que sean numéricas. Además de que cada métrica de distancia nos servirá para propósitos o soluciones diferentes, por ejemplo, ***Chebyshev*** nos puede ser útil para procesos industriales, mientras que ***Manhattan*** para medir distancias un poco más apegadas a la realidad donde no es tan factible irse por la distancia mas corta la cual sería una línea diagonal como la hipotenusa de las distancias ***Euclidianas***, sin embargo, en la actualidad se buscan distancias más factibles las cuales describen una distancia más generalizada o flexible como ***Minkowsky*** que define un parámetro lambda para obtener este efecto la cual por ejemplo con valor de 1.5 es un punto medio entre Euclidiana y Manhattan.")


    st.title(f"DATOS")
    flexometro = Distancias()
    flexometro.LeerDatos(id = 'distancias')
    
    if flexometro.archivo is not None:
        st.title(f"METRICAS")
        options = st.multiselect(
        'Selecciona las metricas de distancia con las que desees trabajar, puedes escribirla o seleccionarla si no se muestra inicialmente.',
        ['Euclideana', 'Chebyshev', 'Manhattan', 'Minkowski'],
        ['Euclideana', 'Manhattan'])

        #st.write('You selected:', options)

        for option in options:
            distancia = pd.DataFrame(flexometro.CalcularDistancia(option))
            
            st.markdown("<h1 style='text-align: center;'>" + option + "</h1>", unsafe_allow_html=True)
            #Seleccion del rango
            values = st.slider(
                'Selecciona el rango de valores de la matriz de distancias.',
                0, distancia.shape[0]-1, (0, distancia.shape[0]-1), key = option)
            #st.write('Values:', values)
            
            c1, c2, c3 = st.columns(3)
            with c1:
                #st.markdown('Introduce un numero entero para el redondeo.')
                redondeo = st.number_input('Introduce un numero entero para el redondeo.', min_value=0, max_value=8, value=3, step=1, format='%d', key = option)
            with c2:
                if option == 'Minkowski':
                    la_lambda = st.number_input('Introduce un numero del parametro lambda.', min_value=0.0, max_value=3.0, value=1.5, step=0.1, format='%f', key = option)
            with c3:
                optimizacion = st.radio(
                "Selecciona el proceso que buscas para optimizacion.",
                ('Ninguno', 'Normalizar', 'Estandarizar'), key = option)

                if optimizacion == 'Normalizar':
                    distancia = flexometro.CalcularDistanciaNormalizada(option, values[0], values[1])
                elif optimizacion == 'Estandarizar':
                    distancia = flexometro.CalcularDistanciaEstandarizada(option, values[0], values[1])
                elif optimizacion == 'Ninguno':
                    distancia = flexometro.CiertasDistancias(values[0], values[1], option)
            st.write(distancia)
        
        with st.expander("Correlacion de atributos."):
            st.title(f"CORRELACION")
            flexometro.Correlaciones()
            co1, co2= st.columns(2)
            with co1: 
                st.subheader('Matriz de correlaciones.')
                st.write(flexometro.matrizCorrelaciones)
            with co2:
                st.subheader('Atributos del conjunto de datos.')
                st.markdown('Se tienen en total **' + str(flexometro.matrizCorrelaciones.shape[0]) + '** atributos en el conjunto de datos.')
                #st.write(flexometro.matrizCorrelaciones.columns.values)
                #st.write(flexometro.matrizCorrelaciones.shape[1])
                atributos = list(flexometro.matrizCorrelaciones.columns.values)
                for i in range (0, flexometro.matrizCorrelaciones.shape[1], 2):
                    st.text('*  ' + atributos[i] + '        * ' + atributos[i+1])
                
            
            st.title(f"MAPA DE CALOR.")
            mapa_de_calor =plt.figure(figsize=(14,7))
            MatrizInf = np.triu(flexometro.matrizCorrelaciones)
            sns.heatmap(flexometro.matrizCorrelaciones, cmap='RdBu_r', annot=True, mask=MatrizInf)
            st.pyplot(mapa_de_calor)

            #DstEuclidiana = cdist(flexometro.datos, flexometro.datos, metric='euclidean')
            #print(type(DstEuclidiana))
            #MEuclidiana = pd.DataFrame(DstEuclidiana)
            #st.write(MEuclidiana)
            


if selected == "Clustering":
    with st.expander("Resumen del Algoritmo"):
        st.title(f"Clustering")
        
        st.markdown("Los algoritmos de clustering nos sirven para segmentar un determinado conjunto de datos en diversos grupos, cada cual con sus características y particularidades que los diferencian. Estas segmentaciones pueden ser muy útiles para obtener perfiles de usuario, patrones climáticos, etc.")
        st.markdown("La técnica de clustering particional puede llegar a resultar muy útil cuando lo que se busca es una segmentación con un mayor número de registros ya que cuando este número incrementa la implementación de un árbol para organizar a los clústeres (Clustering Jerárquico) se vuelve inestable y más lenta donde el hecho de estandarizar y normalizar los datos si bien puede ayudar al rendimiento no lo hace a tal punto de hacer viable el clustering jerárquico, por el contrario hay que tener en cuenta que para poder utilizar la técnica particional se tiene que conocer cuál es el número de clústeres con el que queremos terminar, métrica que es muy importante ya que si se aplica mal podría llevar a tener sesgos en el análisis de los clústeres pero que requiere de un procesamiento extra para realizarse (Método del codo o la rodilla).")
        

    st.title(f"DATOS")
    colador = Cluster()
    colador.LeerDatos(id = 'clustering')

    if colador.archivo is not None:
        st.title(f"SELECCION DE CARACTERISTICAS")
        st.write('Matriz de correlaciones')
        mapa_de_calor = colador.getMapaDeCalor()
        st.write(colador.distancias.matrizCorrelaciones)

        c1, c2 = st.columns(2)

        with c1:
            st.subheader('Seleccion de caracteristicas.')
            colador.variables_iniciales = list(colador.distancias.matrizCorrelaciones.columns.values)
            options = st.multiselect(
            'Elimina las variables con las que no desees trabajar, puedes observar sus relaciones en el mapa de calor a la derecha.',
            colador.variables_iniciales,
            colador.variables_iniciales)

            colador.variables_finales = list(options)
            st.markdown('La matriz de datos con sus variables finales se encuentra a continuacion.')
            matriz_variables_finales = colador.getVariablesFinales()
            st.write(matriz_variables_finales)

        #st.write('You selected:', options)
        with c2: 
            st.subheader('Mapa de calor.')
            st.pyplot(mapa_de_calor)

        st.subheader('Escalado de datos.')
        optimizacion = st.radio(
                "Selecciona el proceso que buscas para optimizacion del algoritmo.",
                ('Ninguno', 'Normalizar', 'Estandarizar'), key = 'clustering')
        if optimizacion == 'Normalizar':
            matriz_variables_finales = pd.DataFrame(colador.NormalizarMatriz(matriz_variables_finales))
            
        elif optimizacion == 'Estandarizar':
            matriz_variables_finales = pd.DataFrame(colador.EstandarizarMatriz(matriz_variables_finales))
        
        matriz_variables_finales.columns = colador.variables_finales
        
        st.markdown('Matriz de entrada a los algoritmos de clustering.')
        st.write(matriz_variables_finales)

        st.title(f"CLUSTERING JERARQUICO")
        st.markdown('***El arbol puede tardar un tiempo dependiendo del numero de registros y la cantidad de variables seleccionada.***')
        options = st.multiselect(
            'Selecciona las metricas de distancia con las que desees trabajar, puedes escribirla o seleccionarla si no se muestra inicialmente.',
            ['Euclideana', 'Chebyshev', 'Manhattan', 'Minkowski'],
            ['Euclideana', 'Manhattan'])
        distancias = {'Euclideana': 'euclidean', 'Chebyshev': 'chebyshev', 'Manhattan': 'cityblock', 'Minkowski': 'minkowski'}
        for option in options:
            st.markdown("<h1 style='text-align: center;'>Cluster con Distancia " + option + "</h1>", unsafe_allow_html=True)
            arbol = colador.getClusterJerarquico(distancias[option], matriz_variables_finales)
            #Meter los arboles a una lista y preguntar si es not null para que no los este haciendo a cada rato
            st.pyplot(arbol)
            colu1, colu2 = st.columns(2)

            with colu1: 
                st.subheader('Número de Clusters.')
                st.markdown('Basandose en el arbol que se le muestra en la parte superior, introduzca el numero de clusters.')
            with colu2:
                num_clusters = st.number_input('Introduce un numero de clusters.', min_value=0, max_value=40, value=2, step=1, format='%d', key = option + str(' jer'))
            
            num_elementos, centroides_jerarquico = colador.getCentroidesJerarquico(matriz_variables_finales, num_clusters, distancias[option])

            column1, column2 = st.columns(2)

            with column1:
                st.subheader('Conteo de elementos por cluster.')
                st.write(num_elementos)
            with column2:
                st.subheader('Centroides del conjunto de datos.')
                st.write(centroides_jerarquico)

        st.title(f"CLUSTERING PARTICIONAL")
        pilar1, pilar2, pilar3 = st.columns(3)
        with pilar1:
            for i in range (0, 3):
                st.write('')
            st.subheader('Metodo de la rodilla.')
            st.markdown('A continuacion se mostrara la grafica del medodo de la rodilla (Knee Method), intente ubicar cual es el punto donde cambia de direccion abruptamente el cual indica el numero de clusteres optimo para la segmentacion.')
        
        with pilar2:
            st.write()
            for i in range (0, 8):
                st.write('')
            num_clusters_par = st.number_input('Introduce un numero de clusters.', min_value=0, max_value=40, value=2, step=1, format='%d', key = option + str(' par'))

        with pilar3:
            rodilla = colador.getRodilla(matriz_variables_finales)
            st.write(rodilla)
        
        with st.expander("Implementacion del metodo de la rodilla"):
            st.title(f"Distancias")
            rodilla_ = colador.getRodillaExacta()
            st.pyplot(fig=rodilla_, clear_figure=None)
        
        num_clusters_par_final, centroides_particional = colador.ClusterParticional(matriz_variables_finales, num_clusters_par)
        columna1, columna2 = st.columns(2)

        with columna1:
            st.subheader('Conteo de elementos por cluster.')
            st.write(num_clusters_par_final)
        with columna2:
            st.subheader('Centroides del conjunto de datos.')
            st.write(centroides_particional)
        
if selected == "Regresion":
    with st.expander("Resumen del Algoritmo"):
        st.title(f"REGRESION LOGISTICA")
        st.markdown("La regresión logística busca predecir valores binarios los cuales corresponden a la etiqueta de los registros (0/1, verdadero/falso, etc). Lo hace aplicando una transformación a la regresión lineal ya que por sí sola una regresión lineal no nos serviría para predecir esta variable binaria.")

        columna1, columna2 = st.columns(2)

        with columna1:
            st.markdown("Para hacer la transformación se usa la función sigmoide.")
            sigmoide = Image.open('./Images/sigmoid.png')
            st.image(sigmoide)
        with columna2:
            for i in range (5):
                st.markdown('')
            st.markdown('Dicha función asigna una probabilidad la cual puede ir de 0 a 1, donde si es mayor a 0.5 asigna el valor de 1 y si es menor o igual a 0.5 entonces asigna el valor de 0.')


    st.title(f"DATOS")

    #LECTURA DE LOS DATOS
    logistico = Logistica()
    logistico.LeerDatos(id = 'logistica')

    if logistico.archivo is not None:
        logistico.datos = logistico.datos.dropna()
        st.title(f"SELECCION DE CARACTERISTICAS")
        st.write('Matriz de correlaciones')
        mapa_de_calor = logistico.getMapaDeCalor()
        st.write(logistico.distancias.matrizCorrelaciones)

        c1, c2 = st.columns(2)

        with c1:
            st.subheader('Seleccion de caracteristicas.')
            logistico.variables_iniciales = list(logistico.distancias.matrizCorrelaciones.columns.values)
            options = st.multiselect(
            'Elimina las variables con las que no desees trabajar, puedes observar sus relaciones en el mapa de calor a la derecha.',
            logistico.variables_iniciales,
            logistico.variables_iniciales)

            logistico.variables_finales = list(options)
            st.markdown('La matriz de datos con sus variables finales se encuentra a continuacion.')
            matriz_variables_finales = logistico.getVariablesFinales()
            st.write(matriz_variables_finales)

        #st.write('You selected:', options)
        with c2: 
            st.subheader('Mapa de calor.')
            st.pyplot(mapa_de_calor)
        
        st.write('')
        st.subheader('Defina la variable clase y sus valores.')

        col1, col2, col3 = st.columns(3)

        with col1:
            variable_clase = st.selectbox(
                'Variable clase',
                logistico.datos.columns)
            st.write('Variable Clase:', variable_clase)
        with col2:
            primer_clase = st.text_input('PRIMERA', 'A')
            st.write('Nombre de la primera clase: ', primer_clase)
        with col3:
            segunda_clase = st.text_input('SEGUNDA', 'B')
            st.write('Nombre de la primera clase: ', segunda_clase)
        
        #st.write(logistico.datos)
        clases = []
        contador = 0
        for clase in pd.unique(logistico.datos[variable_clase]):
            #st.write(clase)
            clases.append(clase)            
            if contador > 1:
                st.header('LA VARIABLE CLASE TIENE MAS DE DOS CLASES O POSIBLES VALORES')
                contador = contador + 1
                break
            contador = contador + 1
        #st.write(contador)
        if contador == 2:

            st.write('')
            st.subheader('Defina el numero de registros por clase para el conjunto de datos.')
            logistico.datos[variable_clase] = logistico.datos[variable_clase].replace({clases[0]: 0, clases[1]: 1})

            colu1, colu2, colu3 = st.columns(3)
            with colu1:
                #st.write(logistico.datos)
                st.markdown('Cantidad de elementos por clase.')
                tamanios = logistico.datos.groupby(variable_clase).size()
                st.write(tamanios)
            with colu2:
                texto = "Tamaño de la clase " + str(primer_clase)
                clase_1_size = st.number_input(texto, min_value=0, max_value=tamanios[0], value=tamanios[0], step=1, format='%d')
            with colu3: 
                texto = "Tamaño de la clase " + str(segunda_clase)
                clase_2_size = st.number_input(texto, min_value=0, max_value=tamanios[1], value=tamanios[1], step=1, format='%d')
            
            ceros = logistico.datos[logistico.datos[variable_clase] == 0.0 ]
            ceros = ceros.sample(n=clase_1_size, random_state=1)

            unos = logistico.datos[logistico.datos[variable_clase] == 1.0 ]
            unos = unos.sample(n=clase_2_size, random_state=1)
            logistico.datos = pd.concat([ceros, unos])
            logistico.datos = shuffle(logistico.datos)
            st.write(logistico.datos)

            #X = np.array(logistico.datos[logistico.variables_finales].drop([variable_clase]))
            #PUEDE QUE SE MUERA PORQUE NO SE CONSIDERA LA VARIABLE CLASE DENTRO DE LAS VARIABLES FINALES
            #logistico.variables_finales.append(variable_clase)

            #st.write(logistico.variables_finales)
            st.subheader('CONJUNTOS DE DATOS PARA ENTRENAMIENTO')
            column1, column2, column3 = st.columns(3)

            with column1: 
                st.markdown('Conjunto de variables independientes')
                X = np.array(logistico.datos[logistico.variables_finales])
                st.write(X)

            with column2:
                st.markdown('Conjunto de la variable clase')
                Y = np.array(logistico.datos[[variable_clase]])
                st.write(Y)

            with column3:
                X_train, X_validation, Y_train, Y_validation = logistico.SeparaConjunto(X, Y)
                logistico.crearModelo(X_train, Y_train)
                probabilidades = logistico.getMatrizProbabilidades(X_validation)
                st.write('Probabilidades de los registros')
                st.write(probabilidades)
           

            
            Y_Clasificacion, matriz_clasificacion = logistico.getMatrizClasificacion(Y_validation, X_validation)
            pil1, pil2 = st.columns(2)

            with pil1: 
                score = logistico.getScore(X_validation, Y_validation)
                st.subheader('Score')
                st.header('Precision del modelo: ' + str(score * 100))
            with pil2: 
                st.write('Matriz de clasificacion')
                st.write(matriz_clasificacion)
            
            

            st.header('Ecuacion final de regresion.')
            intercepto, coeficientes = logistico.getEcuacion()
            ecuacion = "y = " + str(intercepto)
            i = 0
            for coeficiente in coeficientes[0]:
                if coeficiente > 0:
                    ecuacion = ecuacion + ' + ' + str(coeficiente)
                else:
                    ecuacion = ecuacion + ' ' + str(coeficiente)
                ecuacion = ecuacion + '( ' + logistico.variables_finales[i] + ') '
                i = i+1
                #st.write(i)
                #st.write('Coeficiente: '+ str(coeficiente[i]))
            st.subheader(ecuacion)

            st.subheader('Generar nuevos pronosticos.')
            
            columnas1, columnas2 = st.columns(2)
            with columnas1:
                valores_clasificasion = {}
                for variable in logistico.variables_finales:
                    number = st.number_input(variable, min_value=0, max_value=10000000000, value=1, step=1, format='%d')
                    valores_clasificasion[variable] = [number]
                    #st.write('The current number is ', number)
            with columnas2:
                
                pronostico = logistico.genPronosticos(valores_clasificasion)
                if (pronostico == 0):
                    st.write(primer_clase)
                    st.subheader('Pronostico: ' + str(primer_clase))
                else:
                    st.subheader('Pronostico: ' + str(segunda_clase))
                
                
            

if selected == "Pronostico":
    with st.expander("Resumen del Algoritmo"):
        st.title(f"ARBOLES DE DECISION")
        st.markdown("Los árboles de decisión son algoritmos los cuales toman una serie de variables independientes como entrada y las analizan una por una en cada uno de los diferentes niveles del árbol partiendo de una raíz para obtener el determinado valor de una variable dependiente que en el caso de un pronostico es continua.")
        st.title(f"BOSQUES ALEATORIOS")
        st.markdown("Los bosques aleatorios son algoritmos que buscan generalizar máslas soluciones que nos pueden entregar los árboles aleatorios mediante la combinación de varios de estos árboles en la misma estructura de tal forma que cada uno de ellos aporte una solución similar para después llegar a un consenso evitando además un sobreajuste en los datos.")     

    st.title(f"DATOS")

    #LECTURA DE LOS DATOS
    pronostico = Pronostico()
    pronostico.LeerDatos(id = 'pronostico')

    if pronostico.archivo is not None:
        st.title(f"SELECCION DE CARACTERISTICAS")
        st.write('Matriz de correlaciones')
        mapa_de_calor = pronostico.getMapaDeCalor()
        st.write(pronostico.distancias.matrizCorrelaciones)

        c1, c2 = st.columns(2)

        with c1:
            st.subheader('Defina las variables dependientes.')
            pronostico.variables_iniciales = list(pronostico.distancias.matrizCorrelaciones.columns.values)
            options = st.multiselect(
            'Elimina las variables con las que no desees trabajar, puedes observar sus relaciones en el mapa de calor a la derecha.',
            pronostico.variables_iniciales,
            pronostico.variables_iniciales)

            pronostico.variables_finales = list(options)
            st.markdown('La matriz de datos con sus variables finales se encuentra a continuacion.')
            matriz_variables_finales = pronostico.getVariablesFinales()
            st.write(matriz_variables_finales)
            st.write('')
            st.subheader('Defina la variable pronostico')
            variable_pronostico = st.selectbox(
                'Variable Pronostico',
                pronostico.datos.columns)
            st.write('Variable Pronostico Seleccionada:', variable_pronostico)

        #st.write('You selected:', options)
        with c2: 
            st.write('')
            st.subheader('Mapa de calor.')
            st.pyplot(mapa_de_calor)
        
        #st.write(type(pronostico.datos[variable_pronostico][0]))
        
        if (type(pronostico.datos[variable_pronostico][0]) != str):
            #DEFINICION DEL CONJUNTO DE DATOS
            st.subheader('CONJUNTOS DE DATOS PARA ENTRENAMIENTO')
            column1, column2, column3 = st.columns(3)

            with column1: 
                st.markdown('Conjunto de variables independientes')
                X = np.array(pronostico.datos[pronostico.variables_finales])
                st.write(X)

            with column2:
                st.markdown('Conjunto de la variable pronostico')
                Y = np.array(pronostico.datos[[variable_pronostico]])
                st.write(Y)
            
            X_train, X_test, Y_train, Y_test = pronostico.SeparaConjunto(X, Y)

            with st.expander("Arboles de decision"):
                st.subheader('Defina los hiperparametros del arbol')
                col1, col2, col3 = st.columns(3)

                with col1: 
                    profundidad = st.slider('Profundidad', min_value = 3, max_value = 30, step = 1)
                    #st.write(profundidad)
                with col2: 
                    divisiones = st.slider('Numero de registros minimo para dividir', min_value = 2,max_value = 50, step = 1)
                    #st.write(divisiones)
                with col3: 
                    hojas = st.slider('numero minimo de registros en las hojas', min_value = 2,max_value = 50, step = 1)
                    #st.write(hojas)

                pronostico.crearArbol(profundidad, divisiones, hojas, X_train, Y_train)

                valores_pronosticados = pronostico.generarPronosticoArbol(X_test)

                #comparacion = pd.DataFrame(Y_test, valores_pronosticados)
                #pronostico.generaGraficaComparacion('var_x', 'var_y', Y_test, valores_pronosticados)
                #st.pyplot(comparacion)

                colu1, colu2 = st.columns(2)

                with colu1:
                    st.header('Score')
                    pronostico.genScore(Y_test, valores_pronosticados)
                    st.subheader('Precision final del modelo: ' + str(pronostico.score * 100))

                    for i in range (3): st.write('')
                    st.subheader('Metricas de desempeño')
                    mae, mse, rmse = pronostico.getMetricas(Y_test, valores_pronosticados)
                    st.write("MAE: %.4f" % mae)
                    st.write("MSE: %.4f" % mse)
                    st.write("RMSE: %.4f" % rmse)

                    for i in range (3): st.write('')
                    st.subheader('Variables ordenadas por importancia')
                    importancia_variables = pronostico.getImportancia()
                    st.write(importancia_variables)
                with colu2:
                    st.subheader('ARBOL')
                    arbol_desicion = pronostico.genArbol()
                    st.graphviz_chart(arbol_desicion)

                st.subheader('Generar nuevos pronosticos.')
                
                columnas1, columnas2 = st.columns(2)
                with columnas1:
                    valores_clasificasion = {}
                    for variable in pronostico.variables_finales:
                        number = st.number_input(variable, min_value=0, max_value=10000000000, value=1, step=1, format='%d')
                        valores_clasificasion[variable] = [number]
                        #st.write('The current number is ', number)
                with columnas2:
                    
                    valor = pronostico.genPronosticosArbol(valores_clasificasion)
                    st.subheader('Pronostico: ' + str(valor))
            
            with st.expander("Bosques aleatorios"):
                st.header('MODELO DE BOSQUES ALEATORIOS')
                st.subheader('Defina los hiperparametros del bosque')
                col1, col2, col3, col4, col5 = st.columns(5)

                with col1: 
                    profundidad = st.slider('Profundidad bosque', min_value = 3, max_value = 30, step = 1)
                    #st.write(profundidad)
                with col2: 
                    divisiones = st.slider('Numero de registros minimo para dividir bosque', min_value = 2,max_value = 50, step = 1)
                    #st.write(divisiones)
                with col3: 
                    hojas = st.slider('Numero minimo de registros en las hojas bosque', min_value = 2,max_value = 50, step = 1)
                    #st.write(hojas)
                with col4: 
                    no_estimadores = st.slider('Numero minimo de estimadores bosque', min_value = 15,max_value = 300, step = 1)
                    #st.write(no_estimadores)
                with col5: 
                    no_variables = st.slider('numero minimo de variables a considerar bosque', min_value = 2,max_value = len(pronostico.variables_finales), step = 1)
                    #st.write(no_variables)

                pronostico.crearBosque(no_estimadores, no_variables, profundidad, divisiones, hojas, X_train, Y_train)

                valores_pronosticados = pronostico.generarPronosticoBosque(X_test)

                #comparacion = pd.DataFrame(Y_test, valores_pronosticados)
                #pronostico.generaGraficaComparacion('var_x', 'var_y', Y_test, valores_pronosticados)
                #st.pyplot(comparacion)

                colu1, colu2 = st.columns(2)

                with colu1:
                    for i in range (3): st.write('')
                    st.subheader('Numero de arbol en el bosque.')
                    no_estimador = st.slider('Numero de estimador en el bosque', min_value = 0,max_value = no_estimadores-1, step = 1)
                    

                    
                    st.header('Score')
                    pronostico.genScore(Y_test, valores_pronosticados)
                    st.subheader('Precision final del modelo: ' + str(pronostico.score * 100))
                    
                    for i in range (3): st.write('')
                    st.subheader('Metricas de desempeño')
                    mae, mse, rmse = pronostico.getMetricas(Y_test, valores_pronosticados)
                    st.write("MAE: %.4f" % mae)
                    st.write("MSE: %.4f" % mse)
                    st.write("RMSE: %.4f" % rmse)

                    for i in range (3): st.write('')
                    st.subheader('Variables ordenadas por importancia')
                    importancia_variables = pronostico.getImportancia()
                    st.write(importancia_variables)

                    
                with colu2:
                    st.subheader('ARBOL')
                    muestra_bosque = pronostico.genBosque(no_estimador)
                    st.graphviz_chart(muestra_bosque)

                #st.write('Reporte')
                #reporte = pronostico.getReporte()
                #st.write(reporte)

                st.subheader('Generar nuevos pronosticos.')
                
                columnas1, columnas2 = st.columns(2)
                with columnas1:
                    valores_clasificasion = {}
                    for variable in pronostico.variables_finales:
                        number = st.number_input(variable + variable, min_value=0, max_value=10000000000, value=1, step=1, format='%d')
                        valores_clasificasion[variable] = [number]
                        #st.write('The current number is ', number)
                with columnas2:
                    
                    valor = pronostico.genPronosticosBosque(valores_clasificasion)
                    for i in range(3): st.write('')
                    st.subheader('Pronostico: ' + str(valor))

        else:
            st.write('La variable pronostoco no es numerica')
            

if selected == "Clasificacion":
    with st.expander("Resumen del Algoritmo"):
        st.title(f"ARBOLES DE DECISION")
        st.markdown("Los árboles de decisión son algoritmos los cuales toman una serie de variables independientes como entrada y las analizan una por una en cada uno de los diferentes niveles del árbol partiendo de una raíz para obtener el determinado valor de una variable dependiente que en el caso de una clasificacion esta variable es nominal (A, B, C, etc.) o discreta.")
        st.title(f"BOSQUES ALEATORIOS")
        st.markdown("Los bosques aleatorios son algoritmos que buscan generalizar máslas soluciones que nos pueden entregar los árboles aleatorios mediante la combinación de varios de estos árboles en la misma estructura de tal forma que cada uno de ellos aporte una solución similar para después llegar a un consenso evitando además un sobreajuste en los datos.")



    st.title(f"DATOS")

    #LECTURA DE LOS DATOS
    clasificacion = Clasificacion()
    clasificacion.LeerDatos(id = 'clasificacion')

    if clasificacion.archivo is not None:
        st.title(f"SELECCION DE CARACTERISTICAS")
        st.write('Matriz de correlaciones')
        mapa_de_calor = clasificacion.getMapaDeCalor()
        st.write(clasificacion.distancias.matrizCorrelaciones)

        c1, c2 = st.columns(2)

        with c1:
            st.subheader('Seleccion de caracteristicas.')
            clasificacion.variables_iniciales = list(clasificacion.distancias.matrizCorrelaciones.columns.values)
            options = st.multiselect(
            'Elimina las variables con las que no desees trabajar, puedes observar sus relaciones en el mapa de calor a la derecha.',
            clasificacion.variables_iniciales,
            clasificacion.variables_iniciales)

            clasificacion.variables_finales = list(options)
            st.markdown('La matriz de datos con sus variables finales se encuentra a continuacion.')
            matriz_variables_finales = clasificacion.getVariablesFinales()
            st.write(matriz_variables_finales)

        #st.write('You selected:', options)
        with c2: 
            st.subheader('Mapa de calor.')
            st.pyplot(mapa_de_calor)
        

        st.write('')
        st.subheader('Defina la variable clase y sus valores.')

        col1, col2, col3 = st.columns(3)

        with col1:
            variable_clase = st.selectbox(
                'Variable clase',
                clasificacion.datos.columns)
            st.write('Variable Clase:', variable_clase)
            clases = []
            contador = 0
            for clase in pd.unique(clasificacion.datos[variable_clase]):
                #st.write(clase)
                clases.append(clase)            
                if contador > 1:
                    st.header('LA VARIABLE CLASE TIENE MAS DE DOS CLASES O POSIBLES VALORES')
                    contador = contador + 1
                    break
                contador = contador + 1
            #st.write(contador)
            if contador == 2:

                st.write('')
                st.subheader('Tamaño de cada una de las clases.')
                clase_1_size = 2
                clase_2_size = 2
                st.markdown('Cantidad de elementos por clase.')
                tamanios = clasificacion.datos.groupby(variable_clase).size()
                st.write(tamanios)

                primer_clase = tamanios.axes[0][0]
                segunda_clase = tamanios.axes[0][1]
        
        if contador == 2:
            with col2:
                for i in range(10): st.write('')
                texto = "Numero de registros de la primera clase" 
                clase_1_size = st.number_input(texto, min_value=0, max_value=tamanios[0], value=tamanios[0], step=1, format='%d')
            with col3:
                for i in range(10): st.write('')
                texto = "Numero de registros de la segunda clase"
                clase_2_size = st.number_input(texto, min_value=0, max_value=tamanios[1], value=tamanios[1], step=1, format='%d')

            ceros = clasificacion.datos[clasificacion.datos[variable_clase] == primer_clase ]
            #st.write(ceros[variable_clase].size)
            ceros = ceros.sample(n=clase_1_size, random_state=1)


            unos = clasificacion.datos[clasificacion.datos[variable_clase] == segunda_clase ]
            #st.write(unos[variable_clase].size)
            unos = unos.sample(n=clase_2_size - 1, random_state=1)
            clasificacion.datos = pd.concat([ceros, unos])
            clasificacion.datos = shuffle(clasificacion.datos)
                
            #st.write(clasificacion.datos)

            st.subheader('CONJUNTOS DE DATOS PARA ENTRENAMIENTO')
            column1, column2= st.columns(2)

            with column1: 
                st.markdown('Conjunto de variables independientes')
                X = np.array(clasificacion.datos[clasificacion.variables_finales])
                st.write(X)

            with column2:
                st.markdown('Conjunto de la variable clase')
                Y = np.array(clasificacion.datos[[variable_clase]])
                st.write(Y)
            
            st.header('ALGORITMOS DE CLASIFICACION')
            
            with st.expander("Arboles de desicion"):
                st.title(f"Arboles de desicion")

                X_train, X_validation, Y_train, Y_validation = clasificacion.SeparaConjunto(X, Y)

                st.subheader('Defina los hiperparametros del arbol')
                col1, col2, col3 = st.columns(3)

                with col1: 
                    profundidad = st.slider('Profundidad', min_value = 3, max_value = 30, step = 1)
                        #st.write(profundidad)
                with col2: 
                    divisiones = st.slider('Numero de registros minimo para dividir', min_value = 2,max_value = 50, step = 1)
                        #st.write(divisiones)
                with col3: 
                    hojas = st.slider('numero minimo de registros en las hojas', min_value = 2,max_value = 50, step = 1)
                        #st.write(hojas)
                
                clasificacion.genArbolClasificacion(profundidad, divisiones, hojas, X_train, Y_train)

                valores_pronosticados = clasificacion.getClasificacionArbol(X_validation)

                colu1, colu2 = st.columns(2)

                with colu1:
                    st.header('Score')
                    score = clasificacion.getScore(X_validation, Y_validation)
                    st.subheader('Precision final del modelo: ' + str(score * 100))

                    for i in range (3): st.write('')
                    st.subheader('Matriz de clasificasion')
                    matriz_clasificasion, Y_Clasificacion = clasificacion.getMatrizClasificacion(X_validation, Y_validation)
                    st.write(matriz_clasificasion)

                    for i in range (3): st.write('')
                    st.subheader('Variables ordenadas por importancia')
                    importancia_variables = clasificacion.getImportancia()
                    st.write(importancia_variables)

                with colu2:
                    st.write('ARBOL')
                    arbol_desicion = clasificacion.getArbolClasificacion(Y_Clasificacion)
                    st.graphviz_chart(arbol_desicion)
                
                st.subheader('Generar nuevos pronosticos.')
                
                columnas1, columnas2 = st.columns(2)
                with columnas1:
                    valores_clasificasion = {}
                    for variable in clasificacion.variables_finales:
                        number = st.number_input(variable, min_value=0, max_value=10000000000, value=1, step=1, format='%d')
                        valores_clasificasion[variable] = [number]
                        #st.write('The current number is ', number)
                with columnas2:
                    
                    pronostico = clasificacion.genPronosticosArbol(valores_clasificasion)
                    if (pronostico == 0):
                        st.write(primer_clase)
                        st.subheader('Pronostico: ' + str(primer_clase))
                    else:
                        st.subheader('Pronostico: ' + str(segunda_clase))


            
            with st.expander("Bosques Aleatorios"):
                st.subheader('Defina los hiperparametros del bosque')
                col1, col2, col3, col4, col5 = st.columns(5)

                with col1: 
                    profundidad = st.slider('Profundidad bosque', min_value = 3, max_value = 30, step = 1)
                    #st.write(profundidad)
                with col2: 
                    divisiones = st.slider('Numero de registros minimo para dividir bosque', min_value = 2,max_value = 50, step = 1)
                    #st.write(divisiones)
                with col3: 
                    hojas = st.slider('Numero minimo de registros en las hojas bosque', min_value = 2,max_value = 50, step = 1)
                    #st.write(hojas)
                with col4: 
                    no_estimadores = st.slider('Numero minimo de estimadores bosque', min_value = 15,max_value = 300, step = 1)
                    #st.write(no_estimadores)
                with col5: 
                    no_variables = st.slider('numero minimo de variables a considerar bosque', min_value = 2,max_value = len(clasificacion.variables_finales), step = 1)
                    #st.write(no_variables)

                clasificacion.genBosqueClasificasion(no_estimadores, no_variables, profundidad, divisiones, hojas, X_train, Y_train)

                valores_pronosticados = clasificacion.valoresPronosticados(X_validation)



                #comparacion = pd.DataFrame(Y_test, valores_pronosticados)
                #pronostico.generaGraficaComparacion('var_x', 'var_y', Y_test, valores_pronosticados)
                #st.pyplot(comparacion)

                colu1, colu2 = st.columns(2)

                with colu1:
                    for i in range (3): st.write('')
                    st.subheader('Numero de arbol en el bosque.')
                    no_estimador = st.slider('Numero de estimador en el bosque', min_value = 0,max_value = no_estimadores, step = 1)
                    

                    
                    st.header('Score')
                    presicion = clasificacion.getScore(X_validation, Y_validation)
                    st.subheader('Precision final del modelo: ' + str(presicion * 100))
                    
                    for i in range (3): st.write('')
                    st.subheader('Matriz de clasificasion')
                    matriz_clasificasion, Y_Clasificacion = clasificacion.getMatrizClasificacion(X_validation, Y_validation)
                    st.write(matriz_clasificasion)

                    for i in range (3): st.write('')
                    st.subheader('Variables ordenadas por importancia')
                    importancia_variables = clasificacion.getImportancia()
                    st.write(importancia_variables)

                    
                with colu2:
                    st.subheader('ARBOL')
                    muestra_bosque = clasificacion.getBosqueClasificacion(no_estimador, valores_pronosticados)
                    st.graphviz_chart(muestra_bosque)
                
                st.subheader('Generar nuevos pronosticos.')
                
                columnas1, columnas2 = st.columns(2)
                with columnas1:
                    valores_clasificasion = {}
                    for variable in clasificacion.variables_finales:
                        number = st.number_input(variable + variable, min_value=0, max_value=10000000000, value=1, step=1, format='%d')
                        valores_clasificasion[variable] = [number]
                        #st.write('The current number is ', number)
                with columnas2:
                    
                    pronostico = clasificacion.genPronosticosBosque(valores_clasificasion)
                    if (pronostico == 0):
                        st.write(primer_clase)
                        st.subheader('Pronostico: ' + str(primer_clase))
                    else:
                        st.subheader('Pronostico: ' + str(segunda_clase))

                #st.write('Reporte')
                #reporte = pronostico.getReporte()
                #st.write(reporte)

            



