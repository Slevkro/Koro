from typing import Type
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
from apyori import apriori
from scipy.spatial.distance import cdist
import seaborn as sns


from Distancias import Distancias
from Cluster import Cluster

# Configuraciones iniciales

st.set_page_config(layout="wide")

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
        options=["KORO", "...", "...", "Asociacion", "Distancias", "Clustering"],
        icons=["option", "", "", "link", "people", "palette2"],
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
        st.title(f"ANALISIS CORRELACIONAL")
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
            #st.write(rodilla_)
        
        num_clusters_par_final, centroides_particional = colador.ClusterParticional(matriz_variables_finales, num_clusters_par)
        columna1, columna2 = st.columns(2)

        with columna1:
            st.subheader('Conteo de elementos por cluster.')
            st.write(num_clusters_par_final)
        with columna2:
            st.subheader('Centroides del conjunto de datos.')
            st.write(centroides_particional)
        

