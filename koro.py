import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
from apyori import apriori

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
    uploaded_file = st.file_uploader("Choose a file")
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
                    st.subheader('Resumen')
                    st.markdown('Se han generado ' + str(num_reglas) + ' reglas')
                    st.markdown('Ejemplo con la ultima regla.')
                    st.markdown('En la ultima regla se relacionan los items: (' + str(aux[0]) + ') con (' + str(aux[1]) + ') teniendo un porcentaje de importancia dentro del conjunto de datos (Soporte) de ' + str(round(aux[2] * 100, 2)) + '%, un porcentaje de fiabilidad (Confianza) de ' + str(round(aux[3] * 100, 2)) + '% y por ultimo la regla representa un aumento en las posibilidades en '+str(round(aux[4], 2))+' veces (Elevacion) de que las personas que esten interesadas en (' + str(aux[0]) + ') tambien lo esten en (' +  str(aux[1]) + ').')
                
            
            



        


if selected == "Distancias":
    st.title(f" {selected}")

if selected == "Clustering":
    st.title(f" {selected}")
