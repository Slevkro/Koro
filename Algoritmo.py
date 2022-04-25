import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler, MinMaxScaler 

class Algoritmo:
    estandarizar = StandardScaler()                               
    normalizar = MinMaxScaler()
    #datos = pd.DataFrame()
    # archivo
    # datos
    # estandarizada
    # normalizada
    
    def LeerDatos(self, id): 
        #self.archivo = nombre_archivos
        self.archivo = st.file_uploader("Choose a file", key = id)
        if self.archivo is not None:
            self.datos = pd.read_csv(self.archivo)
            self.MostrarDatos()
            

    def MostrarDatos(self):
        st.write(self.datos)
    
    def Estandarizar(self):
        self.estandarizada = self.estandarizar.fit_transform(self.datos)

    def EstandarizarMatriz(self, matriz):
        return self.estandarizar.fit_transform(matriz)   

    def Normalizar(self):
        self.normalizada = self.normalizar.fit_transform(self.datos)
    
    def NormalizarMatriz(self, matriz):
        return self.normalizar.fit_transform(matriz)
