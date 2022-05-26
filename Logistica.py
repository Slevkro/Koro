from sklearn import linear_model
from sklearn.metrics import classification_report
import streamlit as st

from numpy import empty
import pandas as pd
import seaborn as sns

import numpy as np
import matplotlib.pyplot as plt

from Algoritmo import Algoritmo
from Distancias import Distancias

class Logistica(Algoritmo):

    score = 0.0;
    Clasificacion = Clasificacion = linear_model.LogisticRegression()
    distancias = Distancias()
    variables_iniciales = []
    variables_finales = []

    def crearModelo(self, X_train, Y_train):
        self.Clasificacion.fit(X_train, Y_train)
    
    def getMatrizProbabilidades(self, X_validation):
        Probabilidad = self.Clasificacion.predict_proba(X_validation)
        return pd.DataFrame(Probabilidad)
    
    def getScore(self, X_validation, Y_validation):
        return self.Clasificacion.score(X_validation, Y_validation)
    
    def getMatrizClasificacion(self, Y_validation, X_validation):
        Y_Clasificacion = self.Clasificacion.predict(X_validation)
        Matriz_Clasificacion = pd.crosstab(Y_validation.ravel(), 
                                        Y_Clasificacion, 
                                        rownames=['Real'], 
                                        colnames=['Clasificación']) 
        return Y_Clasificacion, Matriz_Clasificacion
    
    def getReporte(self, X_validation, Y_validation, Y_Clasificacion):
        st.write("Exactitud", self.Clasificacion.score(X_validation, Y_validation))
        st.write(classification_report(Y_validation, Y_Clasificacion))

    def getEcuacion(self):
        intercepto = self.Clasificacion.intercept_
        coeficientes =  self.Clasificacion.coef_
        return intercepto, coeficientes

    def getMapaDeCalor(self):
        self.distancias.datos = self.datos
        self.distancias.Correlaciones()

        mapa_de_calor =plt.figure(figsize=(14,7))
        MatrizInf = np.triu(self.distancias.matrizCorrelaciones)
        sns.heatmap(self.distancias.matrizCorrelaciones, cmap='RdBu_r', annot=True, mask=MatrizInf)

        return mapa_de_calor
    
    def getVariablesFinales(self):
        if self.variables_finales is not empty:
            MatrizVariables = np.array(self.datos[self.variables_finales])
            MatrizVariables = pd.DataFrame(MatrizVariables)
            MatrizVariables.columns = self.variables_finales
            return MatrizVariables
