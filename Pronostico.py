from numpy import empty
import pandas as pd
import seaborn as sns

import numpy as np
import matplotlib.pyplot as plt

from Algoritmo import Algoritmo
from Distancias import Distancias

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import plot_tree
from sklearn.tree import export_text

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import tree

class Pronostico(Algoritmo):

    score = 0.0;
    #arbolDecision =  DecisionTreeRegressor()
    distancias = Distancias()
    variables_iniciales = []
    variables_finales = []

    def crearArbol(self, profundidad, divisiones, hojas, X_train, Y_train):
        self.arbolDecision = DecisionTreeRegressor(max_depth=profundidad, min_samples_split=divisiones, min_samples_leaf=hojas, random_state=0)
        self.arbolDecision.fit(X_train, Y_train)
    
    def crearBosque(self, no_estimadores, no_variables, profundidad, divisiones, hojas, X_train, Y_train):
        self.bosqueAleatorio = RandomForestRegressor(n_estimators=no_estimadores, max_depth=profundidad, min_samples_split=divisiones, 
        min_samples_leaf=hojas, random_state=0, max_features=no_variables)
        self.bosqueAleatorio.fit(X_train, Y_train)
    
    def generarPronosticoArbol(self, X_test):
        Y_Pronostico = self.arbolDecision.predict(X_test)
        return pd.DataFrame(Y_Pronostico)
    
    def generarPronosticoBosque(self, X_test):
        Y_Pronostico = self.bosqueAleatorio.predict(X_test)
        return pd.DataFrame(Y_Pronostico)
    
    def generaGraficaComparacion(self, var_x, var_y, Y_test, Y_Pronostico):
        comparacion = plt.figure(figsize=(20, 5))
        plt.plot(Y_test, color='green', marker='o', label='Valores reales')
        plt.plot(Y_Pronostico, color='red', marker='o', label='Predicciones')
        #plt.xlabel('Paciente')
        #plt.ylabel('Tamaño del tumor')
        plt.title('Pacientes con tumores cancerígenos')
        plt.grid(True)
        plt.legend()
        #plt.show()
        return(comparacion)
    
    def genScore(self, Y_test, Y_Pronostico):
        self.score = r2_score(Y_test, Y_Pronostico)
    
    def getMetricas(self, Y_test, Y_Pronostico):
        mae = mean_absolute_error(Y_test, Y_Pronostico)
        mse = mean_squared_error(Y_test, Y_Pronostico)
        rmse = mean_squared_error(Y_test, Y_Pronostico, squared=False)
        return mae, mse, rmse   
    
    def getImportancia(self):
        Importancia = pd.DataFrame({'Variable': list(self.datos[self.variables_finales]),
                            'Importancia': self.arbolDecision.feature_importances_}).sort_values('Importancia', ascending=False)
        return Importancia
    
    def genArbol(self):
        #arbol = plt.figure(figsize=(10, 10))  
        #arbol = plot_tree(self.arbolDecision, feature_names = self.variables_finales)
        arbol  = tree.export_graphviz(self.arbolDecision, out_file=None)
        #plt.show()
        return arbol
    
    def genBosque(self, no_estimador):
        estimador = self.bosqueAleatorio.estimators_[no_estimador]
        muestra = plt.figure(figsize=(16,16))  
        muestra = tree.export_graphviz(estimador, out_file=None)
        return muestra
    
    def getReporte(self):
        Reporte = export_text(self.arbolDecision, feature_names = self.variables_finales)
        return Reporte
    
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

    def genPronosticosArbol(self, variables):
        registro = pd.DataFrame(variables)
        return self.arbolDecision.predict(registro)
    
    def genPronosticosBosque(self, variables):
        registro = pd.DataFrame(variables)
        return self.bosqueAleatorio.predict(registro)