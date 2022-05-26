from numpy import empty
import pandas as pd
import seaborn as sns

import numpy as np
import matplotlib.pyplot as plt

from Algoritmo import Algoritmo
from Distancias import Distancias

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import model_selection
from sklearn.tree import plot_tree
from sklearn.tree import export_text

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import model_selection

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class Clasificacion(Algoritmo):

    score = 0.0;
    #arbolDecision =  DecisionTreeRegressor()
    distancias = Distancias()
    variables_iniciales = []
    variables_finales = []

    def getClasesSize(self, columna):
        return self.datos.groupby(columna).size()
    
    def genArbolClasificacion(self, profundidad, division, hojas, X_train, Y_train):
        self.arbolClasificacion = DecisionTreeClassifier(max_depth=profundidad, min_samples_split=division, min_samples_leaf=hojas, random_state=0)
        self.arbolClasificacion.fit(X_train, Y_train)

    def genBosqueClasificasion(self, estimadores, no_variables,  profundidad, division, hojas, X_train, Y_train):
        self.bosqueClasificacion = RandomForestClassifier(n_estimators=50, max_depth=profundidad, min_samples_split=division, 
        min_samples_leaf=hojas, random_state=0, max_features=no_variables)
        self.bosqueClasificacion.fit(X_train, Y_train)
    
    def getClasificacionArbol(self, X_validation):
        Y_Clasificacion = self.arbolClasificacion.predict(X_validation)
        return pd.DataFrame(Y_Clasificacion)
    
    def getScore(self, X_validation, Y_validation):
        return self.arbolClasificacion.score(X_validation, Y_validation)
    
    def getMatrizClasificacion(self, X_validation, Y_validation):
        Y_Clasificacion = self.arbolClasificacion.predict(X_validation)
        Matriz_Clasificacion = pd.crosstab(Y_validation.ravel(), 
                                        Y_Clasificacion, 
                                        rownames=['Real'], 
                                        colnames=['Clasificación']) 
        return Matriz_Clasificacion, Y_Clasificacion
    
    def getImportancia(self):
        Importancia = pd.DataFrame({'Variable': list(self.datos[self.variables_finales]),
                            'Importancia': self.arbolClasificacion.feature_importances_}).sort_values('Importancia', ascending=False)  
        return Importancia
    
    def getArbolClasificacion(self, Y_Clasificacion):
        
        arbol = plt.figure(figsize=(16,16))  
        plot_tree(self.arbolClasificacion, 
                feature_names = self.variables_finales,
                class_names = Y_Clasificacion)
        plt.show()
        return arbol

    def getBosqueClasificacion(self, no_estimador, Y_Clasificacion):
        Estimador = self.bosqueClasificacion.estimators_[no_estimador]
        bosque = plt.figure(figsize=(16,16))  
        plot_tree(Estimador, 
                feature_names = self.variables_finales,
                class_names = Y_Clasificacion)
        plt.show()
        return bosque
    
    def getReporte(self):
        Reporte = export_text(self.arbolClasificacion, 
                            feature_names = self.variables_finales)
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
    
    def valoresPronosticados(self, X_validation):
        Y_Clasificacion = self.bosqueClasificacion.predict(X_validation)
        return Y_Clasificacion
