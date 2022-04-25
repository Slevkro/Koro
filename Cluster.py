from numpy import empty
from Algoritmo import Algoritmo
from Distancias import Distancias

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Cluster(Algoritmo):
    distancias = Distancias()
    variables_iniciales = []
    variables_finales = []

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
        