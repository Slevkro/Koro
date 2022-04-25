from pandas import DataFrame
from scipy.spatial.distance import cdist    # Para el cálculo de distancias
from scipy.spatial import distance
import seaborn as sns

from Algoritmo import Algoritmo

class Distancias(Algoritmo):
    metrica = 'ninguna'
    lambda_dist = 1.5
    datos = DataFrame()

    def CalcularDistancia(self, metrica):
        self.metrica = metrica
        #Hay que hacerlos dataframes
        if self.metrica == 'Euclideana':
            return cdist(self.datos, self.datos, metric='euclidean')
        elif self.metrica == 'Chebyshev': 
            return cdist(self.datos, self.datos, metric='chebyshev')
        elif self.metrica == 'Manhattan': 
            return cdist(self.datos, self.datos, metric='cityblock')
        elif self.metrica == 'Minkowski': 
            return cdist(self.datos, self.datos, metric='minkowski', p=self.lambda_dist)
    
    def CalcularDistanciaEstandarizada(self, metrica, inicio, fin):
        self.Estandarizar()
        self.metrica = metrica
        #Hay que hacerlos dataframes
        if self.metrica == 'Euclideana':
            return cdist(self.estandarizada[inicio:fin+1], self.estandarizada[inicio:fin+1], metric='euclidean')
        elif self.metrica == 'Chebyshev': 
            return cdist(self.estandarizada[inicio:fin+1], self.estandarizada[inicio:fin+1], metric='chebyshev')
        elif self.metrica == 'Manhattan': 
            return cdist(self.estandarizada[inicio:fin+1], self.estandarizada[inicio:fin+1], metric='cityblock')
        elif self.metrica == 'Minkowski': 
            return cdist(self.estandarizada[inicio:fin+1], self.estandarizada[inicio:fin+1], metric='minkowski', p=self.lambda_dist)
    
    def CalcularDistanciaNormalizada(self, metrica, inicio, fin):
        self.Normalizar()
        self.metrica = metrica
        #Hay que hacerlos dataframes
        if self.metrica == 'Euclideana':
            return cdist(self.normalizada[inicio:fin+1], self.normalizada[inicio:fin+1], metric='euclidean')
        elif self.metrica == 'Chebyshev': 
            return cdist(self.normalizada[inicio:fin+1], self.normalizada[inicio:fin+1], metric='chebyshev')
        elif self.metrica == 'Manhattan': 
            return cdist(self.normalizada[inicio:fin+1], self.normalizada[inicio:fin+1], metric='cityblock')
        elif self.metrica == 'Minkowski': 
            return cdist(self.normalizada[inicio:fin+1], self.normalizada[inicio:fin+1], metric='minkowski', p=self.lambda_dist)
    
    #def getMapaDeCalor(self, distancias):

    def Correlaciones(self):
        self.matrizCorrelaciones = self.datos.corr(method='pearson')

    def CiertasDistancias(self, inicio, fin, metrica):
        if self.metrica == 'Euclideana':
            return cdist(self.datos[inicio:fin+1], self.datos[inicio:fin+1], metric='euclidean')
        elif self.metrica == 'Chebyshev': 
            return cdist(self.datos[inicio:fin+1], self.datos[inicio:fin+1], metric='chebyshev')
        elif self.metrica == 'Manhattan': 
            return cdist(self.datos[inicio:fin+1], self.datos[inicio:fin+1], metric='cityblock')
        elif self.metrica == 'Minkowski': 
            return cdist(self.datos[inicio:fin+1], self.datos[inicio:fin+1], metric='minkowski', p=self.lambda_dist)

    