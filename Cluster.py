from numpy import empty
from Algoritmo import Algoritmo
from Distancias import Distancias

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from kneed import KneeLocator

class Cluster(Algoritmo):
    distancias = Distancias()
    variables_iniciales = []
    variables_finales = []
    SSE = []

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
    
    def getClusterJerarquico(self, metrica, matriz):
        arbol = plt.figure(figsize=(10, 7))
        plt.title(self.archivo.name)
        plt.xlabel('Elementos')
        plt.ylabel('Capas del árbol')
        Arbol = shc.dendrogram(shc.linkage(matriz, method='complete', metric=metrica))
        return arbol
    
    def getCentroidesJerarquico(self, matriz, num_clusters, metrica):
        MJerarquico = AgglomerativeClustering(n_clusters=num_clusters, linkage='complete', affinity=metrica)
        MJerarquico.fit_predict(matriz)
        centroides_jerarquico = self.datos
        centroides_jerarquico['Cluster Jerarquico'] = MJerarquico.labels_
        elementos_por_cluster = centroides_jerarquico.groupby(['Cluster Jerarquico'])['Cluster Jerarquico'].count() 
        centroides_por_cluster = centroides_jerarquico.groupby(['Cluster Jerarquico'])[self.variables_finales].mean()
        return elementos_por_cluster, centroides_por_cluster
    
    def getRodilla(self, matriz):
        #SSE = []
        self.SSE = []
        for i in range(2, 12):
            km = KMeans(n_clusters=i, random_state=0)
            km.fit(matriz)
            self.SSE.append(km.inertia_)

        #Se grafica SSE en función de k
        rodilla = plt.figure(figsize=(10, 7))
        plt.plot(range(2, 12), self.SSE, marker='o')
        plt.xlabel('Cantidad de clusters *k*')
        plt.ylabel('SSE')
        plt.title('Elbow Method')
        return rodilla
    
    def getRodillaExacta(self):
        kl = KneeLocator(range(2, 12), self.SSE, curve="convex", direction="decreasing")
        rodilla_exacta = plt.figure(figsize=(10, 7))
        kl.elbow
        #plt.style.use('ggplot')
        plt.xlabel('Cantidad de clusters *k*')
        plt.ylabel('SSE')
        plt.title('Elbow Method')
        rodilla_exacta = kl.plot_knee()
        return rodilla_exacta

    def ClusterParticional(self, matriz, num_clusters):
        MParticional = KMeans(n_clusters=num_clusters, random_state=0).fit(matriz)
        MParticional.predict(matriz)

        centroides_particional = self.datos
        centroides_particional['Cluster Particional'] = MParticional.labels_

        elementos_por_cluster = centroides_particional.groupby(['Cluster Particional'])['Cluster Particional'].count()
        centroides_por_cluster = centroides_particional.groupby(['Cluster Particional'])[self.variables_finales].mean()

        return elementos_por_cluster, centroides_por_cluster