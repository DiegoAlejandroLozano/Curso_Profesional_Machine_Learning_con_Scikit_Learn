import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if __name__ == '__main__': #Si este es el script principal que vamos a ejecutar
    df_heart = pd.read_csv('..\Datasets\heart.csv')
    print(df_heart)

    #Vamos a divir nuestros datos en features y target
    df_features = df_heart.drop(['target'], axis=1)
    df_target = df_heart['target']

    #Para PCA siempre es necesario normalizar nuestros datos
    df_features = StandardScaler().fit_transform(df_features)

    #Vamos a divir los datos en entramiento y prueba
    x_train, x_test, y_train, y_test = train_test_split(df_features, df_target, test_size=0.3, random_state=42)

    #Creamos nuestro PCA
    pca = PCA(n_components=3)
    pca.fit(x_train)

    #Como se va a realizar una comparación con IncrementalIPC, se realiza lo mismo
    ipca = IncrementalPCA(n_components=3, batch_size=10)
    ipca.fit(x_train)

    plt.plot(
        range(len(pca.explained_variance_)),
        pca.explained_variance_ratio_
    )   
    plt.show()

    '''Configurando la regresión logística para realizar la comparación de los dos
    algorítmos de reducción de dimensionalidad: PCA y IPCA'''

    logistic = LogisticRegression(solver='lbfgs')

    df_train = pca.transform(x_train)
    df_test = pca.transform(x_test)
    logistic.fit(df_train, y_train)
    print("SCORE PCA: ", logistic.score(df_test, y_test))

    df_train = ipca.transform(x_train)
    df_test = ipca.transform(x_test)

    logistic.fit(df_train, y_train)
    print("SCORE IPCA: ", logistic.score(df_test, y_test))
