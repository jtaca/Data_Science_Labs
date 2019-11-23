import sys, pandas as pd
from preprocessing_pd import preprocessing_pd_report
from preprocessing_ct import preprocessing_ct_report

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE, RandomOverSampler

from sklearn.metrics import silhouette_score, adjusted_rand_score, confusion_matrix, multilabel_confusion_matrix
from sklearn.utils.multiclass import unique_labels
#import itertools


from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn.ensemble import RandomForestClassifier





def unsupervised_pd_report(data):
    
    data = data.sort_values('id', ascending=True)

    #Fazer a média das 3 medições com o mesmo id
    data = data.groupby('id').mean().reset_index()
    
    # Normalization
    transf = MinMaxScaler().fit(data)
    data = pd.DataFrame(transf.transform(data), columns= data.columns)
    
    
    X = data.drop(columns=['class', 'id'])
    y = data['class'].values

    from sklearn.decomposition import PCA
    X = PCA(n_components=2, random_state=1).fit_transform(X)

    from sklearn.feature_selection import SelectKBest, chi2, f_classif
    #X = SelectKBest(f_classif, k=2).fit_transform(X, y)

    plt.subplot(111)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()
    n_of_cluster = []
    inertia_values = []
    silhouette_values = []
    rand_index_values = []

    for i in range(2, 10):
        kmeans_model = cluster.KMeans(n_clusters=i, random_state=1).fit(X)
        y_pred = kmeans_model.labels_
        n_of_cluster.append(i) 
        inertia_values.append(kmeans_model.inertia_)
        silhouette_values.append(silhouette_score(X, y_pred))
        rand_index_values.append(adjusted_rand_score(y, y_pred))

        # Inertia, diferença entre valores dentro do mesmo cluster, quão coerentes os clusters são internamente, o objetivo é minimizar este valor
        plt.plot(n_of_cluster, inertia_values)
        plt.xlabel("Nr of clusters")
        plt.ylabel("Inertia")
        plt.show()

        # Silhouette, quão bem classificado está um sample no seu cluster, quanto mais alto for o valor melhor definido está o cluster
        plt.plot(n_of_cluster, silhouette_values)
        plt.xlabel("Nr of clusters")
        plt.ylabel("Silhoutte score")
        plt.show()

        # Ajusted Rand Indexes, é a similariedade entre dois clusters, usamos para comparar o predicted com o verdadeiro
        # requer saber as classes verdadeiras, 
        # o valor 1.0 que é o mais alto é o perfeito significa classificação de labels perfeita
        # distribuição uniforme (random) de labels tem um valor de 0.0
        plt.plot(n_of_cluster, rand_index_values)
        plt.xlabel("Nr of clusters")
        plt.ylabel("Rand index score")
        plt.show()

        plt.plot(n_of_cluster, silhouette_values, label="Silhoutte score")
        plt.plot(n_of_cluster, rand_index_values, label="Rand Index score")
        plt.legend(loc='upper right')
        plt.xlabel("Nr of clusters")
        plt.show()
    
    algorithms = {}
    n_clusters = 2

    # 1b Parameterize clustering algorithms
    algorithms['K Means'] = cluster.KMeans(n_clusters=n_clusters, random_state=1)
    algorithms['Ward Linkage'] = cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    algorithms['Average Linkage'] = cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage='average')

    # 3 Run clustering algorithm and store predictions
    predictions = {}
    efficiency = {}
    for idx, name in enumerate(algorithms):
        clustering = algorithms[name]
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="the number of connected components of the connectivity matrix is [0-9]{1,2}" +
                    " > 1. Completing it to avoid stopping the tree early.",
                category=UserWarning)
            clustering.fit(X)
        efficiency[name]= time.time()-t0
        if hasattr(clustering, 'labels_'): predictions[name] = clustering.labels_.astype(np.int)
        else: predictions[name] = clustering.predict(X)


        plt.figure(figsize=(11, 8))
        plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)
        color_array = ['#377eb8','#ff7f00','#4daf4a','#f781bf','#a65628','#984ea3','#999999','#e41a1c','#dede00']
        plot_num = 1

        for idx, name in enumerate(predictions):
            y_predi = predictions[name]
            plt.subplot(2, 3, plot_num)
            plt.tight_layout()
            plt.title(name+" - "+str(n_clusters)+" clusters", size=15)
            plt.xticks([])
            plt.yticks([])
            colors = np.array(list(islice(cycle(color_array),int(max(y_predi) + 1))))
            colors = np.append(colors, ["#000000"]) #black color for outliers (if any)
            plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_predi])
            silh = str('%.3f'%(silhouette_score(X, y_predi)))
            ari = str('%.3f'%(adjusted_rand_score(y, y_predi)))
            plt.text(.99, .01, 'ARI '+ari+', Silhouette '+silh, transform=plt.gca().transAxes,size=10,horizontalalignment='right')
            plot_num += 1

        plt.subplot(2, 3, plot_num)
        plt.title("True", size=18)
        plt.xticks([])
        plt.yticks([])
        plt.scatter(X[:, 0], X[:, 1], s=10, c=y)

        plt.show()
    
    
    algorithms = {}
    n_clusters = 3

    # 1b Parameterize clustering algorithms
    algorithms['K Means'] = cluster.KMeans(n_clusters=n_clusters, random_state=1)
    algorithms['Ward Linkage'] = cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    algorithms['Average Linkage'] = cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage='average')

    # 3 Run clustering algorithm and store predictions
    predictions = {}
    efficiency = {}
    for idx, name in enumerate(algorithms):
        clustering = algorithms[name]
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="the number of connected components of the connectivity matrix is [0-9]{1,2}" +
                    " > 1. Completing it to avoid stopping the tree early.",
                category=UserWarning)
            clustering.fit(X)
        efficiency[name]= time.time()-t0
        if hasattr(clustering, 'labels_'): predictions[name] = clustering.labels_.astype(np.int)
        else: predictions[name] = clustering.predict(X)


        plt.figure(figsize=(11, 8))
        plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)
        color_array = ['#377eb8','#ff7f00','#4daf4a','#f781bf','#a65628','#984ea3','#999999','#e41a1c','#dede00']
        plot_num = 1

        for idx, name in enumerate(predictions):
            y_predi = predictions[name]
            plt.subplot(2, 3, plot_num)
            plt.tight_layout()
            plt.title(name+" - "+str(n_clusters)+" clusters", size=15)
            plt.xticks([])
            plt.yticks([])
            colors = np.array(list(islice(cycle(color_array),int(max(y_predi) + 1))))
            colors = np.append(colors, ["#000000"]) #black color for outliers (if any)
            plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_predi])
            silh = str('%.3f'%(silhouette_score(X, y_predi)))
            ari = str('%.3f'%(adjusted_rand_score(y, y_predi)))
            plt.text(.99, .01, 'ARI '+ari+', Silhouette '+silh, transform=plt.gca().transAxes,size=10,horizontalalignment='right')
            plot_num += 1

        plt.subplot(2, 3, plot_num)
        plt.title("True", size=18)
        plt.xticks([])
        plt.yticks([])
        plt.scatter(X[:, 0], X[:, 1], s=10, c=y)

        plt.show()
    
    algorithms = {}
    n_clusters = 4

    # 1b Parameterize clustering algorithms
    algorithms['K Means'] = cluster.KMeans(n_clusters=n_clusters, random_state=1)
    algorithms['Ward Linkage'] = cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    algorithms['Average Linkage'] = cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage='average')

    # 3 Run clustering algorithm and store predictions
    predictions = {}
    efficiency = {}
    for idx, name in enumerate(algorithms):
        clustering = algorithms[name]
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="the number of connected components of the connectivity matrix is [0-9]{1,2}" +
                    " > 1. Completing it to avoid stopping the tree early.",
                category=UserWarning)
            clustering.fit(X)
        efficiency[name]= time.time()-t0
        if hasattr(clustering, 'labels_'): predictions[name] = clustering.labels_.astype(np.int)
        else: predictions[name] = clustering.predict(X)


        plt.figure(figsize=(11, 8))
        plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)
        color_array = ['#377eb8','#ff7f00','#4daf4a','#f781bf','#a65628','#984ea3','#999999','#e41a1c','#dede00']
        plot_num = 1

        for idx, name in enumerate(predictions):
            y_predi = predictions[name]
            plt.subplot(2, 3, plot_num)
            plt.tight_layout()
            plt.title(name+" - "+str(n_clusters)+" clusters", size=15)
            plt.xticks([])
            plt.yticks([])
            colors = np.array(list(islice(cycle(color_array),int(max(y_predi) + 1))))
            colors = np.append(colors, ["#000000"]) #black color for outliers (if any)
            plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_predi])
            silh = str('%.3f'%(silhouette_score(X, y_predi)))
            ari = str('%.3f'%(adjusted_rand_score(y, y_predi)))
            plt.text(.99, .01, 'ARI '+ari+', Silhouette '+silh, transform=plt.gca().transAxes,size=10,horizontalalignment='right')
            plot_num += 1

        plt.subplot(2, 3, plot_num)
        plt.title("True", size=18)
        plt.xticks([])
        plt.yticks([])
        plt.scatter(X[:, 0], X[:, 1], s=10, c=y)

        plt.show()



def unsupervised_ct_report(data):
    X = data.drop(columns=['Cover_Type'])
    y = data['Cover_Type'].values
    print(X.shape)

    # não há missing values
    from sklearn.impute import SimpleImputer

    try:
        imp_nr = SimpleImputer(strategy='mean', missing_values=np.nan, copy=True)
        imp_sb = SimpleImputer(strategy='most_frequent', missing_values='', copy=True)
        df_nr = pd.DataFrame(imp_nr.fit_transform(cols_nr), columns=cols_nr.columns)
        df_sb = pd.DataFrame(imp_sb.fit_transform(cols_sb), columns=cols_sb.columns)
    except:
        print("No missing values") 

    #Não se faz dummify, pois não existem features do tipo category 
    cols_sb = data.select_dtypes(include='category')
    if cols_sb.empty:
        print('No category type columns!') #shows no categoriy types in the dataset
    else:
        print('Exists category type columns!')

    # Hold-Out 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.995, random_state=13)
    print(X_train.shape)

    # normalize
    from sklearn.preprocessing import Normalizer

    transf = StandardScaler().fit(X_train)
    data = pd.DataFrame(transf.transform(X_train))
    data.describe(include='all')
    
    X = data
    y = y_train

    #from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=1).fit(X)
    X = pca.transform(X)

    plt.subplot(111)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()
    
    n_of_cluster = []
    inertia_values = []
    silhouette_values = []
    rand_index_values = []

    for i in range(2, 12):
        kmeans_model = cluster.KMeans(n_clusters=i, random_state=1).fit(X)
        y_pred = kmeans_model.labels_
        n_of_cluster.append(i) 
        inertia_values.append(kmeans_model.inertia_)
        silhouette_values.append(silhouette_score(X, y_pred))
        rand_index_values.append(adjusted_rand_score(y, y_pred))

    # Inertia, diferença entre valores dentro do mesmo cluster, quão coerentes os clusters são internamente, o objetivo é minimizar este valor
    plt.plot(n_of_cluster, inertia_values)
    plt.xlabel("Nr of clusters")
    plt.ylabel("Inertia")
    plt.show()

    # Silhouette, quão bem classificado está um sample no seu cluster, quanto mais alto for o valor melhor definido está o cluster
    plt.plot(n_of_cluster, silhouette_values)
    plt.xlabel("Nr of clusters")
    plt.ylabel("Silhoutte score")
    plt.show()

    # Ajusted Rand Indexes, é a similariedade entre dois clusters, usamos para comparar o predicted com o verdadeiro
    # requer saber as classes verdadeiras, 
    # o valor 1.0 que é o mais alto é o perfeito significa classificação de labels perfeita
    # distribuição uniforme (random) de labels tem um valor de 0.0
    plt.plot(n_of_cluster, rand_index_values)
    plt.xlabel("Nr of clusters")
    plt.ylabel("Rand index score")
    plt.show()

    plt.plot(n_of_cluster, silhouette_values, label="Silhoutte score")
    plt.plot(n_of_cluster, rand_index_values, label="Rand Index score")
    plt.legend(loc='upper right')
    plt.xlabel("Nr of clusters")
    plt.show()
    
    algorithms = {}
    n_clusters = 2

    # 1b Parameterize clustering algorithms
    algorithms['K Means'] = cluster.KMeans(n_clusters=n_clusters, random_state=1)
    algorithms['Ward Linkage'] = cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    algorithms['Average Linkage'] = cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage='average')

    # 3 Run clustering algorithm and store predictions
    predictions = {}
    efficiency = {}
    for idx, name in enumerate(algorithms):
        clustering = algorithms[name]
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="the number of connected components of the connectivity matrix is [0-9]{1,2}" +
                    " > 1. Completing it to avoid stopping the tree early.",
                category=UserWarning)
            clustering.fit(X)
        efficiency[name]= time.time()-t0
        if hasattr(clustering, 'labels_'): predictions[name] = clustering.labels_.astype(np.int)
        else: predictions[name] = clustering.predict(X)

    plt.figure(figsize=(11, 8))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)
    color_array = ['#377eb8','#ff7f00','#4daf4a','#f781bf','#a65628','#984ea3','#999999','#e41a1c','#dede00']
    plot_num = 1

    for idx, name in enumerate(predictions):
        y_predi = predictions[name]
        plt.subplot(2, 3, plot_num)
        plt.tight_layout()
        plt.title(name, size=18)
        plt.xticks([])
        plt.yticks([])
        colors = np.array(list(islice(cycle(color_array),int(max(y_predi) + 1))))
        colors = np.append(colors, ["#000000"]) #black color for outliers (if any)
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_predi])
        silh = str('%.3f'%(silhouette_score(X, y_predi)))
        ari = str('%.3f'%(adjusted_rand_score(y, y_predi)))
        plt.text(.99, .01, 'ARI '+ari+', Silhouette '+silh, transform=plt.gca().transAxes,size=10,horizontalalignment='right')
        plot_num += 1

    plt.subplot(2, 3, plot_num)
    plt.title("True", size=18)
    plt.xticks([])
    plt.yticks([])
    plt.scatter(X[:, 0], X[:, 1], s=10, c=y)

    plt.show()
    
    algorithms = {}
    n_clusters = 3

    # 1b Parameterize clustering algorithms
    algorithms['K Means'] = cluster.KMeans(n_clusters=n_clusters, random_state=1)
    algorithms['Ward Linkage'] = cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    algorithms['Average Linkage'] = cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage='average')

    # 3 Run clustering algorithm and store predictions
    predictions = {}
    efficiency = {}
    for idx, name in enumerate(algorithms):
        clustering = algorithms[name]
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="the number of connected components of the connectivity matrix is [0-9]{1,2}" +
                    " > 1. Completing it to avoid stopping the tree early.",
                category=UserWarning)
            clustering.fit(X)
        efficiency[name]= time.time()-t0
        if hasattr(clustering, 'labels_'): predictions[name] = clustering.labels_.astype(np.int)
        else: predictions[name] = clustering.predict(X)


        plt.figure(figsize=(11, 8))
        plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)
        color_array = ['#377eb8','#ff7f00','#4daf4a','#f781bf','#a65628','#984ea3','#999999','#e41a1c','#dede00']
        plot_num = 1

        for idx, name in enumerate(predictions):
            y_predi = predictions[name]
            plt.subplot(2, 3, plot_num)
            plt.tight_layout()
            plt.title(name, size=18)
            plt.xticks([])
            plt.yticks([])
            colors = np.array(list(islice(cycle(color_array),int(max(y_predi) + 1))))
            colors = np.append(colors, ["#000000"]) #black color for outliers (if any)
            plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_predi])
            silh = str('%.3f'%(silhouette_score(X, y_predi)))
            ari = str('%.3f'%(adjusted_rand_score(y, y_predi)))
            plt.text(.99, .01, 'ARI '+ari+', Silhouette '+silh, transform=plt.gca().transAxes,size=10,horizontalalignment='right')
            plot_num += 1

        plt.subplot(2, 3, plot_num)
        plt.title("True", size=18)
        plt.xticks([])
        plt.yticks([])
        plt.scatter(X[:, 0], X[:, 1], s=10, c=y)

        plt.show()
    
    algorithms = {}
    n_clusters = 4

    # 1b Parameterize clustering algorithms
    algorithms['K Means'] = cluster.KMeans(n_clusters=n_clusters, random_state=1)
    algorithms['Ward Linkage'] = cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    algorithms['Average Linkage'] = cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage='average')

    # 3 Run clustering algorithm and store predictions
    predictions = {}
    efficiency = {}
    for idx, name in enumerate(algorithms):
        clustering = algorithms[name]
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="the number of connected components of the connectivity matrix is [0-9]{1,2}" +
                    " > 1. Completing it to avoid stopping the tree early.",
                category=UserWarning)
            clustering.fit(X)
        efficiency[name]= time.time()-t0
        if hasattr(clustering, 'labels_'): predictions[name] = clustering.labels_.astype(np.int)
        else: predictions[name] = clustering.predict(X)


    plt.figure(figsize=(11, 8))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)
    color_array = ['#377eb8','#ff7f00','#4daf4a','#f781bf','#a65628','#984ea3','#999999','#e41a1c','#dede00']
    plot_num = 1

    for idx, name in enumerate(predictions):
        y_predi = predictions[name]
        plt.subplot(2, 3, plot_num)
        plt.tight_layout()
        plt.title(name, size=18)
        plt.xticks([])
        plt.yticks([])
        colors = np.array(list(islice(cycle(color_array),int(max(y_predi) + 1))))
        colors = np.append(colors, ["#000000"]) #black color for outliers (if any)
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_predi])
        silh = str('%.3f'%(silhouette_score(X, y_predi)))
        ari = str('%.3f'%(adjusted_rand_score(y, y_predi)))
        plt.text(.99, .01, 'ARI '+ari+', Silhouette '+silh, transform=plt.gca().transAxes,size=10,horizontalalignment='right')
        plot_num += 1

    plt.subplot(2, 3, plot_num)
    plt.title("True", size=18)
    plt.xticks([])
    plt.yticks([])
    plt.scatter(X[:, 0], X[:, 1], s=10, c=y)

    plt.show()

    algorithms = {}
    n_clusters = 5

    # 1b Parameterize clustering algorithms
    algorithms['K Means'] = cluster.KMeans(n_clusters=n_clusters, random_state=1)
    algorithms['Ward Linkage'] = cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    algorithms['Average Linkage'] = cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage='average')

    # 3 Run clustering algorithm and store predictions
    predictions = {}
    efficiency = {}
    for idx, name in enumerate(algorithms):
        clustering = algorithms[name]
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="the number of connected components of the connectivity matrix is [0-9]{1,2}" +
                    " > 1. Completing it to avoid stopping the tree early.",
                category=UserWarning)
            clustering.fit(X)
        efficiency[name]= time.time()-t0
        if hasattr(clustering, 'labels_'): predictions[name] = clustering.labels_.astype(np.int)
        else: predictions[name] = clustering.predict(X)


    plt.figure(figsize=(11, 8))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)
    color_array = ['#377eb8','#ff7f00','#4daf4a','#f781bf','#a65628','#984ea3','#999999','#e41a1c','#dede00']
    plot_num = 1

    for idx, name in enumerate(predictions):
        y_predi = predictions[name]
        plt.subplot(2, 3, plot_num)
        plt.tight_layout()
        plt.title(name, size=18)
        plt.xticks([])
        plt.yticks([])
        colors = np.array(list(islice(cycle(color_array),int(max(y_predi) + 1))))
        colors = np.append(colors, ["#000000"]) #black color for outliers (if any)
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_predi])
        silh = str('%.3f'%(silhouette_score(X, y_predi)))
        ari = str('%.3f'%(adjusted_rand_score(y, y_predi)))
        plt.text(.99, .01, 'ARI '+ari+', Silhouette '+silh, transform=plt.gca().transAxes,size=10,horizontalalignment='right')
        plot_num += 1

    plt.subplot(2, 3, plot_num)
    plt.title("True", size=18)
    plt.xticks([])
    plt.yticks([])
    plt.scatter(X[:, 0], X[:, 1], s=10, c=y)

    plt.show()
    
    
    

def classification_pd_report(data):
    #using a 10-fold cross validation
    #Fazer a média das 3 medições
    data = data.groupby('id').mean().reset_index()

    y = data.pop('class').values
    df2 = pd.DataFrame(y, columns=['class'])
    transf = Normalizer().fit(data)
    norm_data = pd.DataFrame(transf.transform(data, copy=True), columns= data.columns)
    norm_data = pd.concat([norm_data, df2], axis=1)
    data = norm_data
    
    

    init_n_rows = str(data.shape[0])
    data = data.sort_values('id', ascending=True)
    data = data.groupby('id').mean().reset_index()
    final_n_rows = str(data.shape[0])

    unbal = data
    
    unbal = data
    target_count = unbal['class'].value_counts()
    
    min_class = target_count.idxmin()
    ind_min_class = target_count.index.get_loc(min_class)
    print(min_class)

    df_class_min = unbal[unbal['class'] == min_class]
    df_class_max = unbal[unbal['class'] != min_class]
    
    RANDOM_STATE = 42
    target_values_0 = target_count.values[ind_min_class]
    target_values_1 = target_count.values[1-ind_min_class]
    values = {'data': [target_values_0, target_values_1]}

    df_under = df_class_max.sample(len(df_class_min))
    values['UnderSample'] = [target_count.values[ind_min_class], len(df_under)]

    df_over = df_class_min.sample(len(df_class_max), replace=True)
    values['OverSample'] = [len(df_over), target_count.values[1-ind_min_class]]

    smote = SMOTE(ratio='minority', random_state=RANDOM_STATE)

    y = unbal.pop('class').values
    X = unbal.values
    smote_x, smote_y = smote.fit_sample(X, y)
    smote_target_count = pd.Series(smote_y).value_counts()

    df_SMOTE = pd.DataFrame(smote_x)
    df_SMOTE.columns = unbal.columns
    df_SMOTE['class'] = smote_y

    values['SMOTE'] = [smote_target_count.values[ind_min_class], smote_target_count.values[1-ind_min_class]]
    
    data = df_SMOTE

    data.to_csv(r'data.csv', index = False)

    #feature selection

    y = data.pop('class').values
    X = data.values
    
    X_norm = MinMaxScaler().fit_transform(X)

    X_norm = pd.DataFrame(X_norm, columns=data.columns)
    kbest = SelectKBest(chi2, k=100)
    X_new = kbest.fit_transform(X_norm, y)

    column_names = data.columns[kbest.get_support()]
    X_new = pd.DataFrame(X_norm, columns=column_names)

    df2 = pd.DataFrame(y, columns=['class'])
    data = pd.concat([X_new, df2], axis=1)

    print("2. Classifiers:\n ")

    labels = pd.unique(y)

   
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None, stratify=y)
    
    cv = KFold(n_splits=10, random_state=42, shuffle=False)
    
    y = data.pop('class').values
    X = data.values

    print("2.1 NB\n")
    print("a) Suggested parameterization: MultinomialNB with SMOTE and KBest feature selection")

    model = MultinomialNB()
    scoresNB = []
    y_predict_total =[]
    y_test_total =[]
    tn, fp, fn, tp = 0,0,0,0
    #print("Initial shape: "+str(X.shape))
    
    X = MinMaxScaler().fit_transform(X)
    X = pd.DataFrame(X)
    
    y_pred = cross_val_predict(model, X, y, cv=10)
    
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    
    acuNB = round(metrics.accuracy_score(y, y_pred),2)
    #acuNB = round((tp+tn)/(tp+tn+fp+fn),2)
    specificityNB = round(tn / (tn+fp),2)
    sensitivityNB = round(tp / (tp + fn),2)
    precisionNB = round(metrics.precision_score(y, y_pred),2)
    f1NB = round(metrics.f1_score(y, y_pred),2)
    
    
    
    #kfold = model_selection.KFold(n_splits=10, random_state=100)
    #model_kfold = MultinomialNB()
    #results_kfold = model_selection.cross_val_score(model_kfold, X, y, cv=kfold)
    #print(results_kfold)
    
    
    #for train_index, test_index in cv.split(X):
        
        #X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        
        
        
        #model.fit(X_train, y_train)

        #y_predict = gaussNB.predict(X_test)

        #y_predict_total.extend(y_predict)
        #y_test_total.extend(y_train)

        #scoresNB.append(model.score(X_test, y_test))
        
    
    
    #print(np.mean(scoresNB))


    #y_predict = model.predict(X_test)

    #acu = accuracy_score(y_test, y_predict)
    
    #print("Accuracy:", round(acu,2))
    #tn, fp, fn, tp = confusion_matrix(y_test, y_predict).ravel()
    
    #print("Specificity: ", round(specificity,2))
    
    #print("Sensitivity: ", round(sensitivity,2))
    
    print("b) Confusion matrix: ")

    conf_mat = confusion_matrix(y, y_pred)
    print(conf_mat)
    
    print("2.2 kNN\n")
    print("a) Suggested parameterization: n_neighbors= 49, metric= euclidean, with SMOTE and KBest feature selection")


    model = KNeighborsClassifier(n_neighbors=49, metric='euclidean')
    scoresNB = []
    y_predict_total =[]
    y_test_total =[]
    tn, fp, fn, tp = 0,0,0,0
    
    X = MinMaxScaler().fit_transform(X)
    X = pd.DataFrame(X)
    
    y_pred = cross_val_predict(model, X, y, cv=10)
    
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    
    acukNN = round(metrics.accuracy_score(y, y_pred),2)
    #acuNB = round((tp+tn)/(tp+tn+fp+fn),2)
    specificitykNN = round(tn / (tn+fp),2)
    sensitivitykNN = round(tp / (tp + fn),2)
    precisionkNN = round(metrics.precision_score(y, y_pred),2)
    f1kNN = round(metrics.f1_score(y, y_pred),2)


    print("b) Confusion matrix: ")

    conf_mat = confusion_matrix(y, y_pred)
    print(conf_mat)

          
    print("2.3 DT\n")
    print("a) Suggested parameterization: max_depth=5, min_samples_split=0.1, min_samples_leaf=0.04, criterion=entropy with SMOTE and KBest feature selection")


    model = DecisionTreeClassifier(max_depth=5, min_samples_split=0.1, min_samples_leaf=0.04, criterion="entropy", random_state=1)
    scoresNB = []
    y_predict_total =[]
    y_test_total =[]
    tn, fp, fn, tp = 0,0,0,0
    
    X = MinMaxScaler().fit_transform(X)
    X = pd.DataFrame(X)
    
    y_pred = cross_val_predict(model, X, y, cv=10)
    
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    
    acuDT = round(metrics.accuracy_score(y, y_pred),2)
    #acuNB = round((tp+tn)/(tp+tn+fp+fn),2)
    specificityDT = round(tn / (tn+fp),2)
    sensitivityDT = round(tp / (tp + fn),2)
    precisionDT= round(metrics.precision_score(y, y_pred),2)
    f1DT = round(metrics.f1_score(y, y_pred),2)


    print("b) Confusion matrix: ")

    conf_mat = confusion_matrix(y, y_pred)
    print(conf_mat)
    
    
    print("2.5 RF\n")
    print("a) Suggested parameterization: n_estimators=50, max_depth=25, max_features='log2' with SMOTE and KBest feature selection")


    model = RandomForestClassifier(n_estimators=50, max_depth=25, max_features='log2')

    scoresNB = []
    y_predict_total =[]
    y_test_total =[]
    tn, fp, fn, tp = 0,0,0,0
    
    X = MinMaxScaler().fit_transform(X)
    X = pd.DataFrame(X)
    
    y_pred = cross_val_predict(model, X, y, cv=10)
    
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    
    acuRF = round(metrics.accuracy_score(y, y_pred),2)
    #acuNB = round((tp+tn)/(tp+tn+fp+fn),2)
    specificityRF = round(tn / (tn+fp),2)
    sensitivityRF = round(tp / (tp + fn),2)
    precisionRF= round(metrics.precision_score(y, y_pred),2)
    f1RF = round(metrics.f1_score(y, y_pred),2)
    
    
    print("2.6 XGBoost\n")
    print("a) Suggested parameterization: learning_rate=0.08 with SMOTE and KBest feature selection")


    model = xgb.XGBClassifier(random_state=1,learning_rate=0.08)
    scoresNB = []
    y_predict_total =[]
    y_test_total =[]
    tn, fp, fn, tp = 0,0,0,0
    
    X = MinMaxScaler().fit_transform(X)
    X = pd.DataFrame(X)
    
    y_pred = cross_val_predict(model, X, y, cv=10)
    
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    
    acuXGBoost = round(metrics.accuracy_score(y, y_pred),2)
    #acuNB = round((tp+tn)/(tp+tn+fp+fn),2)
    specificityXGBoost = round(tn / (tn+fp),2)
    sensitivityXGBoost = round(tp / (tp + fn),2)
    precisionXGBoost= round(metrics.precision_score(y, y_pred),2)
    f1XGBoost = round(metrics.f1_score(y, y_pred),2)
    
    


    print("b) Confusion matrix: ")

    conf_mat = confusion_matrix(y, y_pred)
    print(conf_mat)

          
    print("3. Comparative performance: NB | kNN | DT | RF | XGBoost")

    print("3.1 Accuracy: ")
    
    print(str(acuNB)+" | "+ str(acukNN) +" | "+ str(acuDT)+" | "+ str(acuRF)+" | "+ str(acuXGBoost))

    # 0.76 | 0.81 | 0.56 | 0.90 3.2 

    print("3.2 Sensitivity: ")
    #  0.76 | 0.81 | 0.56 | 0.90 3.2 
    print(str(specificityNB)+" | "+ str(specificitykNN) +" | "+
          str(specificityDT)+" | "+ str(specificityRF)+" | "+ str(specificityXGBoost))

    print("3.3 Sensitivity: ")
    #  0.76 | 0.81 | 0.56 | 0.90 3.2 
    
    print(str(sensitivityNB)+" | "+ str(sensitivitykNN) +" | "+
          str(sensitivityDT)+" | "+ str(sensitivityRF)+" | "+ str(sensitivityXGBoost))
    
    print("3.5 Precision: ")
    #  0.76 | 0.81 | 0.56 | 0.90 3.2 
    
    print(str(precisionNB)+" | "+ str(precisionkNN) +" | "+
          str(precisionDT)+" | "+ str(precisionRF)+" | "+ str(precisionXGBoost))
    
    print("3.6 F1-Score: ")
    #  0.76 | 0.81 | 0.56 | 0.90 3.2 
    
    print(str(f1NB)+" | "+ str(f1kNN) +" | "+
          str(f1DT)+" | "+ str(f1RF)+" | "+ str(f1XGBoost))








def classification_ct_report(data):
    y = data.pop('Cover_Type').values
    df2 = pd.DataFrame(y, columns=['Cover_Type'])
    transf = Normalizer().fit(data)
    norm_data = pd.DataFrame(transf.transform(data, copy=True), columns= data.columns)
    norm_data = pd.concat([norm_data, df2], axis=1)
    unbal = norm_data
    
    X = unbal.drop(columns=['Cover_Type'])
    y = unbal['Cover_Type'].values
    
    sampler = RandomUnderSampler(random_state=12)
    X_res, y_res = sampler.fit_resample(X, y)
    
    
    data = pd.DataFrame(X_res, columns=unbal.columns[:-1])
    data.describe(include='all')
    df2 = pd.DataFrame(y_res, columns=['Cover_Type'])
    data = pd.concat([data, df2], axis=1)
    
    
    
    
    
    
    print("\n 2. Classifiers:")

    labels = pd.unique(y)

   
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None, stratify=y)
    
    cv = KFold(n_splits=10, random_state=42, shuffle=False)
    
    y = data.pop('Cover_Type').values
    X = data.values

    print("\n 2.1 NB")
    print("a) Suggested parameterization: MultinomialNB with Undersampling")

    model = MultinomialNB()
    
    X = MinMaxScaler().fit_transform(X)
    X = pd.DataFrame(X)
    
    y_pred = cross_val_predict(model, X, y, cv=10)
    
    from sklearn.metrics import multilabel_confusion_matrix
    
    acuNB = round(metrics.accuracy_score(y, y_pred),2)
    
    print("b) Confusion matrix: ")

    conf_mat = confusion_matrix(y, y_pred)
    print(conf_mat)
    
    
    print("\n 2.2 kNN")
    print("a) Suggested parameterization: n_neighbors=1, metric='manhattan' with Undersampling")

    model = KNeighborsClassifier(n_neighbors=1, metric='manhattan')
    
    X = MinMaxScaler().fit_transform(X)
    X = pd.DataFrame(X)
    
    y_pred = cross_val_predict(model, X, y, cv=10)
    
    acukNN = round(metrics.accuracy_score(y, y_pred),2)
    
    print("b) Confusion matrix: ")

    conf_mat = confusion_matrix(y, y_pred)
    print(conf_mat)
    
 

    print("\n 2.2 DT")
    print("a) Suggested parameterization: max_depth=13, min_samples_split=0.0002, min_samples_leaf=0.00005, criterion='entropy' with Undersampling")

    model = DecisionTreeClassifier(max_depth=13, min_samples_split=0.0002, min_samples_leaf=0.00005, criterion="entropy", random_state=1)

    
    X = MinMaxScaler().fit_transform(X)
    X = pd.DataFrame(X)
    
    y_pred = cross_val_predict(model, X, y, cv=10)
    
    acuDT = round(metrics.accuracy_score(y, y_pred),2)
    
    print("b) Confusion matrix: ")

    conf_mat = confusion_matrix(y, y_pred)
    print(conf_mat)
    
 

    print("\n 2.2 RF")
    print("a) Suggested parameterization: n_estimators=100, max_depth=25, max_features='log2' with Undersampling")

    model = RandomForestClassifier(n_estimators=100, max_depth=25, max_features='log2')
    
    X = MinMaxScaler().fit_transform(X)
    X = pd.DataFrame(X)
    
    y_pred = cross_val_predict(model, X, y, cv=10)
    
    acuRF = round(metrics.accuracy_score(y, y_pred),2)
    
    print("b) Confusion matrix: ")

    conf_mat = confusion_matrix(y, y_pred)
    print(conf_mat)
    
    
    print("\n 2.2 XGBoost")
    print("a) Suggested parameterization: learning_rate=0.1 with Undersampling")

    model = xgb.XGBClassifier(random_state=1,learning_rate=0.1)
    
    X = MinMaxScaler().fit_transform(X)
    X = pd.DataFrame(X)
    
    y_pred = cross_val_predict(model, X, y, cv=10)
    
    acuXGBoost = round(metrics.accuracy_score(y, y_pred),2)
    
    print("b) Confusion matrix: ")

    conf_mat = confusion_matrix(y, y_pred)
    print(conf_mat)
    
    
    
    print("3. Comparative performance: NB | kNN | DT | RF | XGBoost ")

    print("3.1 Accuracy: ")
    
    print(str(acuNB)+" | "+ str(acukNN) +" | "+ str(acuDT)+" | "+ str(acuRF)+" | "+ str(acuXGBoost))

    # 0.76 | 0.81 | 0.56 | 0.90 3.2 
    






def report(source, dataframe, task):
    task = task.strip()
    if task == "preprocessing":
        if source == "PD":
            return preprocessing_pd_report(dataframe)
        if source == "CT":
            return preprocessing_ct_report(dataframe)
    
    if task == "unsupervised":
        if source == "PD":
            return unsupervised_pd_report(dataframe)
        if source == "CT":
            return unsupervised_ct_report(dataframe)
    
    if task == "classification":
        if source == "PD":
            return classification_pd_report(dataframe)
        if source == "CT":
            return classification_ct_report(dataframe)
    

    

    return "Not yet available."




if __name__ == '__main__':

    '''A: read arguments'''
    args = sys.stdin.readline().rstrip('\n').split(' ')
    n, source, task = int(args[0]), args[1], args[2]
    
    '''B: read dataset'''
    data, header = [], sys.stdin.readline().rstrip('\r\n').split(',')
    
    for i in range(n-1):
        data.append(sys.stdin.readline().rstrip('\r\n').split(','))
    ## Tudo forcado a ser float64 pois eram objetos
    dataframe = pd.DataFrame(data, columns=header, dtype=float)
    
    #dataframe = pd.read_csv('pd_speech_features.csv', sep=',')
    #dataframe = pd.read_csv('convAfterUndersampling.csv', sep=',')
    #task = "classification"
    #source = "CT"
    
    
    '''C: output results'''
    print(report(source, dataframe, task))


