import sys, pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE, RandomOverSampler

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler

def preprocessing_pd_report(data):
    print("-- Report start --")
    print("Initial shape:", data.shape)
    print("1 - Preprocessing")
    
    data = remove_same_id_samples(data)
    # data = remove_outliers(data)
    data = resample_missing_values(data)
    data = normalization(data)
    data = variable_dummification(data)
    data = balancing(data)
    data = feature_selection(data)
    
    print("Final shape:", data.shape)
    return "-- End of report --"


def remove_same_id_samples(data):
    init_n_rows = str(data.shape[0])
    data = data.sort_values('id', ascending=True)
    data = data.groupby('id').mean().reset_index()
    final_n_rows = str(data.shape[0])

    
    if init_n_rows != final_n_rows:
        print("1.1 - Removed repeated 'id' using the mean value, reduced from " + init_n_rows + " rows to " + final_n_rows + ".")
    else:
        print("1.1 - No repeated 'id' values to be removed.")
    return data
    

# TODO: not working great
def remove_outliers(data):
    y = data.pop('class').values
    col_id = data.pop('id').values
    df1 = pd.DataFrame(col_id, columns=['id'])
    df2 = pd.DataFrame(y, columns=['class'])
    
    for col_name in data.keys():
        q1 = data[col_name].quantile(0.25)
        q3 = data[col_name].quantile(0.75)
        iqr = q3-q1
        #Interquartile range
        fence_low = q1-1.5*iqr
        fence_high = q3+1.5*iqr
        data = data.loc[(data[col_name] > fence_low) & (data[col_name] < fence_high)]
        print(data)
        
    data = pd.concat([df1, data, df2], axis=1)
    print(data.shape)
    print(data)
        
    return data
    

def resample_missing_values(data):
    # TODO: does this work?
    #try:
    #    imp = SimpleImputer(strategy='mean', missing_values=np.nan, copy=True, add_indicator=True)
    #    imp.fit(data.values)
    #    mat = imp.transform(data.values)
    #    print(mat)
    #    data = pd.DataFrame(mat, columns=data.columns)
    #except:
    #    print("Has missing values")
    print("1.2 - Missing Values: PD dataset has no missing values so nothing is done here.")
    return data
    
    
def normalization(data):
    y = data.pop('class').values
    df2 = pd.DataFrame(y, columns=['class'])
    transf = Normalizer().fit(data)
    norm_data = pd.DataFrame(transf.transform(data, copy=True), columns= data.columns)
    norm_data = pd.concat([norm_data, df2], axis=1)
    print("1.3 - Normalization: normalized using Normalizer from sklearn")
    return norm_data
    
    
def variable_dummification(data):
    print("1.4 - Variable Dummification: PD dataset has no cathegorical variables so nothing is done here.")
    return data


def balancing(data):
    print("1.5 - Balancing:")
    unbal = data
    target_count = unbal['class'].value_counts()

    min_class = target_count.idxmin()
    ind_min_class = target_count.index.get_loc(min_class)

    print('a) Majority class:', target_count[ind_min_class], '| Minority class:', target_count[1-ind_min_class], '| Proportion:', round(target_count[ind_min_class] / target_count[1-ind_min_class], 2), ': 1')

    RANDOM_STATE = 42
    target_values_0 = target_count.values[ind_min_class]
    target_values_1 = target_count.values[1-ind_min_class]
    values = {'data': [target_values_0, target_values_1]}
    
    if target_values_0 != target_values_1:
        df_class_min = unbal[unbal['class'] == min_class]
        df_class_max = unbal[unbal['class'] != min_class]

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
        
        print("b) Balanced dataset options:")
        for a in values:
            print('-', a, values[a])

        print("c) Dataset balanced using SMOTE.")
        return df_SMOTE
            
    else:
        print("b) Dataset was already balanced.")
    
    return data
    
    
def feature_selection(data):
    print("1.6 - Balancing:")
    print("a) For best classification accuracy in NB, kNN, DT we do KBest feature selection.")
    # TODO: deve ser sempre chamado ou chama se depois individualmente nos algoritmos que se quer?
    print("Size before feature Selection: "+str(data.shape))
    y = data.pop('class').values
    X = data.values
    X_norm = MinMaxScaler().fit_transform(X)

    X_norm = pd.DataFrame(X_norm, columns=data.columns)
    #print(X_norm)
    kbest = SelectKBest(chi2, k=100)
    X_new = kbest.fit_transform(X_norm, y)
    print("Shape after reduction : "+str(X_new.shape))

    column_names = data.columns[kbest.get_support()]
    X_new = pd.DataFrame(X_norm, columns=column_names)

    df2 = pd.DataFrame(y, columns=['class'])
    resultKBest = pd.concat([X_new, df2], axis=1)

    print("b) For the classification using RF we do not do any feature selection.")
    return resultKBest
