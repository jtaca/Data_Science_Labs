import sys, pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE, RandomOverSampler

def preprocessing_ct_report(data):
    print("-- Report start --")
    print("Initial shape:", data.shape)
    print("1 - Preprocessing")
    
    # data = remove_outliers(data)
    data = resample_missing_values(data)
    data = normalization(data)
    # TODO: mudar o resto para a versão que funciona com o covtype
    #data = variable_dummification(data)
    #data = balancing(data)
    #data = feature_selection(data)
    
    print("Final shape:", data.shape)
    return "-- End of report --"
    

# TODO: not working
def remove_outliers(data):
    y = data.pop('Cover_Type').values
    df2 = pd.DataFrame(y, columns=['Cover_Type'])
    
    for col_name in data.keys():
        q1 = data[col_name].quantile(0.25)
        q3 = data[col_name].quantile(0.75)
        iqr = q3-q1
        #Interquartile range
        fence_low = q1-1.5*iqr
        fence_high = q3+1.5*iqr
        data = data.loc[(data[col_name] > fence_low) & (data[col_name] < fence_high)]
        print(data)
        
    data = pd.concat([data, df2], axis=1)
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
    print("1.1 - Missing Values: CovType dataset has no missing values so nothing is done here.")
    return data
    
    
def normalization(data):
    y = data.pop('Cover_Type').values
    df2 = pd.DataFrame(y, columns=['Cover_Type'])
    transf = Normalizer().fit(data)
    norm_data = pd.DataFrame(transf.transform(data, copy=True), columns= data.columns)
    norm_data = pd.concat([norm_data, df2], axis=1)
    print("1.2 - Normalization: normalized using Normalizer from sklearn")
    return norm_data
    
    
def variable_dummification(data):
    print("1.4 - Variable Dummification: PD dataset has no cathegorical variables so nothing is done here.")
    return data


def balancing(data):
    print("1.5 - Balancing:")
    unbal = data
    target_count = unbal['Cover_Type'].value_counts()

    min_class = target_count.idxmin()
    ind_min_class = target_count.index.get_loc(min_class)

    print('a) Majority Cover_Type:', target_count[ind_min_class], '| Minority Cover_Type:', target_count[1-ind_min_class], '| Proportion:', round(target_count[ind_min_class] / target_count[1-ind_min_class], 2), ': 1')

    RANDOM_STATE = 42
    target_values_0 = target_count.values[ind_min_class]
    target_values_1 = target_count.values[1-ind_min_class]
    values = {'data': [target_values_0, target_values_1]}
    
    if target_values_0 != target_values_1:
        df_class_min = unbal[unbal['Cover_Type'] == min_class]
        df_class_max = unbal[unbal['Cover_Type'] != min_class]

        df_under = df_class_max.sample(len(df_class_min))
        values['UnderSample'] = [target_count.values[ind_min_class], len(df_under)]

        df_over = df_class_min.sample(len(df_class_max), replace=True)
        values['OverSample'] = [len(df_over), target_count.values[1-ind_min_class]]

        smote = SMOTE(ratio='minority', random_state=RANDOM_STATE)

        y = unbal.pop('Cover_Type').values
        X = unbal.values
        smote_x, smote_y = smote.fit_sample(X, y)
        smote_target_count = pd.Series(smote_y).value_counts()

        df_SMOTE = pd.DataFrame(smote_x)
        df_SMOTE.columns = unbal.columns
        df_SMOTE['Cover_Type'] = smote_y

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
    # TODO: deve ser sempre chamado ou chama se depois individualmente nos algoritmos que se quer?
    return data
