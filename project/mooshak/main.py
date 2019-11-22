import sys, pandas as pd
from preprocessing import preprocessing_pd_report

def report(source, dataframe, task):
    task = task.strip()
    if task == "preprocessing":
        return preprocessing_pd_report(dataframe)
    else:
        return "Not yet available."

if __name__ == '__main__':

    '''A: read arguments'''
    args = sys.stdin.readline().rstrip('\n').split(' ')
    n, source, task = int(args[0]), args[1], args[2]
    
    '''B: read dataset'''
    data, header = [], sys.stdin.readline().rstrip('\r\n').split(',')
    for i in range(n-1):
        data.append(sys.stdin.readline().rstrip('\r\n').split(','))
    # Tudo forçado a ser float64 pois eram objetos
    dataframe = pd.DataFrame(data, columns=header, dtype=float)

    '''C: output results'''
    print(report(source, dataframe, task))
