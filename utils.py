import os
import numpy as np
import pandas as pd

N_OF_TIMESTEP = 854
N_OF_FEATURE = 50

def mod_X(x):
    MY_COL = [f"{i}{ee}" for i in range(N_OF_FEATURE//2) for ee in ['X', 'Y']]
    ID_ROW = np.array([i for i in range(len(x)) for _ in range(N_OF_TIMESTEP)])
    TIMESTEP =  np.array([j for _ in range(len(x)) for j in range(N_OF_TIMESTEP)])
    
    result = x.to_numpy()
    result = result.reshape(-1, 50)
    result = pd.DataFrame(result, columns=MY_COL)
    result.insert(0, 'id', ID_ROW)
    result.insert(1, 'timestep', TIMESTEP)
    return result

if __name__ == '__main__':
    train_df = pd.read_csv(os.path.join('data', 'unionTrain.csv'))
    X_train, y_train = train_df.drop(['Severity', 'sequence_id'], axis=1), train_df['Severity']
    modded_X = mod_X(X_train)
    print(modded_X)