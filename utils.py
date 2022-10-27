import os
import numpy as np
import pandas as pd

N_OF_TIMESTEP = 854
N_OF_FEATURE = 50

def mod_X(x_df):
    MY_COL = [f"{i}{ee}" for i in range(N_OF_FEATURE//2) for ee in ['X', 'Y']]
    ID_ROW = np.array([i for i in range(len(x_df)) for _ in range(N_OF_TIMESTEP)])
    TIMESTEP =  np.array([j for _ in range(len(x_df)) for j in range(N_OF_TIMESTEP)])
    
    result = x_df.to_numpy()
    result = result.reshape(-1, 50)
    result = pd.DataFrame(result, columns=MY_COL)
    result.insert(0, 'id', ID_ROW)
    result.insert(1, 'timestep', TIMESTEP)
    return result

def inverse_mod_X(X_np):
    result = X_np.copy()
    result = result.reshape(-1, N_OF_TIMESTEP, result.shape[1])
    return result

def mod_y(y_df):
    result = pd.get_dummies(y_df)
    return result

def inverse_mod_y(y_np):
    result = y_np.to_numpy()
    return result

def mod_df(df):
    X_result, y_result = df.drop(['Severity', 'sequence_id'], axis=1), df['Severity']
    X_result, y_result = mod_X(X_result), mod_y(y_result)
    return X_result, y_result
    

def drop_features(df, features_number_list):
    features_list = [f'{e}{t}' for e in features_number_list for t in ['X', 'Y']]
    result = df.drop(features_list, axis=1)
    return result

if __name__ == '__main__':
    train_df = pd.read_csv(os.path.join('data', 'unionTrain.csv'))
    # X_train, y_train = train_df.drop(['Severity', 'sequence_id'], axis=1), train_df['Severity']
    # modded_X = mod_X(X_train)
    # print(modded_X)
    X_train, y_train = mod_df(train_df)
    print(X_train)
    print(y_train)