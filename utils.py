import os
import numpy as np
import pandas as pd

from scipy.signal import savgol_filter, medfilt
from scipy.ndimage import maximum_filter

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

N_OF_TIMESTEP = 854
N_OF_FEATURE = 50

def mod_X(x_df):
    MY_COL = [f'{i}{ee}' for i in range(N_OF_FEATURE//2) for ee in ['X', 'Y']]
    ID_ROW = np.array([i for i in range(len(x_df)) for _ in range(N_OF_TIMESTEP)])
    TIMESTEP =  np.array([j for _ in range(len(x_df)) for j in range(N_OF_TIMESTEP)])
    
    result = x_df.to_numpy()
    result = result.reshape(-1, N_OF_FEATURE)
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


def convert_to_numpy(pd_series):
    if type(pd_series).__module__ != np.__name__:
        return pd_series.to_numpy()
    else:
        return pd_series.copy()

def apply_savgol_filter(np_array, window_size=21, polynomial=1):
    preprocessed_array = convert_to_numpy(np_array)
    preprocessed_array = preprocessed_array
    preprocessed_array[:] = savgol_filter(preprocessed_array[:], window_size, polynomial)
    return preprocessed_array.reshape(-1)

def apply_median_filter(np_array, window_size=17):
    preprocessed_array = convert_to_numpy(np_array)
    preprocessed_array = preprocessed_array.reshape(-1, N_OF_TIMESTEP)
    preprocessed_array[:] = medfilt(preprocessed_array[:], window_size)
    return preprocessed_array.reshape(-1)

def apply_maximum_filter(np_array, window_size=25):
    preprocessed_array = convert_to_numpy(np_array)
    preprocessed_array = preprocessed_array.reshape(-1, N_OF_TIMESTEP)
    preprocessed_array[:] = maximum_filter(preprocessed_array[:], window_size)
    return preprocessed_array.reshape(-1)

def apply_is_zero(np_array):
    preprocessed_array = convert_to_numpy(np_array)
    is_zero = (preprocessed_array<1).astype(np.float32)
    return is_zero

def apply_std(np_array):
    preprocessed_array = convert_to_numpy(np_array)
    preprocessed_array = preprocessed_array.reshape(-1, N_OF_TIMESTEP)
    preprocessed_array = np.std(preprocessed_array[:], axis=1).reshape(-1, 1)
    preprocessed_array = np.repeat(preprocessed_array, N_OF_TIMESTEP, axis=1)
    return preprocessed_array.reshape(-1)

def apply_mean(np_array):
    preprocessed_array = convert_to_numpy(np_array)
    preprocessed_array = preprocessed_array.reshape(-1, N_OF_TIMESTEP)
    preprocessed_array = np.mean(preprocessed_array[:], axis=1).reshape(-1, 1)
    preprocessed_array = np.repeat(preprocessed_array, N_OF_TIMESTEP, axis=1)
    return preprocessed_array.reshape(-1)

def apply_max_value(np_array):
    preprocessed_array = convert_to_numpy(np_array)
    preprocessed_array = preprocessed_array.reshape(-1, N_OF_TIMESTEP)
    preprocessed_array = np.max(preprocessed_array[:], axis=1).reshape(-1, 1)
    preprocessed_array = np.repeat(preprocessed_array, N_OF_TIMESTEP, axis=1)
    return preprocessed_array.reshape(-1)

def apply_min_value(np_array):
    preprocessed_array = convert_to_numpy(np_array)
    preprocessed_array = preprocessed_array.reshape(-1, N_OF_TIMESTEP)
    preprocessed_array = np.min(preprocessed_array[:], axis=1).reshape(-1, 1)
    preprocessed_array = np.repeat(preprocessed_array, N_OF_TIMESTEP, axis=1)
    return preprocessed_array.reshape(-1)

def plot_history(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validate'], loc='upper left')
    plt.show()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')
    plt.show()
    
def evaluate_model(model, X, y, show=False):
    predict = model.predict(X, batch_size=4)
    predict = np.argmax(predict, axis=1)+1
    real = np.argmax(y, axis=1)+1
    if show:
        f1_train = f1_score(real, predict)
        accuracy_train = accuracy_score(real, predict)
        print(classification_report(real, predict, digits=4))
        print("---------------------------------------------------------")
        sns.heatmap(confusion_matrix(real, predict),annot = True,fmt = '2.0f')
        plt.show()
    return classification_report(real, predict, digits=4)
    

if __name__ == '__main__':
    train_df = pd.read_csv(os.path.join('data', 'unionTrain.csv'))
    # X_train, y_train = train_df.drop(['Severity', 'sequence_id'], axis=1), train_df['Severity']
    # print(train_df)
    X_train, y_train = mod_df(train_df)
    print(X_train['0X'].shape)
    print(apply_std(X_train['0X']))
    # print(X_train)
    # print(y_train)
