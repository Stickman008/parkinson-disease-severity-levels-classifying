import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report


def plot_history(history, show_accuracy=True, validate=True):
    plt.plot(history.history['loss'])
    if validate:
        plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    if validate:
        plt.legend(['train', 'validate'], loc='upper left')
    else:
        plt.legend(['train'], loc='upper left')
    plt.show()
    if show_accuracy:
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='best')
        plt.show()
    
def evaluate_model(model, X, y, show=False, batch_size=32):
    predict = model.predict(X, batch_size=batch_size)
    predict = np.argmax(predict, axis=1)+1
    real = np.argmax(y, axis=1)+1
    if show:
        f1_train = f1_score(real, predict)
        accuracy_train = accuracy_score(real, predict)
        print(classification_report(real, predict, digits=4))
        print("---------------------------------------------------------")
        sns.heatmap(confusion_matrix(real, predict),annot = True,fmt = '2.0f')
        plt.show()
    return classification_report(real, predict, digits=4, output_dict=True)