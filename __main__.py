import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('data/KNN_Project_Data.csv')
df.head()

sns.pairplot(df, hue='TARGET CLASS')

predictors = df.drop('TARGET CLASS', axis=1)
y = df['TARGET CLASS']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(predictors)

X = pd.DataFrame(scaler.transform(predictors), columns=predictors.columns)
X.head()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=101
)

from sklearn.neighbors import KNeighborsClassifier

def predict_y(k = 1):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    return knn.predict(X_test)

error_rate = []
for i in range(1, 100):
    y_pred = predict_y(i)
    error_rate.append(np.mean(y_pred != y_test))

error_df = pd.DataFrame(error_rate, columns=['Error rate'])
error_df.describe()

optimal_ks = error_df[error_df['Error rate'] == min(error_df['Error rate'])]

from sklearn.metrics import confusion_matrix, classification_report

current_accuracy = 0
optimal_k = 0
for i in optimal_ks.index:
    y_pred = predict_y(i)
    pred_f, pred_t = confusion_matrix(y_test, y_pred)
    tn, fn = pred_f
    fp, tp = pred_t
    print(f'TN\t{tn}\nFN\t{fn}\nFP\t{fp}\nTP\t{tp}')
    accuracy = (tn + tp)/(tn + tp + fp + fn)
    if accuracy > current_accuracy:
        current_accuracy = accuracy
        optimal_k = i

y_pred = predict_y(optimal_k)
print(classification_report(y_test, y_pred))
