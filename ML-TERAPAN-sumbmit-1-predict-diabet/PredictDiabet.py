# %% [markdown]
# Data Loading
# 

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score,GridSearchCV

from sklearn.metrics import make_scorer, f1_score
import numpy as np

# %%
dataset = pd.read_csv('./diabetes.csv')
dataset.info()

# %% [markdown]
# **EDA**
# 

# %%
dataset.head(5)

# %% [markdown]
# Exploratory Data Analysis - Deskripsi Variabel
# 

# %%
isna_isnull = pd.DataFrame({
    'isna': dataset.isna().sum(),
    'isnull': dataset.isnull().sum(),
})

isna_isnull

# %%
dataset.describe()

# %%
def build_boxplot(title: str):
    plt.figure(figsize=(16, 16))
    plt.title(title)
    for index, column in enumerate(dataset.columns, start=1):
        plt.subplot(3, 3, index)
        sns.boxplot(x=dataset[column])

# %% [markdown]
# CEK OUTLIER
# 

# %%
build_boxplot("SEBELUM HANDLING OUTLIER")

# %%
Q1 = dataset.quantile(0.25, numeric_only=True)
Q3 = dataset.quantile(0.75, numeric_only=True)
IQR = Q3-Q1

dataset = dataset[~((dataset < (Q1-1.5*IQR)) |
                    (dataset > (Q3+1.5*IQR))).any(axis=1)]

# %%
build_boxplot("SESUDAH HANDLING OUTLIER")

# %%
# Data label dan jumlahnya
label_count = dataset.Outcome.value_counts()
out = ['Diabet', 'Non Diabet']

# Membuat diagram batang
plt.bar(out[0], label_count[0])
plt.bar(out[1], label_count[1])

# Menambahkan label sumbu dan judul diagram
plt.xlabel('Kategori')
plt.ylabel('Jumlah')
plt.title('Dataset Diabet dan Non Diabet')
plt.legend(['diabet', 'non diabet'])
plt.show()

# %%
plt.figure(figsize=(18, 18))
list_col = dataset.drop(columns='Outcome').columns

for index, col in enumerate(list_col, start=1):
    plt.subplot(5, 2, index)
    sns.histplot(
        data=dataset,
        x=col,
        kde=True,
        hue='Outcome',
        alpha=0.3,
    )

# %%
sns.pairplot(dataset, hue='Outcome', diag_kind='kde')

# %%
plt.figure(figsize=(12, 8))
corelation_matrix = dataset.corr().round(2)
sns.heatmap(data=corelation_matrix, annot=True,
            cmap='coolwarm', linewidths=0.6)
plt.title("Correlation Matrix ", size=20)

# %%
# - Menggunakan heatmap untuk mencari korelasi yang sangat berpengaruh dengan diabetes.
# Dari data set diperoleh urutan dari yang paling berpengaruh yaitu Glucosa, BMI, Age, Pregnancies, BloodPressure, DiabetesPedingFreeFunction, Insulin, dan yang paling jauh Insulin
np.sort([1, 0.26, 0.18, 0.27, 0.1, 0.03, 0.18, 0.49, 0.23])

# %% [markdown]
# Data Preparation
# 

# %%
X = dataset.drop("Outcome", axis=1)
y = dataset["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f'Total # of sample in whole dataset: {len(X)}')
print(f'Total # of sample in train dataset: {len(X_train)}')
print(f'Total # of sample in test dataset: {len(X_test)}')

# %%
list_matrix = []

# %%
def append_matrix(name: str, accuracy: float, precision: float, recall: float, f1: float, roc_auc: float):
    list_matrix.append({
        'name': name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    })

# %%
model_dict = {
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'AdaBoostClassifier': AdaBoostClassifier()
}

# %%
for name, model in model_dict.items():
    clf = model
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    append_matrix(name, acc, prec, rec, f1, roc_auc)

# %%
result_eval = pd.DataFrame(list_matrix)
result_eval

# %%
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [2, 4, 8],
    'subsample': [0.8, 0.9, 1.0],
    'criterion': ['friedman_mse']
}

# best = 0.0
# epoch = 0
# while (best <= 0.75 and epoch < 101):
# epoch += 1
# print(epoch)


def custom_f1(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, average='micro')
    return 1 if f1 >= 0.8 else 0


clf = GradientBoostingClassifier()

grid_search = GridSearchCV(
    clf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='f1_macro')

grid_search.fit(X_train, y_train)

# Membuat model dengan hyperparameter terbaik
current_best_clf = grid_search.best_estimator_

# Melakukan prediksi
y_pred = current_best_clf.predict(X_test)

# Menghitung metrik evaluasi
prec = precision_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="micro")
roc_auc = roc_auc_score(y_test, y_pred)
current_best_clf

# Setelah loop selesai, Anda memiliki model terbaik
print("Model Terbaik:")
print(f1)

# %%
current_best_clf


