# %% 

import pandas as pd

df = pd.read_csv('data/data_churn.csv')
df.head()

# %% 

oot = df[df['dtRef'] == df['dtRef'].max()].copy()

# %% 

df_train = df[df['dtRef'] < df['dtRef'].max()].copy()

# %% 

features = df_train.columns[2: -1]
target = 'flagChurn'

X, y = df_train[features], df_train[target]

# %%
# SAMPLE

from sklearn import model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,
                                                                    random_state= 42,
                                                                    test_size= 0.2, 
                                                                    stratify= y
                                                                    )

# %% 

print('Taxa váriavel resposta geral: ', y.mean())
print('Taxa váriavel resposta de Treino: ', y_train.mean())
print('Taxa váriavel resposta de Teste: ', y_test.mean())

# %%
# EXPLORE (MISSING VALUES)

X_train.isna().sum().sort_values(ascending= False)

# %% 

df_analise = X_train.copy()
df_analise[target] = y_train

summario = df_analise.groupby(by=target).agg(('mean', 'median')).T
summario

summario['diff_abs'] = summario[0] - summario[1]
summario['diff_rel'] = summario[0] / summario[1]
summario.sort_values(by= 'diff_rel', ascending= False)

# %%

from sklearn import tree
import matplotlib.pyplot as plt

arvore = tree.DecisionTreeClassifier(random_state= 42)
arvore.fit(X_train, y_train)

# %% 

feature_importance = (pd.Series(arvore.feature_importances_, index= X_train.columns)
                        .sort_values(ascending= False)
                        .reset_index()
                        )

feature_importance['acum'] = feature_importance[0].cumsum()
feature_importance[feature_importance[0] < 0.96]

# %%

best_features = feature_importance[feature_importance[0] < 0.96]['index'].tolist()
best_features

# %%
# MODIFY 

from feature_engine import discretisation, encoding
from sklearn import pipeline

tree_discretisation = discretisation.DecisionTreeDiscretiser(variables= best_features,
                                                            regression= False,
                                                            bin_output= 'bin_number',
                                                            cv = 3)


onehot = encoding.OneHotEncoder(variables= best_features, ignore_format= True)

# %%
# MODEL

from sklearn import linear_model

model = linear_model.LogisticRegression(penalty= None, random_state= 42, max_iter= 100000)

model_pipeline = pipeline.Pipeline(
    steps= [
        ('Discretizar', tree_discretisation),
        ('OneHot', onehot),
        ('Model', model)
    ]
)

model_pipeline.fit (X_train, y_train)

# %%

from sklearn import metrics

y_train_predict = model_pipeline.predict(X_train)
y_train_proba = model_pipeline.predict_proba(X_train)[:, 1]

acc_train = metrics.accuracy_score(y_train, y_train_predict)
auc_train = metrics.roc_auc_score(y_train, y_train_proba)

roc_train = metrics.roc_curve(y_train, y_train_proba)

print('Acurácia de Treino: ', acc_train)
print('AUC de Treino: ', auc_train)

# %%

y_test_predict = model_pipeline.predict(X_test)
y_test_proba = model_pipeline.predict_proba(X_test)[:, 1]

acc_test = metrics.accuracy_score(y_test, y_test_predict)
auc_test = metrics.roc_auc_score(y_test, y_test_proba)

roc_test = metrics.roc_curve(y_test, y_test_proba)

print('Acurácia de Teste: ', acc_test)
print('AUC de Teste: ', auc_test)

# %% 

y_oot_predict = model_pipeline.predict(oot[features])
y_oot_proba = model_pipeline.predict_proba(oot[features])[:, 1]

acc_oot = metrics.accuracy_score(oot[target], y_oot_predict)
auc_oot = metrics.roc_auc_score(oot[target], y_oot_proba)

roc_oot = metrics.roc_curve(oot[target], y_oot_proba)

print('Acurácia de OOT: ', acc_oot)
print('AUC de OOT: ', auc_oot)


# %%

plt.figure(dpi= 400)
plt.plot(roc_train[0], roc_train[1], label= 'Treino')
plt.plot(roc_test[0], roc_test[1], label= 'Teste')
plt.plot(roc_oot[0], roc_oot[1], label= 'OOT')
plt.plot([0, 1], [0, 1], '--', color= 'grey')
plt.grid(True)
plt.ylabel ('Sensibilidade')
plt.xlabel ('1 - Especificidade')
plt.title('Curva ROC')
plt.legend(
    [f'Treino: :{100*auc_train: .2f}',
    f'Teste: :{100*auc_test: .2f}',
    f'Out-Of-Time: :{100*auc_oot: .2f}']
)

# %% 

