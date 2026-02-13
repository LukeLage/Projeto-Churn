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

from feature_engine import discretisation

tree_discretisation = discretisation.DecisionTreeDiscretiser(variables= best_features,
                                                            regression= False,
                                                            bin_output= 'bin_number',
                                                            cv = 3)

tree_discretisation.fit(X_train[best_features], y_train)

# %%

X_train_transform = tree_discretisation.transform(X_train[best_features])
X_train_transform

# %%
# MODEL

from sklearn import linear_model

reg = linear_model.LogisticRegression(penalty= None, random_state= 42, max_iter= 100000)
reg.fit(X_train_transform, y_train)

# %%

from sklearn import metrics

y_train_predict = reg.predict(X_train_transform)
y_train_proba = reg.predict_proba(X_train_transform)[:, 1]

acc_train = metrics.accuracy_score(y_train, y_train_predict)
auc_train = metrics.roc_auc_score(y_train, y_train_proba)
print('Acurácia de Treino: ', acc_train)
print('AUC de Treino: ', auc_train)

# %%

X_test_transform = tree_discretisation.transform(X_test[best_features])

y_test_predict = reg.predict(X_test_transform)
y_test_proba = reg.predict_proba(X_test_transform)[:, 1]

acc_test = metrics.accuracy_score(y_test, y_test_predict)
auc_test = metrics.roc_auc_score(y_test, y_test_proba)
print('Acurácia de Teste: ', acc_test)
print('AUC de Teste: ', auc_test)

# %% 

oot_transform = tree_discretisation.transform(oot[best_features])

y_oot_predict = reg.predict(oot_transform)
y_oot_proba = reg.predict_proba(oot_transform)[:, 1]

acc_oot = metrics.accuracy_score(oot[target], y_oot_predict)
auc_oot = metrics.roc_auc_score(oot[target], y_oot_proba)
print('Acurácia de OOT: ', acc_oot)
print('AUC de OOT: ', auc_oot)