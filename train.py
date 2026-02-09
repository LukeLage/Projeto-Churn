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
feature_importance[feature_importance[0]> 0.01]