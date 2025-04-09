import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score


  
RANDOM_STATE = 42
plt.rcParams['figure.dpi'] = 150

# fetch dataset 
df = pd.read_csv("Wholesale customers data.csv")
  
spend_cols = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']

X_raw = df[spend_cols].copy()
y_cannel = df["Channel"] - 1

# Cleaning and transformation

X_log = np.log1p(X_raw)
clip_upper = X_log.quantile(0.99)
X_log = X_log.clip(upper=clip_upper, axis=1)

# Gaussian Mixture

scaler = StandardScaler.fit(X_log)
X_scaled = scaler.transform(X_log)

bic, aic, models = [], [], {}
for k in range(2, 10):
    gmm = GaussianMixture(n_components=k, random_state=RANDOM_STATE)
    gmm.fit(X_scaled)
    bic.append(gmm.bic(X_scaled))
    aic.append(gmm.aic(X_scaled))
    models[k] = gmm

best_k = min(range(2,10), key=lambda k: bic[k-2])
best_gmm = models[best_k]
labels = best_gmm.predict(X_scaled)

