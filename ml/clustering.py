"""Unsupervised lifestyle profiles, independent of the stress labels."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from ml.config import CLUSTER_FEATURES, DISPLAY_NAMES, SEED
from ml.preprocessing import make_preprocessor


def train_clusters(X_train, k_values=range(2, 7)):
    preprocess = make_preprocessor(CLUSTER_FEATURES)
    transformed = preprocess.fit_transform(X_train)
    rows, candidates = [], {}
    for k in k_values:
        model = KMeans(n_clusters=k, n_init=10, random_state=SEED)
        labels = model.fit_predict(transformed)
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(transformed, labels, sample_size=min(1000, len(labels)), random_state=SEED)
        rows.append({'k': k, 'inertia': float(model.inertia_), 'silhouette': float(score)})
        candidates[k] = model
    if not rows:
        raise ValueError('Clustering needs at least two distinct lifestyle patterns.')
    scores = pd.DataFrame(rows)
    best_k = int(scores.sort_values('silhouette', ascending=False).iloc[0]['k'])
    model = candidates[best_k]
    pipeline = Pipeline([('preprocess', preprocess), ('model', model)])
    imputed = preprocess.named_steps['columns'].named_transformers_['numeric'].named_steps['scale'].inverse_transform(transformed)
    profiles = pd.DataFrame(imputed, columns=CLUSTER_FEATURES).assign(cluster=model.labels_).groupby('cluster').mean()
    profiles['members'] = pd.Series(model.labels_).value_counts().sort_index()
    descriptions = {}
    for cluster_id, center in enumerate(model.cluster_centers_):
        strongest = np.argsort(np.abs(center))[::-1][:2]
        traits = [f'{DISPLAY_NAMES[CLUSTER_FEATURES[i]]}: {"above" if center[i] >= 0 else "below"} training average'
                  for i in strongest]
        descriptions[str(cluster_id)] = '; '.join(traits)
    pca = PCA(n_components=2, random_state=SEED).fit(transformed)
    coordinates = pd.DataFrame(pca.transform(transformed), columns=['PC1', 'PC2'])
    coordinates['cluster'] = model.labels_
    return {'pipeline': pipeline, 'scores': scores, 'profiles': profiles,
            'descriptions': descriptions, 'pca': pca, 'coordinates': coordinates,
            'best_k': best_k}
