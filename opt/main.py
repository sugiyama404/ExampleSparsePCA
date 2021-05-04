import numpy as np
from sklearn.linear_model import LassoLarsIC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_friedman1
from sklearn.decomposition import SparsePCA

X, y = make_friedman1(n_samples=200, n_features=30, random_state=0)

# 標準化
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

spca_std = SparsePCA(n_components=5, random_state=0)
spca_std.fit(X)
X_transformed = spca_std.transform(X)

# データから抽出されたスパース成分。
print(spca_std.components_)
# 各イテレーションにおけるエラーのベクトル
print(spca_std.error_)
# 想定される構成要素の数(既存バグでエラー、予測変換でも出てこない)
# print(transformer.n_components_)
print(spca_std.n_components)
# イテレーションの実行回数
print(spca_std.n_iter_)
# トレーニングセットから推定された、各特徴の経験的平均値。X.mean(axis=0)と同じです。
print(spca_std.mean_)

# defauoltはalpha=1で出す。スパース性が低いとalpha=0が算出されるので注意が必要
model_bic = LassoLarsIC(criterion='bic')
model_bic.fit(X, y)
alpha_bic_ = model_bic.alpha_
print(alpha_bic_)
model_aic = LassoLarsIC(criterion='aic')
model_aic.fit(X, y)
alpha_aic_ = model_aic.alpha_
print(alpha_aic_)

spca_bic = SparsePCA(n_components=5, random_state=0, alpha=alpha_bic_)
spca_bic.fit(X)
X_spca_bic = spca_bic.transform(X)
print(X_spca_bic.shape)

spca_aic = SparsePCA(n_components=5, random_state=0, alpha=alpha_aic_)
spca_aic.fit(X)
X_spca_aic = spca_aic.transform(X)
print(X_spca_aic.shape)

print('defoult a=1                 :', np.mean(spca_std.components_ == 0))
print('bic a=0.02871277337904591   :', np.mean(spca_bic.components_ == 0))
print('aic a=0.022802510590416925  :', np.mean(spca_aic.components_ == 0))
