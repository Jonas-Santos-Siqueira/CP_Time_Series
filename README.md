# **Intervalos de Predição via Bootstrap em Ensemble** - (EnbPI)

**Ensemble Bootstrap Prediction Intervals** (EnbPI) para **regressão em séries temporais**, com API no estilo `statsmodels` e suporte a **Moving Block Bootstrap (MBB)**.  
O método constrói **intervalos de predição conformais** (livres de distribuição) a partir de um **ensemble bootstrap** e **resíduos OOB (leave‑one‑out)**. Em predição, os intervalos podem ser **atualizados sequencialmente** via **janela deslizante** sem re‑treinar o ensemble.

---

## Como funciona (resumo)

Para uma série alvo $$y_t$$ com regressoras $$x_t$$:

1. Treine $$B$$ modelos-base em **amostras bootstrap** do treino (i.i.d. ou **blocos contíguos** para séries temporais).
2. Para cada $$i$$ no treino, agregue as predições **dos modelos que não viram $$i$$** (OOB) e calcule o **resíduo não‑conforme** $$\hat\varepsilon_i = |y_i - \hat f_{-i}(x_i)|$$.
3. Para um novo $$x_t$$, o **centro** do intervalo é:
   - $$\hat f^{\phi}(x_t)$$ (agregação `mean`/`median`), ou
   - o **quantil $$(1-\alpha)$$** das predições leave‑one‑out $$\{\hat f_{-i}(x_t)\}_{i=1}^T$$ (`center="loo_quantile"`).
4. A **meia‑largura** $$w_t$$ é o **quantil $$(1-\alpha)$$** dos resíduos atuais (com **janela deslizante** opcional).  
5. Intervalo: $$[\,\text{centro} \pm w_t\,]$$.

---

## Instalação / Uso rápido

1) Baixe o pacote e adicione a pasta ao `PYTHONPATH` (ou `sys.path.append`).  
2) Importe e use:

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from enbpi import EnbPIModel

# dados sintéticos
rng = np.random.default_rng(42)
X = np.linspace(0, 10, 200)[:, None]
y = 2.0*X[:, 0] + rng.normal(scale=1.0, size=len(X))

# split
X_train, y_train = X[:150], y[:150]
X_test,  y_test  = X[150:], y[150:]

# EnbPI com bootstrap i.i.d. e centro como quantil LOO
model = EnbPIModel(
    base_model=LinearRegression(),
    B=30, alpha=0.1,
    aggregation="mean",
    center="loo_quantile",
    bootstrap="iid",
    batch_size=1,
    random_state=123,
)
model.fit(X_train, y_train)
res = model.get_prediction(X_test, y_true=y_test)
print(res.summary())
# res.plot_interval(show=True)
```

---

## Multi‑passos à frente (H‑step)

O EnbPI não se limita a 1‑passo. Para $$H$$ passos (ex.: 6):

- **Direto por horizonte (recomendado):** treine **um modelo por horizonte** $$h=1,\dots,H$$ com alvo $$y_{t+h}$$ e features em $$t$$. Cada modelo possui seus próprios resíduos OOB/quantis.
- **Recursivo:** aplique o mesmo modelo de 1‑passo $$H$$ vezes alimentando predições como lags (mais simples, porém acumula erro).  
- **Cobertura simultânea:** se desejar $$1-\alpha$$ simultâneo em $$H$$ pontos, ajuste o nível por passo (ex.: Bonferroni $$\alpha'=\alpha/H$$ ou Sidák $$\alpha'=1-(1-\alpha)^{1/H}$$).

---

## Parâmetros importantes

- **`B`** (nº de modelos): 20–50 costuma funcionar bem.  
- **`bootstrap`**: `iid` (amostragem por linhas) ou `block` (MBB por blocos contíguos).  
- **`block_length`**: tamanho do bloco no MBB (10–50 típico; ajuste pela dependência temporal).  
- **`aggregation`**: `mean` ou `median` nas agregações do ensemble.  
- **`center`**: `phi` (agregação pontual) ou `loo_quantile` (quantil das predições leave‑one‑out).  
- **`window_size`**: comprimento da janela deslizante dos resíduos (menor = mais reativo).  
- **`batch_size`**: frequência de atualização sequencial dos resíduos quando `y_true` é fornecido em `get_prediction`.

---

## API (essencial)

```python
from enbpi import EnbPIModel

m = EnbPIModel(base_model, B=30, alpha=0.1,
               aggregation="mean", center="loo_quantile",
               bootstrap="block", block_length=25,
               batch_size=1, window_size=None, random_state=123)

m.fit(X_train, y_train)

# Sem feedback (produção): usa resíduos já calibrados
res = m.get_prediction(X_future, y_true=None)

# Com feedback (val/teste): atualiza janela a cada batch
res = m.get_prediction(X_test, y_true=y_test)

# Acesso aos resultados
res.lower_bounds, res.upper_bounds, res.point_predictions
print(res.summary())
# res.plot_interval(show=True)
```

---

## Organização do código

- `enbpi/core.py` — **EnbPIModel / EnbPIResults** (OOB, quantis, janela deslizante, predição).  
- `enbpi/bootstrap.py` — **Moving Block Bootstrap** (`moving_block_bootstrap_indices`).  
- `enbpi/base.py` — classes base para API ao estilo `statsmodels`.  
- `enbpi/__init__.py` — exports.

> Observação: o modelo base deve expor `.fit(X, y)` e `.predict(X_new)`. Se também expuser `fit_with_indices(idx)`, o EnbPI repassará os índices de bootstrap (útil para estimadores que exigem contiguidade).

---

## Referências
- Xu, C. & Xie, Y. (2021). *Conformal Prediction Interval for Dynamic Time‑Series*. ICML.  
- Xu, Chen, and Yao Xie. "Conformal prediction for time series." IEEE transactions on pattern analysis and machine intelligence 45.10 (2023): 11575-11587. 
