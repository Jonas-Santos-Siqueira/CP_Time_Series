# EnbPI (Python)

**Ensemble Bootstrap Prediction Intervals** (EnbPI) for **time‑series regression**, com API no estilo `statsmodels` e suporte a **Moving Block Bootstrap (MBB)** e **VAR do `statsmodels` via adaptador.

O método constrói **intervalos de predição conformais** livres de distribuição, usando um **ensemble bootstrap** de modelos base e **resíduos OOB (leave‑one‑out)** para calibrar os quantis. Em tempo de predição, os intervalos podem ser **atualizados sequencialmente** (janela deslizante) sem re‑treinar os modelos base.

---

## Como funciona (resumo)

Para uma série alvo $$y_t$$ com regressoras $$x_t$$:

1. Treinamos $$B$$ modelos em **amostras bootstrap** do conjunto de treino (i.i.d. ou por **blocos contíguos** para séries temporais).
2. Para cada $$i$$ no treino, agregamos as predições **dos modelos que não usaram $$i$$** (OOB) e computamos o **resíduo não‑conforme** \(\hat\varepsilon_i = \lvert y_i - \hat f_{-i}(x_i)\rvert\).
3. Para um novo ponto $$x_t$$, definimos o **centro** do intervalo como:
   - $$\hat f^{\phi}(x_t)$$ (agregação `mean`/`median`) ou
   - o **quantil $$(1-\alpha)$$** das predições leave‑one‑out $${\hat f_{-i}(x_t)\}_{i=1}^T$$ (`center="loo_quantile"`).
4. A meia‑largura $$w_t$$ é o **quantil $$(1-\alpha)$$** dos resíduos atuais (com **janela deslizante** opcional).  
5. Intervalo: $$[\,\text{centro} \pm w_t\,]$$.

---

## Instalação / Uso

1) Baixe e descompacte esta pasta para um local do seu `PYTHONPATH` (ou adicione o caminho via `sys.path.append`).  
2) Importe:
```python
from enbpi import EnbPIModel
```

### Arquivos principais

- `enbpi/core.py` — **EnbPIModel / EnbPIResults** (núcleo do método, janela deslizante, OOB, quantis).
- `enbpi/bootstrap.py` — **Moving Block Bootstrap** (`moving_block_bootstrap_indices`).
- `enbpi/wrappers.py` — **Wrappers**:
  - `StatsmodelsVARAdapter` — usa **VAR(p)** do `statsmodels` de forma **contígua por blocos** (VAR “puro”).
  - `StatsmodelsOLSRegressor` — OLS do `statsmodels` com API “sklearn‑like”.
- `enbpi/__init__.py` — exports.

---

## Quick start (univariado, qualquer regressor sklearn)

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

## Opção A — **VAR “puro”** (statsmodels) + EnbPI com **Moving Block Bootstrap** ✅

Use o adaptador **`StatsmodelsVARAdapter`** para plugar um `VAR(p)` “de verdade” no EnbPI.  
O adaptador recebe a série **contígua** `Y_full` (treino), a ordem `p` e `target_idx` (qual variável prever).  
A cada membro do ensemble, o EnbPI passa **índices de bootstrap por blocos sobre as linhas de X**; o adaptador os converte em blocos contíguos de **Y**, re‑estima o **VAR(p)** e prevê **1 passo** para o alvo.

```python
import numpy as np
from enbpi import EnbPIModel, StatsmodelsVARAdapter

# --- Y_train: (T x k) contíguo; construa X lagged e y (alvo) para target_idx ---
def build_lagged(Y, p, target_idx):
    n, k = Y.shape
    X, y = [], []
    for t in range(p, n):
        lags = [Y[t - lag] for lag in range(1, p+1)]
        X.append(np.concatenate(lags, axis=0))
        y.append(Y[t, target_idx])
    return np.asarray(X), np.asarray(y)

p = 2
target_idx = 0

# Suponha que você já tenha Y_train e Y_test_full (inclui p lags antes do 1º test)
X_train, y_train = build_lagged(Y_train, p, target_idx)
X_test,  y_test  = build_lagged(Y_test_full, p, target_idx)

base = StatsmodelsVARAdapter(Y_full=Y_train, p=p, target_idx=target_idx, trend='c')

model = EnbPIModel(
    base_model=base,
    B=40, alpha=0.1,
    aggregation="mean",
    center="loo_quantile",
    bootstrap="block",     # MBB contíguo
    block_length=25,       # ajuste conforme a dependência temporal
    batch_size=1,
    random_state=123,
)
model.fit(X_train, y_train)
res = model.get_prediction(X_test, y_true=y_test)
print(res.summary())
```

**Por que isso é “VAR puro”?** Cada réplica bootstrap re‑estima um `VAR(p)` do `statsmodels` em uma **série contígua por blocos** de `Y_train` (o adaptador adiciona `p` observações no início de cada bloco para garantir os lags). As predições 1‑passo usam os **coeficientes do VAR** (intercepto + \(A_i\)).

---

## Opção B — OLS do statsmodels por equação (wrapper)

Se preferir estimar a equação do alvo por OLS explicitamente:
```python
from enbpi import EnbPIModel, StatsmodelsOLSRegressor

base = StatsmodelsOLSRegressor(fit_intercept=True)
model = EnbPIModel(base, B=30, alpha=0.1,
                   aggregation="median", center="loo_quantile",
                   bootstrap="block", block_length=20, random_state=123)
```

---

## Dicas de modelagem & diagnóstico

- **`B` (nº de modelos):** 20–50 costuma ser suficiente.  
- **`bootstrap` & `block_length`:** use `bootstrap="block"` quando `X` deriva de lags; `block_length` entre 10–50 é um bom ponto de partida (ajuste via validação).  
- **`center`:** `loo_quantile` tende a ser mais **robusto** em séries com assimetria ou preditivos instáveis; `phi` (média/mediana) é o **centro pontual** clássico.  
- **`window_size`:** janela deslizante para resíduos; menor janela → **mais reatividade** (bom para drift), maior janela → **mais estabilidade**.  
- **Cobertura empírica:** verifique `results.summary()`; para inspeção visual, use `results.plot_interval()` e verifique pontos fora do intervalo em sequência (sinal de mudança de regime).

---

## Arquivos

- `enbpi/core.py` — EnbPI (treino, OOB, quantis, janela).  
- `enbpi/bootstrap.py` — MBB (índices por blocos contíguos).  
- `enbpi/wrappers.py` — `StatsmodelsVARAdapter`, `StatsmodelsOLSRegressor`.  
- `enbpi/__init__.py` — exports.

---

## Referências
- Xu, C. & Xie, Y. (2021). *Conformal Prediction Interval for Dynamic Time‑Series*. ICML.  
- Xu, Chen, and Yao Xie. "Conformal prediction for time series." IEEE transactions on pattern analysis and machine intelligence 45.10 (2023): 11575-11587. 

