# 第0章：線形代数の基礎 (NumPy対応表)

統計的機械学習の数理において、当たり前のように登場する基本操作と、その Python (NumPy) での実装をまとめます。

## 1. 転置 (Transpose)
行と列を入れ替える操作。ベクトルを横から縦に、あるいは縦から横にする際によく使われます。

- **数理記号**: $\mathbf{x}^T$, $A^T$
- **NumPy**: `x.T`, `A.T`

```python
import numpy as np
A = np.array([[1, 2], [3, 4]])
print(A.T) 
# [[1, 3],
#  [2, 4]]
```

## 2. ベクトルの大きさ（ノルム）
数式で $\|\mathbf{x}\|$ と出てきたら、それはベクトルの「強さ」や「原点からの距離」を測っています。
- 数理記号: $\|\mathbf{x}\|_2$
- NumPy: np.linalg.norm(x)3

## 3. ベクトルの重なり（内積）

$\mathbf{w}^T \mathbf{x}$ は、データ $\mathbf{x}$ に重み $\mathbf{w}$ を掛けて合計を出す、機械学習で最も頻出する計算です。
- 数理記号: $\mathbf{w}^T \mathbf{x}$
- NumPy: w.T @ x (Python 3.5以降の @ 演算子が便利)
  
```Python
w = np.array([0.5, 0.2, 0.1])
x = np.array([10, 20, 30])
prediction = w @ x  # 0.5*10 + 0.2*20 + 0.1*30 = 12.0
```
## 4. 逆行列 (Inverse Matrix)

「行列 $A$ を掛けて変わってしまったものを、元に戻す」ための行列。

- 数理記号: $A^{-1}$
- NumPy: np.linalg.inv(A)
```Python
A = np.array([[1, 2], [3, 4]])
A_inv = np.linalg.inv(A)

# A と A_inv を掛けると単位行列（1に相当）になる
identity = A @ A_inv 
# [[1, 0], [0, 1]] に近い値が出る
```

💡 学習メモ  
第0章の記号たちは「プログラミング言語の文法」のようなものです。「なぜこうなるか」の証明に深入りしすぎず、「この記号が来たらこの関数を呼ぶ」というマッピングを体に馴染ませることを優先します。
---

### 3. `.gitignore`
Jupyter Labを使用する際に発生する不要なファイルを管理対象から外します。リポジトリのルートに置いてください。

```text
# Jupyter Notebook checkpoints
.ipynb_checkpoints/
*/.ipynb_checkpoints/*

# Python caches
__pycache__/
*.py[cod]

# Virtual environments
venv/
.env/

# OS files
.DS_Store