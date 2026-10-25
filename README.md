# 论文阅读与代码复现笔记

这里记录我在应用统计、机器学习与论文复现过程中的学习内容。

## 当前精读论文

**Deep Generative Models for Reject Inference in Credit Scoring**

- [论文原文（arXiv）](https://arxiv.org/abs/1904.11376)
- [作者原始代码](https://github.com/rogelioamancisidor/reject_inference)
- [我的复现仓库](https://github.com/1258529809/reject_inference)
- [Model 1 单个完整 epoch 报告](https://github.com/1258529809/reject_inference/blob/3ece90db89705b7e5bbde8d981e6fbeb85adb82c/docs/model1-stage10-one-epoch.md)
- [GitHub Actions 运行记录](https://github.com/1258529809/reject_inference/actions/runs/39925544573)

---

## 1. 我对 Model 1 的基本理解

Model 1 同时使用有标签的 accepted 数据和无标签的 rejected 数据进行训练。

模型包含四个主要部分：

| 模型组成 | 数学表达式 | 作用 |
|---|---|---|
| 分类器 | $q_\phi(y\mid x)$ | 根据客户特征预测类别概率 |
| 编码器 | $q_\phi(z\mid x,y)$ | 估计潜变量的近似后验分布 |
| 类别条件高斯先验 | $p_\theta(z\mid y)$ | 描述不同类别在潜空间中的先验分布 |
| 解码器 | $p_\theta(x\mid z)$ | 根据潜变量描述客户特征的分布 |

模型的生成过程为：

$$
p_\theta(x,y,z)
=
p(y)p_\theta(z\mid y)p_\theta(x\mid z)
$$

推断过程为：

$$
q_\phi(y,z\mid x)
=
q_\phi(y\mid x)q_\phi(z\mid x,y)
$$

需要注意：

$$
q_\phi(z\mid x,y)
\approx
p_\theta(z\mid x,y)
$$

编码器给出的是近似变分后验，不是真实后验；而 $p_\theta(z\mid y)$ 是类别条件先验，也不是后验分布。

---

## 2. ELBO 与有标签损失

由于潜变量积分通常无法直接计算，模型使用 ELBO 近似联合对数似然：

$$
\log p_\theta(x,y)
=
\operatorname{ELBO}(x,y)
+
D_{\mathrm{KL}}
\left(
q_\phi(z\mid x,y)
\parallel
p_\theta(z\mid x,y)
\right)
$$

因为 KL 散度非负：

$$
\operatorname{ELBO}(x,y)
\leq
\log p_\theta(x,y)
$$

所以最大化 ELBO 等价于最小化负 ELBO。

有标签生成目标可以概括为：

$$
\operatorname{ELBO}_l(x,y)
=
\mathbb E_{q_\phi(z\mid x,y)}
[\log p_\theta(x\mid z)]
-
D_{\mathrm{KL}}
\left(
q_\phi(z\mid x,y)
\parallel
p_\theta(z\mid y)
\right)
+
\log p(y)
$$

作者还加入监督分类交叉熵：

$$
\mathcal L_{\mathrm{CE}}
=
-\sum_k y_k\log q_\phi(y=k\mid x)
$$

因此有标签训练损失为：

$$
\boxed{
\mathcal J_l
=
-\operatorname{ELBO}_l(x,y)
+
\alpha\mathcal L_{\mathrm{CE}}
}
$$

分类器不是由 ELBO 直接计算出来的，而是通过优化这个目标函数逐渐学习神经网络参数。

---

## 3. 无标签数据为什么能训练分类器？

对于 rejected 客户，模型不知道真实类别，但分类器仍可计算：

$$
q_\phi(y=0\mid x),
\qquad
q_\phi(y=1\mid x)
$$

作者分别假设 $y=0$ 和 $y=1$，计算两个类别下的生成损失，再按分类概率加权。

在最小化负 ELBO 的记号下：

$$
\boxed{
\mathcal J_u(x)
=
\sum_{y=0}^{1}
q_\phi(y\mid x)
\mathcal L_{\mathrm{gen}}(x,y)
-
H(q_\phi(y\mid x))
}
$$

其中：

$$
H(q)
=
-\sum_y q(y\mid x)\log q(y\mid x)
$$

因为 $q_\phi(y\mid x)$ 直接参与无标签目标，所以梯度可以传递到分类器参数：

$$
\frac{\partial\mathcal J_u}{\partial\phi}
$$

这说明 rejected 数据即使没有真实标签，也可以通过“哪个类别的生成模型更能解释该客户”向分类器提供训练信号。

但这不代表模型知道 rejected 客户的真实结果。它仍然依赖生成模型和潜变量分布假设是否合理。

---

## 4. 分类器与参数学习

### 4.1 分类器最终输出什么？

Model 1 最终用于信用评分的是一个分类器：

$
\boxed{q_\phi(y\mid x)}
$

其中：

- $x$：客户的观测特征；
- $y$：客户类别，例如是否违约；
- $\phi$：分类器的神经网络参数。

分类器通过 Softmax 同时输出两个类别的概率，而不是分别建立两个分类器：

$
q_\phi(y\mid x)
=
\begin{bmatrix}
q_\phi(y=0\mid x) &
q_\phi(y=1\mid x)
\end{bmatrix}
$

例如：

$
q_\phi(y\mid x)=[0.82,0.18]
$

如果 $y=1$ 表示违约，那么预测违约概率为 $q_\phi(y=1\mid x)=0.18$，预测类别则是：

$
\hat y=\arg\max_y q_\phi(y\mid x)
$

因此需要区分：Model 1 只有一个分类器、存在有标签和无标签两条训练路径，并输出两个类别概率。两条路径共同更新同一个分类器。

### 4.2 网络结构

在作者公开的 paper 数据配置下，分类器输入7个特征，包含一个有70个神经元的隐藏层：

$
\boxed{
x\in\mathbb R^7
\rightarrow
\text{全连接层}
\rightarrow
\operatorname{Softplus}
\rightarrow
\operatorname{Dropout}
\rightarrow
\text{全连接层}
\rightarrow
\operatorname{Softmax}
}
$

张量维度为：

$
(B,7)\rightarrow(B,70)\rightarrow(B,2)
$

其中 $B$ 是 minibatch 大小。前向计算为：

$
h=\operatorname{softplus}(xW_1+b_1)
$

$
q_\phi(y\mid x)
=
\operatorname{softmax}(hW_2+b_2)
$

作者公开配置的 Dropout 比例为 $0.01$。

### 4.3 分类器的可训练参数

| 参数 | 形状 | 作用 |
|---|---:|---|
| $W_1$ | $7\times70$ | 输入层到隐藏层的权重 |
| $b_1$ | $70$ | 隐藏层偏置 |
| $W_2$ | $70\times2$ | 隐藏层到输出层的权重 |
| $b_2$ | $2$ | 输出层偏置 |

参数标量总数为：

$
7\times70+70+70\times2+2=702
$

因此应当区分：

- 分类器数量：1个；
- 参数数组：4个；
- 参数标量：702个；
- 输出类别概率：2个。

Softplus、Softmax、Dropout 和损失函数参与计算，但它们本身没有需要训练的参数。

### 4.4 损失函数、反向传播和 Adam 的分工

损失函数不会直接求出最优参数，MLP 也不会脱离损失函数自动计算参数。完整学习过程为：

$
\boxed{
\text{前向计算}
\rightarrow
\text{计算损失}
\rightarrow
\text{反向传播计算梯度}
\rightarrow
\text{Adam更新参数}
}
$

| 环节 | 作用 |
|---|---|
| MLP 前向传播 | 使用当前参数计算类别概率 |
| 损失函数 | 衡量当前模型表现 |
| 反向传播 | 使用链式法则计算损失对参数的梯度 |
| Adam | 根据梯度修改参数 |

分类器的四个参数全部通过这一过程学习：

$
\mathcal L
\rightarrow
\left\{
\frac{\partial\mathcal L}{\partial W_1},
\frac{\partial\mathcal L}{\partial b_1},
\frac{\partial\mathcal L}{\partial W_2},
\frac{\partial\mathcal L}{\partial b_2}
\right\}
\rightarrow
\text{Adam更新}
$

反向传播只负责计算梯度，真正修改参数的是优化器。

### 4.5 两条训练路径如何更新同一个分类器？

有标签客户同时具有 $x$ 和真实标签 $y$。分类器通过交叉熵获得直接的监督信号：

$
\mathcal L_{\mathrm{CE}}
=
-\sum_k y_k\log q_\phi(y=k\mid x)
$

作者实现中的有标签生成项不依赖分类器输出，因此对分类器而言，有标签路径的直接梯度主要来自交叉熵。

被拒绝客户只有特征 $x$，没有真实标签，不能计算监督交叉熵。模型分别假设 $y=0$ 和 $y=1$，计算对应的生成损失，再由分类概率加权：

$
\mathcal J_u(x)
=
\sum_y
q_\phi(y\mid x)\mathcal L_{\mathrm{gen}}(x,y)
-
H\!\left(q_\phi(y\mid x)\right)
$

因为 $q_\phi(y\mid x)$ 出现在无标签目标中，所以梯度能够传回分类器的 $W_1,b_1,W_2,b_2$。这就是无标签 rejected 数据能够影响分类器训练的原因。

### 4.6 分类器与 VAE 其他组件的关系

训练阶段可以理解为两条相互联系的分支：

~~~text
                    ┌→ 分类器 q(y|x) → 类别概率
客户特征 x ─────────┤
                    └→ 编码器 q(z|x,y)
                              ↓
                         潜变量 z
                              ↓
                         解码器 p(x|z)
~~~

不能将模型理解为：

~~~text
x → 重构 x̂ → 预测 y
~~~

实际预测新客户时，分类器直接使用原始客户特征：

$
\boxed{x\rightarrow q_\phi(y\mid x)}
$

不需要先经过编码器或解码器。编码器、类别条件先验和解码器主要在训练阶段帮助模型利用有标签与无标签数据。

---

## 5. Adam 参数更新

训练过程可以概括为：

1. 输入一个 minibatch；
2. 完成模型前向计算；
3. 计算 labeled 或 unlabeled 损失；
4. 通过自动微分得到参数梯度；
5. 使用 Adam 更新参数；
6. 对后续 minibatch 重复以上过程。

Adam 维护梯度的一阶矩和二阶矩：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

参数更新为：

$$
\theta_t
=
\theta_{t-1}
-
\eta
\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

由于随机潜变量采样、minibatch 差异和随机优化，训练损失不保证每一步都下降。

本次源码核验还确认：

- Model 1 共有30个参数数组；
- labeled 和 unlabeled 使用同一组模型参数；
- 两条训练路径维护各自的 Adam 优化器状态；
- 分类器、编码器、解码器和条件高斯先验均能被更新。

---

## 6. 混淆矩阵

在信用评分中，暂时规定：

- 正类 $Y=1$：违约客户；
- 负类 $Y=0$：未违约客户。

|  | 真实违约 | 真实未违约 |
|---|---:|---:|
| 预测违约 | TP | FP |
| 预测未违约 | FN | TN |

- **TP**：真实违约，模型也预测违约；
- **FP**：真实未违约，但模型预测违约；
- **FN**：真实违约，但模型预测未违约；
- **TN**：真实未违约，模型也预测未违约。

真正率（召回率）：

$$
\boxed{
\mathrm{TPR}
=
\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}
}
$$

假正率：

$$
\boxed{
\mathrm{FPR}
=
\frac{\mathrm{FP}}{\mathrm{FP}+\mathrm{TN}}
}
$$

精确率：

$$
\mathrm{Precision}
=
\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}
$$

准确率：

$$
\mathrm{Accuracy}
=
\frac{\mathrm{TP}+\mathrm{TN}}
{\mathrm{TP}+\mathrm{FP}+\mathrm{FN}+\mathrm{TN}}
$$

F1 分数：

$$
F_1
=
\frac{2\mathrm{TP}}
{2\mathrm{TP}+\mathrm{FP}+\mathrm{FN}}
$$

---

## 7. ROC 与 AUC

在不同分类阈值 $c$ 下，可以分别计算：

$$
\mathrm{TPR}(c)
=
\frac{\mathrm{TP}(c)}
{\mathrm{TP}(c)+\mathrm{FN}(c)}
$$

$$
\mathrm{FPR}(c)
=
\frac{\mathrm{FP}(c)}
{\mathrm{FP}(c)+\mathrm{TN}(c)}
$$

ROC 曲线由所有阈值对应的点组成：

$$
\mathrm{ROC}
=
\{(\mathrm{FPR}(c),\mathrm{TPR}(c)):c\in\mathbb R\}
$$

AUC 是 ROC 曲线下面积：

$$
\boxed{
\mathrm{AUC}
=
\int_0^1 \mathrm{TPR}(u)\,du
}
$$

AUC 也可以解释为：

$$
\boxed{
\mathrm{AUC}
=
P(s(X^+)>s(X^-))
+
\frac12P(s(X^+)=s(X^-))
}
$$

即随机抽取一个真实违约客户和一个真实未违约客户时，模型把违约客户评为更高风险的概率。

AUC 衡量的是排序能力，不是分类准确率，也不直接衡量概率校准程度。

---

## 8. 一轮真实数据训练结果

在作者公开参数下，我完成了 Model 1 的一个完整 epoch：

| 项目 | 结果 |
|---|---:|
| Labeled 更新 | 31次 |
| Unlabeled 更新 | 309次 |
| 总参数更新 | 340次 |
| 参数数组 | 30个，全部发生变化 |
| NaN/Inf | 未发现 |
| 测试 AUC | 0.6202256 |

这个 AUC 可以解释为：

> 从 accepted 测试集中随机抽取一个真实违约客户和一个真实未违约客户，模型约有62.02%的概率给违约客户更高的风险分数。

但需要注意：

- AUC 来自有真实标签的 accepted 测试集；
- rejected 客户没有真实还款结果；
- accepted 测试集上的 AUC 不能直接证明 rejected 客户预测正确；
- 本次只训练一个 epoch，不能视为论文最终实验结果复现。

---

## 9. 当前学习结论

目前我已经完成：

- Model 1 概率结构与基本训练逻辑梳理；
- labeled 和 unlabeled 损失验证；
- 梯度与 Adam 更新验证；
- 真实数据安全读取；
- 一个完整 epoch 的真实数据训练；
- 原始 Beta calibration 与 AUC 评估。

当前最重要的理解是：

> Model 1 的核心不是 Adam，而是把未知类别视为离散潜变量，对两个类别进行精确枚举，并让分类概率参与无标签 ELBO，使 rejected 数据能够向分类器提供梯度。

本次工作完成了作者公开源码的训练链路复现，但尚未运行401 epochs、完整交叉验证或论文中的 VAE 预训练，因此不宣称已经复现论文最终指标。
