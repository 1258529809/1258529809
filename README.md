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

```math
p_\theta(x,y,z)
=
p(y)p_\theta(z\mid y)p_\theta(x\mid z)
```

推断过程为：

```math
q_\phi(y,z\mid x)
=
q_\phi(y\mid x)q_\phi(z\mid x,y)
```

需要注意：

```math
q_\phi(z\mid x,y)
\approx
p_\theta(z\mid x,y)
```

编码器给出的是近似变分后验，不是真实后验；而 $p_\theta(z\mid y)$ 是类别条件先验，也不是后验分布。

---

## 2. ELBO 与有标签损失

由于潜变量积分通常无法直接计算，模型使用 ELBO 近似联合对数似然：

```math
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
```

因为 KL 散度非负：

```math
\operatorname{ELBO}(x,y)
\leq
\log p_\theta(x,y)
```

所以最大化 ELBO 等价于最小化负 ELBO。

有标签生成目标可以概括为：

```math
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
```

作者还加入监督分类交叉熵：

```math
\mathcal L_{\mathrm{CE}}
=
-\sum_k y_k\log q_\phi(y=k\mid x)
```

因此有标签训练损失为：

```math
\boxed{
\mathcal J_l
=
-\operatorname{ELBO}_l(x,y)
+
\alpha\mathcal L_{\mathrm{CE}}
}
```

分类器不是由 ELBO 直接计算出来的，而是通过优化这个目标函数逐渐学习神经网络参数。

---

## 3. 无标签数据为什么能训练分类器？

对于 rejected 客户，模型不知道真实类别，但分类器仍可计算：

```math
q_\phi(y=0\mid x),
\qquad
q_\phi(y=1\mid x)
```

作者分别假设 $y=0$ 和 $y=1$，计算两个类别下的生成损失，再按分类概率加权。

在最小化负 ELBO 的记号下：

```math
\boxed{
\mathcal J_u(x)
=
\sum_{y=0}^{1}
q_\phi(y\mid x)
\mathcal L_{\mathrm{gen}}(x,y)
-
H(q_\phi(y\mid x))
}
```

其中：

```math
H(q)
=
-\sum_y q(y\mid x)\log q(y\mid x)
```

因为 $q_\phi(y\mid x)$ 直接参与无标签目标，所以梯度可以传递到分类器参数：

```math
\frac{\partial\mathcal J_u}{\partial\phi}
```

这说明 rejected 数据即使没有真实标签，也可以通过“哪个类别的生成模型更能解释该客户”向分类器提供训练信号。

但这不代表模型知道 rejected 客户的真实结果。它仍然依赖生成模型和潜变量分布假设是否合理。

---

## 4. 分类器与参数学习

### 4.1 先回答最容易混淆的问题

> **分类器的参数不是分成“两部分”来求的。**
>
> `W₁、b₁、W₂、b₂` 全部由损失函数提供学习方向，反向传播计算梯度，最后由 Adam 修改数值。

四者的分工是：

| 环节 | 做什么 | 会不会自己产生分类器参数？ |
|---|---|---|
| MLP 前向传播 | 使用当前参数计算预测概率 | 不会，只使用参数 |
| 损失函数 | 衡量当前预测有多差 | 不会，只给出一个损失值 |
| 反向传播 | 计算损失对每个参数的偏导数 | 不会，只计算梯度 |
| Adam | 根据梯度更新参数 | 会修改参数数值 |

所以正确的关系是：

```text
损失函数定义目标
        ↓
反向传播计算四个梯度
        ↓
Adam 更新 W₁、b₁、W₂、b₂
```

而不是“损失函数求一部分参数，MLP 再求另一部分参数”。

### 4.2 分类器本身是什么？

Model 1 只有一个分类器：`qφ(y|x)`。它直接接收客户特征 `x`，同时输出两个类别的概率。

```text
客户特征 x（7维）
       ↓  使用 W₁、b₁
隐藏层 h（70维，Softplus + Dropout）
       ↓  使用 W₂、b₂
Softmax 输出 π（2维）
       ↓
π = [P(y=0|x), P(y=1|x)]
```

前向计算可以写成：

```math
h = \operatorname{softplus}(xW_1+b_1)
```

```math
\pi=q_\phi(y\mid x)=\operatorname{softmax}(hW_2+b_2)
```

假设输出 `π = [0.82, 0.18]`，并且 `y=1` 表示违约，那么模型给出的违约概率就是 `0.18`，预测类别为概率更大的 `y=0`。

### 4.3 分类器有哪些可训练参数？

| 参数 | 形状 | 作用 |
|---|---:|---|
| `W₁` | `7 × 70` | 将7个输入特征连接到70个隐藏单元 |
| `b₁` | `70` | 隐藏层偏置 |
| `W₂` | `70 × 2` | 将70个隐藏单元连接到两个类别得分 |
| `b₂` | `2` | 输出层偏置 |

分类器共有4个参数数组、702个参数标量：

```text
7 × 70 + 70 + 70 × 2 + 2 = 702
```

作者代码先随机初始化两个权重矩阵，并把两个偏置向量初始化为0。训练不是“从公式一次算出最优参数”，而是反复进行小幅更新。

Softplus、Softmax、Dropout 和交叉熵都参与计算，但它们本身没有分类器需要学习的权重。

### 4.4 一次有标签更新到底发生什么？

假设输入一个带真实标签的 minibatch：

1. **前向传播**：用当前的 `W₁、b₁、W₂、b₂` 得到 `qφ(y|x)`。
2. **计算分类损失**：把预测概率与真实标签比较。
3. **反向传播**：Theano 自动计算4个梯度。
4. **Adam 更新**：利用梯度修改4个参数数组。

交叉熵为：

```math
\mathcal L_{\mathrm{CE}}=-\sum_k y_k\log q_\phi(y=k\mid x)
```

需要计算的梯度是：

```math
\frac{\partial\mathcal L}{\partial W_1},\quad \frac{\partial\mathcal L}{\partial b_1},\quad \frac{\partial\mathcal L}{\partial W_2},\quad \frac{\partial\mathcal L}{\partial b_2}
```

梯度回答的是：“每个参数稍微改变时，损失会怎样变化？”Adam 再根据这个方向和大小更新参数。

这里必须把参数分成两组来看：

| 参数组 | 数量 | 在有标签损失中，梯度主要来自哪里？ |
|---|---:|---|
| 分类器：`W₁、b₁、W₂、b₂` | 4个数组 | 监督交叉熵 `α × L_CE` |
| 编码器、解码器、条件先验 | 26个数组 | 有标签生成损失，也就是负 ELBO 部分 |

因此，有标签总损失会更新整个 Model 1，但对分类器这4个参数而言，直接训练信号来自交叉熵；MLP 本身没有另一套“自己算参数”的机制。

### 4.5 无标签数据为什么也能更新这个分类器？

rejected 客户没有真实标签，不能直接计算监督交叉熵。分类器先给出：

```text
qφ(y=0|x) 和 qφ(y=1|x)
```

模型再分别假设 `y=0` 与 `y=1`，计算两个类别下的生成损失，并用分类概率加权：

```math
\mathcal J_u(x)=\sum_y q_\phi(y\mid x)\mathcal L_{\mathrm{gen}}(x,y)-H\!\left(q_\phi(y\mid x)\right)
```

因为 `qφ(y|x)` 直接出现在无标签损失中，所以该损失同样能对 `W₁、b₁、W₂、b₂` 求梯度。于是：

| 参数组 | 数量 | 在无标签损失中，梯度从哪里来？ |
|---|---:|---|
| 分类器 | 4个数组 | 分类概率加权和熵项 |
| 编码器、解码器、条件先验 | 26个数组 | 每个假设类别对应的生成损失 |

于是：

- 有标签路径通过交叉熵直接训练分类器；
- 无标签路径通过概率加权的生成损失和熵项间接训练分类器；
- 两条路径更新的是同一个分类器，不是两个分类器。

### 4.6 分类器参数与整个 Model 1 参数的关系

Model 1 一共有30个参数数组：

| 模块 | 参数数组数量 |
|---|---:|
| 分类器 | 4 |
| 编码器 | 8 |
| 解码器 | 8 |
| 类别条件高斯先验 | 10 |
| **合计** | **30** |

有标签和无标签训练路径共享这30个模型参数，但作者实现为两条路径分别维护 Adam 的内部状态。

### 4.7 训练结束后怎么预测？

训练阶段需要分类器、编码器、类别条件先验和解码器共同构建目标函数；但对新客户进行信用评分时，直接使用分类器：

```text
新客户特征 x → 分类器 qφ(y|x) → 违约概率和预测类别
```

不需要先把 `x` 重构成 `x̂`，也不是 `x → x̂ → ŷ`。

相关作者源码：

- [分类器在 SS_M1 中的实例化](https://github.com/1258529809/reject_inference/blob/4360be4136c96f090c9b6bf04abaee6e9f4c851f/python/model1/SS_M1.py#L60-L61)
- [CLS 网络结构与 Softmax 输出](https://github.com/1258529809/reject_inference/blob/4360be4136c96f090c9b6bf04abaee6e9f4c851f/python/model1/dropout_layers.py#L220-L260)

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

```math
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
```

```math
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
```

参数更新为：

```math
\theta_t
=
\theta_{t-1}
-
\eta
\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
```

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

```math
\boxed{
\mathrm{TPR}
=
\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}
}
```

假正率：

```math
\boxed{
\mathrm{FPR}
=
\frac{\mathrm{FP}}{\mathrm{FP}+\mathrm{TN}}
}
```

精确率：

```math
\mathrm{Precision}
=
\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}
```

准确率：

```math
\mathrm{Accuracy}
=
\frac{\mathrm{TP}+\mathrm{TN}}
{\mathrm{TP}+\mathrm{FP}+\mathrm{FN}+\mathrm{TN}}
```

F1 分数：

```math
F_1
=
\frac{2\mathrm{TP}}
{2\mathrm{TP}+\mathrm{FP}+\mathrm{FN}}
```

---

## 7. ROC 与 AUC

在不同分类阈值 $c$ 下，可以分别计算：

```math
\mathrm{TPR}(c)
=
\frac{\mathrm{TP}(c)}
{\mathrm{TP}(c)+\mathrm{FN}(c)}
```

```math
\mathrm{FPR}(c)
=
\frac{\mathrm{FP}(c)}
{\mathrm{FP}(c)+\mathrm{TN}(c)}
```

ROC 曲线由所有阈值对应的点组成：

```math
\mathrm{ROC}
=
\{(\mathrm{FPR}(c),\mathrm{TPR}(c)):c\in\mathbb R\}
```

AUC 是 ROC 曲线下面积：

```math
\boxed{
\mathrm{AUC}
=
\int_0^1 \mathrm{TPR}(u)\,du
}
```

AUC 也可以解释为：

```math
\boxed{
\mathrm{AUC}
=
P(s(X^+)>s(X^-))
+
\frac12P(s(X^+)=s(X^-))
}
```

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
