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

## 4. Adam 参数更新

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

## 5. 混淆矩阵

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

## 6. ROC 与 AUC

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

## 7. 一轮真实数据训练结果

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

## 8. 当前学习结论

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
