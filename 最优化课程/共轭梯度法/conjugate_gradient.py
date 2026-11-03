"""线性共轭梯度法：最小化 0.5*x.T@A@x - b.T@x。"""
import numpy as np


def conjugate_gradient(A, b, x0, tol=1e-4, max_iter=None):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    x = np.array(x0, dtype=float, copy=True)
    if b.ndim != 1 or x.shape != b.shape or A.shape != (b.size, b.size):
        raise ValueError("A 必须为 n×n 矩阵，b 和 x0 必须为长度 n 的向量")
    if not all(np.isfinite(v).all() for v in (A, b, x)):
        raise ValueError("输入必须为有限数值")
    if not np.allclose(A, A.T):
        raise ValueError("A 必须对称")
    np.linalg.cholesky(A)  # 检查正定性
    if tol <= 0:
        raise ValueError("tol 必须为正数")
    if max_iter is None:
        max_iter = 10 * b.size
    g = A @ x - b
    d = -g
    print(f"k={0:2d}, x={x}, ||g||={np.linalg.norm(g):.6e}")
    for k in range(max_iter):
        if np.linalg.norm(g) <= tol:
            return x, k
        Ad = A @ d
        alpha = (g @ g) / (d @ Ad)
        x = x + alpha * d
        g_new = g + alpha * Ad
        print(f"k={k+1:2d}, x={x}, ||g||={np.linalg.norm(g_new):.6e}, alpha={alpha:.8f}")
        # 用实际梯度确认终止，避免递推误差影响判断。
        if np.linalg.norm(A @ x - b) <= tol:
            return x, k + 1
        beta = (g_new @ g_new) / (g @ g)
        d = -g_new + beta * d
        g = g_new
    raise RuntimeError("达到最大迭代次数，尚未满足精度要求")


if __name__ == "__main__":
    examples = [
        ([[4, 1], [1, 3]], [1, 2], [0, 0]),
        ([[2, -1, 0], [-1, 2, -1], [0, -1, 2]], [1, 0, 1], [0, 0, 0]),
        ([[1, 0], [0, 2]], [3, 4], [1, 1]),
    ]
    np.set_printoptions(precision=8, suppress=True)
    for i, (A, b, x0) in enumerate(examples, 1):
        print(f"\n例 {i}")
        x, steps = conjugate_gradient(A, b, x0)
        reference = np.linalg.solve(A, b)
        assert np.allclose(x, reference, atol=1e-8)
        print(f"结果：{x}；迭代次数：{steps}")
        print(f"直接解对照：{reference}")
