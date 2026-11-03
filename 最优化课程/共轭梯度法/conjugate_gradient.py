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


# 新增块：最速下降法（精确线搜索）
def steepest_descent(A, b, x0, tol=1e-4, max_iter=10000):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    x = np.array(x0, dtype=float, copy=True)
    if b.ndim != 1 or x.shape != b.shape or A.shape != (b.size, b.size):
        raise ValueError("A、b、x0 维度不匹配")
    if not all(np.isfinite(v).all() for v in (A, b, x)):
        raise ValueError("输入必须为有限数值")
    if not np.allclose(A, A.T):
        raise ValueError("A 必须对称")
    np.linalg.cholesky(A)
    if tol <= 0:
        raise ValueError("tol 必须为正数")
    for k in range(max_iter + 1):
        g = A @ x - b
        print(f"k={k:2d}, x={x}, ||g||={np.linalg.norm(g):.6e}")
        if np.linalg.norm(g) <= tol:
            return x, k
        if k == max_iter:
            break
        # 每次都沿当前负梯度，不保留上一轮搜索方向。
        alpha = (g @ g) / (g @ (A @ g))
        x = x - alpha * g
    raise RuntimeError("最速下降法达到最大迭代次数，尚未满足精度要求")


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

    # 新增块：相同初始点、精度和最大次数下对比两种方法。
    print("\n========== 最速下降法与共轭梯度法对比 ==========")
    tol = 1e-4
    summary = []
    for i, (A, b, x0) in enumerate(examples, 1):
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)
        reference = np.linalg.solve(A, b)
        for name, solver in [("最速下降法", steepest_descent),
                             ("共轭梯度法", conjugate_gradient)]:
            print(f"\n例 {i}：{name}")
            x, steps = solver(A, b, x0, tol=tol, max_iter=10000)
            gradient_norm = np.linalg.norm(A @ x - b)
            error = np.linalg.norm(x - reference)
            value = 0.5 * x @ A @ x - b @ x
            assert gradient_norm <= tol
            summary.append((i, name, steps, gradient_norm, error, value))
    print("\n例题 | 方法 | 迭代次数 | 梯度范数 | 解误差 | 目标函数值")
    for i, name, steps, norm, error, value in summary:
        print(f"{i} | {name} | {steps} | {norm:.3e} | {error:.3e} | {value:.8f}")
