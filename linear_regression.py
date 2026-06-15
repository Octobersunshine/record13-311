import numpy as np


def linear_regression(x: list | np.ndarray, y: list | np.ndarray) -> tuple[float, float] | str:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.shape != y.shape:
        return "x 和 y 形状不一致"
    if x.size < 2:
        return "数据点不足"

    n = x.size
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x ** 2)

    denominator = n * sum_x2 - sum_x ** 2
    if abs(denominator) < 1e-10:
        return "所有 x 值相同，无法计算斜率"

    slope = (n * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / n

    return float(slope), float(intercept)


if __name__ == "__main__":
    print("=== 测试1：正常数据点 ===")
    x_data = [1, 2, 3, 4, 5]
    y_data = [2, 4, 5, 4, 5]
    result = linear_regression(x_data, y_data)
    if isinstance(result, tuple):
        slope, intercept = result
        print(f"斜率: {slope:.4f}")
        print(f"截距: {intercept:.4f}")
        print(f"回归方程: y = {slope:.4f}x + {intercept:.4f}")
    else:
        print(f"提示: {result}")

    print()
    print("=== 测试2：只有一个数据点 ===")
    result = linear_regression([1], [2])
    if isinstance(result, tuple):
        slope, intercept = result
        print(f"斜率: {slope:.4f}, 截距: {intercept:.4f}")
    else:
        print(f"提示: {result}")

    print()
    print("=== 测试3：空数据 ===")
    result = linear_regression([], [])
    if isinstance(result, tuple):
        slope, intercept = result
        print(f"斜率: {slope:.4f}, 截距: {intercept:.4f}")
    else:
        print(f"提示: {result}")
