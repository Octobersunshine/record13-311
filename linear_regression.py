import numpy as np


def linear_regression(x: list | np.ndarray, y: list | np.ndarray) -> tuple[float, float, float] | str:
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

    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if abs(ss_tot) > 1e-10 else 1.0

    return float(slope), float(intercept), float(r_squared)


def predict(x: float | list | np.ndarray, slope: float, intercept: float) -> float | list[float]:
    if isinstance(x, (list, np.ndarray)):
        x_arr = np.asarray(x, dtype=np.float64)
        return [float(v) for v in (slope * x_arr + intercept)]
    return float(slope * float(x) + intercept)


if __name__ == "__main__":
    print("=== 测试1：正常数据点 ===")
    x_data = [1, 2, 3, 4, 5]
    y_data = [2, 4, 5, 4, 5]
    result = linear_regression(x_data, y_data)
    if isinstance(result, tuple):
        slope, intercept, r_squared = result
        print(f"斜率: {slope:.4f}")
        print(f"截距: {intercept:.4f}")
        print(f"R² 决定系数: {r_squared:.4f}")
        print(f"回归方程: y = {slope:.4f}x + {intercept:.4f}")
        print()
        print("--- 预测测试 ---")
        x_new = 6
        y_pred = predict(x_new, slope, intercept)
        print(f"预测 x={x_new} 时, y ≈ {y_pred:.4f}")
        x_new_list = [6, 7, 8]
        y_pred_list = predict(x_new_list, slope, intercept)
        print(f"预测 x={x_new_list} 时, y ≈ {[round(v, 4) for v in y_pred_list]}")
    else:
        print(f"提示: {result}")

    print()
    print("=== 测试2：只有一个数据点 ===")
    result = linear_regression([1], [2])
    if isinstance(result, tuple):
        slope, intercept, r_squared = result
        print(f"斜率: {slope:.4f}, 截距: {intercept:.4f}, R²: {r_squared:.4f}")
    else:
        print(f"提示: {result}")

    print()
    print("=== 测试3：空数据 ===")
    result = linear_regression([], [])
    if isinstance(result, tuple):
        slope, intercept, r_squared = result
        print(f"斜率: {slope:.4f}, 截距: {intercept:.4f}, R²: {r_squared:.4f}")
    else:
        print(f"提示: {result}")
