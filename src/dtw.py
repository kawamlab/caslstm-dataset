import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Union


def dtw(
    data1: NDArray, data2: NDArray
) -> Tuple[
    float, NDArray, NDArray, Union[np.floating, float], Union[np.floating, float]
]:
    """
    Dynamic Time Warping implementation for trajectory comparison

    Args:
        data1: First trajectory data (N x 2 array)
        data2: Second trajectory data (M x 2 array)

    Returns:
        distance: Normalized DTW distance
        path: Optimal warping path
        accumulated_cost: Accumulated cost matrix
        path_quality: Mean matching distance along the path
        path_smoothness: Mean difference between consecutive matching distances
    """
    # コスト行列の初期化
    n = len(data1)
    m = len(data2)

    cost = np.zeros((n, m))

    # 各点の距離を計算
    # for i in range(n):
    #     for j in range(m):
    #         cost[i, j] = np.linalg.norm(data1[i] - data2[j])
    diff = data1[:, np.newaxis] - data2  # (n, m, 2)の配列
    cost = np.linalg.norm(diff, axis=2)  # (n, m)の配列

    # 累積コスト行列の初期化
    accumulated_cost = np.full((n, m), np.inf)
    accumulated_cost[0, 0] = cost[0, 0]

    # 累積コストの計算
    for i in range(1, n):
        accumulated_cost[i, 0] = accumulated_cost[i - 1, 0] + cost[i, 0]

    for j in range(1, m):
        accumulated_cost[0, j] = accumulated_cost[0, j - 1] + cost[0, j]

    for i in range(1, n):
        for j in range(1, m):
            accumulated_cost[i, j] = cost[i, j] + min(
                accumulated_cost[i - 1, j],
                accumulated_cost[i, j - 1],
                accumulated_cost[i - 1, j - 1],
            )

    # バックトレースで最適パスを求める
    i = n - 1
    j = m - 1
    path = [(i, j)]

    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            min_cost = min(
                accumulated_cost[i - 1, j],
                accumulated_cost[i, j - 1],
                accumulated_cost[i - 1, j - 1],
            )
            if accumulated_cost[i - 1, j] == min_cost:
                i -= 1
            elif accumulated_cost[i, j - 1] == min_cost:
                j -= 1
            else:
                i -= 1
                j -= 1
        path.append((i, j))

    path.reverse()
    path = np.array(path)

    # distanceを正規化
    distance = accumulated_cost[-1, -1] / (n + m)

    # マッチングの局所的な質を評価
    path_quality = np.mean([np.linalg.norm(data1[i] - data2[j]) for i, j in path])

    # 連続するマッチング間の滑らかさ
    path_smoothness = np.mean(
        [
            abs(
                np.linalg.norm(data1[i1] - data2[j1])
                - np.linalg.norm(data1[i2] - data2[j2])
            )
            for (i1, j1), (i2, j2) in zip(path[:-1], path[1:])
        ]
    )

    return distance, path, accumulated_cost, path_quality, path_smoothness
