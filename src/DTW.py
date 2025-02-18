import pathlib

import matplotlib.pyplot as plt
import numpy as np
from natsort import natsorted
from pydantic import BaseModel

plt.rcParams["font.size"] = 14
plt.rcParams["font.family"] = "TeX Gyre Termes"


class StoneData(BaseModel):
    x: float
    y: float
    v_x: float
    v_y: float
    omega: float


def load_data(data_file: pathlib.Path) -> list[StoneData]:
    data = []
    # utf-8 with BOM
    with open(data_file, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()

        for line in lines[1:]:
            x, y, v_x, v_y, omega = line.strip().split(",")
            data.append(
                StoneData(
                    x=float(x),
                    y=float(y),
                    v_x=float(v_x),
                    v_y=float(v_y),
                    omega=float(omega),
                )
            )

    return data


def dtw(
    data1: np.ndarray, data2: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray, np.floating, np.floating]:
    # コスト行列の初期化
    n = len(data1)
    m = len(data2)

    cost = np.zeros((n, m))

    # 各点の距離を計算
    for i in range(n):
        for j in range(m):
            cost[i, j] = np.linalg.norm(data1[i] - data2[j])

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
            if accumulated_cost[i - 1, j] == min(
                accumulated_cost[i - 1, j],
                accumulated_cost[i, j - 1],
                accumulated_cost[i - 1, j - 1],
            ):
                i -= 1
            elif accumulated_cost[i, j - 1] == min(
                accumulated_cost[i - 1, j],
                accumulated_cost[i, j - 1],
                accumulated_cost[i - 1, j - 1],
            ):
                j -= 1
            else:
                i -= 1
                j -= 1

        path.append((i, j))

    path.reverse()

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

    return distance, np.array(path), accumulated_cost, path_quality, path_smoothness


def test():
    # get 2 data
    data1 = load_data(data_files[404])
    data2 = load_data(data_files[500])

    # x,y
    data1_xy = np.array([[d.x, d.y] for d in data1])
    data2_xy = np.array([[d.x, d.y] for d in data2])

    # # サンプルデータとして、sin波の2次元座標を生成
    # # [x, y] = [x, sin(x)]

    # x = np.linspace(0, 6.28, num=100)
    # y = np.sin(x) + np.random.uniform(size=100) / 10.0
    # data1_xy = np.array([x, y]).T

    # # [x, y] = [x, cos(x)]
    # y = np.cos(x)
    # data2_xy = np.array([x, y]).T

    # # サンプル軌跡データの生成
    # # 軌跡1: 円形に近い形
    # t1 = np.linspace(0, 2 * np.pi, 50)
    # x1 = np.cos(t1) + np.random.normal(0, 0.1, 50)
    # y1 = np.sin(t1) + np.random.normal(0, 0.1, 50)
    # data1_xy = np.column_stack((x1, y1))

    # # 軌跡2: 同様の円形だが、ポイント数が異なり、若干歪んでいる
    # t2 = np.linspace(0, 2 * np.pi, 40)
    # x2 = 1.2 * np.cos(t2) + 0.2 * np.sin(2 * t2) + np.random.normal(0, 0.1, 40)
    # y2 = 1.1 * np.sin(t2) + 0.1 * np.cos(2 * t2) + np.random.normal(0, 0.1, 40)
    # data2_xy = np.column_stack((x2, y2))

    distance, path, accumulated_cost, path_quality, path_smoothness = dtw(
        data1_xy, data2_xy
    )

    # 評価指標の表示
    print(f"正規化DTW距離: {distance:.4f}")
    print(f"パスの品質（平均マッチング距離）: {path_quality:.4f}")
    print(f"パスの滑らかさ: {path_smoothness:.4f}")

    # # 結果の可視化
    # plt.figure(figsize=(15, 5))

    # # 軌跡の比較プロット
    # plt.subplot(131)
    # plt.plot(data1_xy[:, 0], data1_xy[:, 1], "b-", label="Trajectory 1")
    # plt.plot(data2_xy[:, 0], data2_xy[:, 1], "r-", label="Trajectory 2")
    # plt.title("Trajectories Comparison")
    # plt.legend()
    # plt.grid(True)
    # plt.axis("equal")

    # # 累積コスト行列とワーピングパス
    # plt.subplot(132)
    # plt.imshow(accumulated_cost, origin="lower", cmap="viridis", aspect="equal")
    # plt.colorbar(label="Accumulated Cost")
    # plt.plot(
    #     [0, min(accumulated_cost.shape)],
    #     [0, min(accumulated_cost.shape)],
    #     "k--",
    #     alpha=0.5,
    #     label="Diagonal",
    # )
    # plt.plot(path[:, 1], path[:, 0], "r-", linewidth=2, label="Warping Path")
    # plt.title("Accumulated Cost Matrix & Warping Path")
    # plt.xlabel("Trajectory 2 Index")
    # plt.ylabel("Trajectory 1 Index")

    # # マッチング結果
    # plt.subplot(133)
    # plt.plot(data1_xy[:, 0], data1_xy[:, 1], "b-", label="Trajectory 1")
    # plt.plot(data2_xy[:, 0], data2_xy[:, 1], "r-", label="Trajectory 2")
    # # いくつかの代表的なマッチングを表示
    # for idx in range(0, len(path), 5):
    #     i, j = path[idx]
    #     plt.plot(
    #         [data1_xy[i, 0], data2_xy[j, 0]], [data1_xy[i, 1], data2_xy[j, 1]], "k-"
    #     )
    # plt.title("Matching Results")
    # plt.legend()
    # plt.grid(True)
    # plt.axis("equal")

    # plt.tight_layout()

    # plt.savefig("output.png")


if __name__ == "__main__":
    root_dir = pathlib.Path(__file__).resolve().parents[1]
    # Load the data from the Dataset directory
    data_dir = root_dir / "Dataset"

    data_files = natsorted(list(data_dir.glob("*.csv")))

    data = [load_data(data_file) for data_file in data_files]
    trajectories = [np.array([[d.x, d.y] for d in trajectory]) for trajectory in data]

    n_datasets = len(data_files)
    print(f"Number of datasets: {n_datasets}")

    # 各指標の行列を初期化
    distances = np.zeros((n_datasets, n_datasets))
    path_qualities = np.zeros((n_datasets, n_datasets))
    path_smoothnesses = np.zeros((n_datasets, n_datasets))

    paths = {}  # 特徴的なペアのパスのみ保存する場合
    accumulated_costs = {}  # 特徴的なペアの累積コスト行列のみ保存する場合

    for i in range(n_datasets):
        for j in range(i + 1, n_datasets):  # 対称行列なので半分だけ計算
            data_1 = load_data(data_files[i])
            data_2 = load_data(data_files[j])

            distance, path, accumulated_cost, path_quality, path_smoothness = dtw(
                trajectories[i], trajectories[j]
            )
            # 対称行列として保存
            distances[i, j] = distances[j, i] = distance
            path_qualities[i, j] = path_qualities[j, i] = path_quality
            path_smoothnesses[i, j] = path_smoothnesses[j, i] = path_smoothness

            # # 特徴的なペアの場合のみパスと累積コスト行列を保存
            # if distance < threshold:  # または他の条件
            #     paths[(i, j)] = path
            #     accumulated_costs[(i, j)] = acc_cost

            # 進捗表示(100回に1回)
            if (i * n_datasets + j) % 100 == 0:
                print(f"Dataset {i + 1}/{n_datasets}, Pair {j + 1}/{n_datasets}")

    plt.figure(figsize=(15, 5))

    # DTW距離の分布
    plt.subplot(131)
    plt.hist(distances.flatten(), bins=50)
    plt.title("DTW Distance Distribution")

    # パス品質の分布
    plt.subplot(132)
    plt.hist(path_qualities.flatten(), bins=50)
    plt.title("Path Quality Distribution")

    # パスの滑らかさの分布
    plt.subplot(133)
    plt.hist(path_smoothnesses.flatten(), bins=50)
    plt.title("Path Smoothness Distribution")

    plt.tight_layout()
    plt.savefig("output.png")

    for name, data in [
        ("DTW Distance", distances),
        ("Path Quality", path_qualities),
        ("Path Smoothness", path_smoothnesses),
    ]:
        print(f"\n{name}:")
        print(f"Mean: {np.mean(data):.4f}")
        print(f"Std: {np.std(data):.4f}")
        print(f"Min: {np.min(data):.4f}")
        print(f"Max: {np.max(data):.4f}")
