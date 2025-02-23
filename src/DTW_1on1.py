import pathlib

import matplotlib.pyplot as plt
import numpy as np
from natsort import natsorted
from pydantic import BaseModel

from DTW import dtw

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

    data1_num = 404
    data2_num = 500

    # get 2 data
    data1 = load_data(data_files[data1_num])
    data2 = load_data(data_files[data2_num])

    # x,y
    data1_xy = np.array([[d.x, d.y] for d in data1])

    # data2をすこしずらしてみる
    data2_xy = np.array([[d.x, d.y] for d in data2])
    data2_xy[:, 0] += 1

    # data2_xyのYを反転
    data2_xy = np.array([[-d.x, d.y] for d in data2])

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

    # 結果の可視化
    plt.figure(figsize=(15, 5))

    # 軌跡の比較プロット
    plt.subplot(131)
    plt.plot(data1_xy[:, 0], data1_xy[:, 1], "b-", label="Trajectory 1")
    plt.plot(data2_xy[:, 0], data2_xy[:, 1], "r-", label="Trajectory 2")
    plt.title("Trajectories Comparison")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")

    # 累積コスト行列とワーピングパス
    plt.subplot(132)
    plt.imshow(accumulated_cost, origin="lower", cmap="viridis", aspect="equal")
    plt.colorbar(label="Accumulated Cost")
    plt.plot(
        [0, min(accumulated_cost.shape)],
        [0, min(accumulated_cost.shape)],
        "k--",
        alpha=0.5,
        label="Diagonal",
    )
    plt.plot(path[:, 1], path[:, 0], "r-", linewidth=2, label="Warping Path")
    plt.title("Accumulated Cost Matrix & Warping Path")
    plt.xlabel("Trajectory 2 Index")
    plt.ylabel("Trajectory 1 Index")

    # マッチング結果
    plt.subplot(133)
    plt.plot(data1_xy[:, 0], data1_xy[:, 1], "b-", label="Trajectory 1")
    plt.plot(data2_xy[:, 0], data2_xy[:, 1], "r-", label="Trajectory 2")
    # いくつかの代表的なマッチングを表示
    for idx in range(0, len(path), 5):
        i, j = path[idx]
        plt.plot(
            [data1_xy[i, 0], data2_xy[j, 0]], [data1_xy[i, 1], data2_xy[j, 1]], "k-"
        )
    plt.title("Matching Results")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")

    plt.tight_layout()

    plt.savefig(f"output/1on1_{data1_num}_{data2_num}.png")
