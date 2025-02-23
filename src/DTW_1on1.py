import pathlib
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from natsort import natsorted
from pydantic import BaseModel
from scipy.interpolate import interp1d

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


def prepare_time_series(data1: list[StoneData], data2: list[StoneData]):
    """
    異なる長さの時系列データを処理する
    """
    # データ長を取得
    len1 = len(data1)
    len2 = len(data2)

    # 両方のデータの時間軸を作成
    t1 = np.arange(len1) * 0.2
    t2 = np.arange(len2) * 0.2

    # 短い方の終了時刻まで揃える
    t_end = min(t1[-1], t2[-1])
    t_common = np.arange(0, t_end + 0.2, 0.2)

    # データ1の補間関数を作成
    v_x1_interp = interp1d(t1, [d.v_x for d in data1], kind="linear")
    v_y1_interp = interp1d(t1, [d.v_y for d in data1], kind="linear")
    omega1_interp = interp1d(t1, [d.omega for d in data1], kind="linear")

    # データ2の補間関数を作成
    v_x2_interp = interp1d(t2, [d.v_x for d in data2], kind="linear")
    v_y2_interp = interp1d(t2, [d.v_y for d in data2], kind="linear")
    omega2_interp = interp1d(t2, [d.omega for d in data2], kind="linear")

    return {
        "time": t_common,
        "v_x1": v_x1_interp(t_common),
        "v_y1": v_y1_interp(t_common),
        "omega1": omega1_interp(t_common),
        "v_x2": v_x2_interp(t_common),
        "v_y2": v_y2_interp(t_common),
        "omega2": omega2_interp(t_common),
    }


def plot_comprehensive_analysis(
    data1_num: int, data2_num: int, data_files: list, db_path: str
):
    """
    総合的な分析結果を1枚の画像にまとめる
    """
    # データの読み込み
    data1 = load_data(data_files[data1_num])
    data2 = load_data(data_files[data2_num])

    # 軌跡データの準備
    data1_xy = np.array([[d.x, d.y] for d in data1])
    data2_xy = np.array([[d.x, d.y] for d in data2])

    # DTW計算
    distance, path, accumulated_cost, path_quality, path_smoothness = dtw(
        data1_xy, data2_xy
    )

    # 時系列データの準備
    ts_data = prepare_time_series(data1, data2)

    # データベースからの追加情報取得
    conn = sqlite3.connect(db_path)

    # data1_numに関連する全ての距離データ
    query1 = f"""
    SELECT data_num2 as other_seq, distance, path_quality, path_smoothness
    FROM results WHERE data_num1 = {data1_num}
    UNION
    SELECT data_num1 as other_seq, distance, path_quality, path_smoothness
    FROM results WHERE data_num2 = {data1_num}
    """
    df1 = pd.read_sql_query(query1, conn)

    # グラフ設定
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig)

    # 軌跡比較プロット（X-Y軸入れ替え）
    ax_traj = fig.add_subplot(gs[0, 0])
    ax_traj.plot(data1_xy[:, 1], data1_xy[:, 0], "b-", label=f"Seq {data1_num}")
    ax_traj.plot(data2_xy[:, 1], data2_xy[:, 0], "r-", label=f"Seq {data2_num}")
    ax_traj.set_title("Trajectories Comparison")
    ax_traj.set_xlabel("y [m]")
    ax_traj.set_ylabel("x [m]")
    ax_traj.legend()
    ax_traj.grid(True)
    ax_traj.axis("equal")

    # 累積コスト行列とワーピングパス（同じ）
    ax_cost = fig.add_subplot(gs[0, 1])
    im = ax_cost.imshow(
        accumulated_cost, origin="lower", cmap="viridis", aspect="equal"
    )
    plt.colorbar(im, ax=ax_cost, label="Accumulated Cost")
    ax_cost.plot(
        [0, min(accumulated_cost.shape)],
        [0, min(accumulated_cost.shape)],
        "k--",
        alpha=0.5,
        label="Diagonal",
    )
    ax_cost.plot(path[:, 1], path[:, 0], "r-", linewidth=2, label="Warping Path")
    ax_cost.set_title("Accumulated Cost & Warping Path")
    ax_cost.set_xlabel("Trajectory 2 Index")
    ax_cost.set_ylabel("Trajectory 1 Index")

    # マッチング結果（X-Y軸入れ替え）
    ax_match = fig.add_subplot(gs[0, 2])
    ax_match.plot(data1_xy[:, 1], data1_xy[:, 0], "b-", label=f"Seq {data1_num}")
    ax_match.plot(data2_xy[:, 1], data2_xy[:, 0], "r-", label=f"Seq {data2_num}")
    for idx in range(0, len(path), 5):
        i, j = path[idx]
        ax_match.plot(
            [data1_xy[i, 1], data2_xy[j, 1]],
            [data1_xy[i, 0], data2_xy[j, 0]],
            "k-",
            alpha=0.3,
        )
    ax_match.set_title("Matching Results")
    ax_match.set_xlabel("y [m]")
    ax_match.set_ylabel("x [m]")
    ax_match.legend()
    ax_match.grid(True)
    ax_match.axis("equal")

    # Y軸の範囲を調整（必要に応じて）
    y_min = min(np.min(data1_xy[:, 0]), np.min(data2_xy[:, 0]))
    y_max = max(np.max(data1_xy[:, 0]), np.max(data2_xy[:, 0]))
    y_range = y_max - y_min
    ax_traj.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)
    ax_match.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)

    # 速度プロフィール
    ax_vel = fig.add_subplot(gs[1, 0])
    ax_vel.plot(ts_data["time"], ts_data["v_x1"], "b-", label=f"vx {data1_num}")
    ax_vel.plot(ts_data["time"], ts_data["v_y1"], "b--", label=f"vy {data1_num}")
    ax_vel.plot(ts_data["time"], ts_data["v_x2"], "r-", label=f"vx {data2_num}")
    ax_vel.plot(ts_data["time"], ts_data["v_y2"], "r--", label=f"vy {data2_num}")
    ax_vel.set_title("Velocity Profiles")
    ax_vel.set_xlabel("Time [s]")
    ax_vel.set_ylabel("Velocity [m/s]")
    ax_vel.legend()
    ax_vel.grid(True)

    # 角速度プロフィール
    ax_omega = fig.add_subplot(gs[1, 1])
    ax_omega.plot(ts_data["time"], ts_data["omega1"], "b-", label=f"Seq {data1_num}")
    ax_omega.plot(ts_data["time"], ts_data["omega2"], "r-", label=f"Seq {data2_num}")
    ax_omega.set_title("Angular Velocity Profiles")
    ax_omega.set_xlabel("Time [s]")
    ax_omega.set_ylabel("Angular Velocity [rad/s]")
    ax_omega.legend()
    ax_omega.grid(True)

    # 距離分布
    ax_dist = fig.add_subplot(gs[1, 2])
    ax_dist.hist(
        df1["distance"], bins=30, alpha=0.5, label=f"Seq {data1_num} distances"
    )
    ax_dist.axvline(
        distance, color="r", linestyle="--", label=f"Current distance: {distance:.3f}"
    )
    ax_dist.set_title("Distance Distribution")
    ax_dist.set_xlabel("DTW Distance")
    ax_dist.set_ylabel("Count")
    ax_dist.legend()
    ax_dist.grid(True)

    # データ長情報の追加
    length_info = (
        f"Data Lengths:\n"
        f"Sequence {data1_num}: {len(data1)} points ({len(data1) * 0.2:.1f}s)\n"
        f"Sequence {data2_num}: {len(data2)} points ({len(data2) * 0.2:.1f}s)"
    )

    # 評価指標の表
    ax_metrics = fig.add_subplot(gs[2, :])
    ax_metrics.axis("off")
    metrics_text = (
        f"Comparison Metrics for Sequences {data1_num} and {data2_num}:\n\n"
        f"Normalized DTW Distance: {distance:.4f}\n"
        f"Path Quality (Mean Matching Distance): {path_quality:.4f}\n"
        f"Path Smoothness: {path_smoothness:.4f}\n\n"
        f"Statistics for Sequence {data1_num}:\n"
        f"Mean Distance: {df1['distance'].mean():.4f}\n"
        f"Min Distance: {df1['distance'].min():.4f}\n"
        f"Max Distance: {df1['distance'].max():.4f}\n"
        f"Distance Std Dev: {df1['distance'].std():.4f}\n\n"
        f"{length_info}"
    )
    ax_metrics.text(0.1, 0.5, metrics_text, fontsize=12, fontfamily="monospace")

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    root_dir = pathlib.Path(__file__).resolve().parents[1]
    data_dir = root_dir / "Dataset"
    data_files = natsorted(list(data_dir.glob("*.csv")))

    # 比較するシーケンス番号
    data1_num = 404
    data2_num = 500

    # 総合的な分析の実行と保存
    fig = plot_comprehensive_analysis(data1_num, data2_num, data_files, "results.db")
    fig.savefig(
        root_dir / "output" / f"comprehensive_analysis_{data1_num}_{data2_num}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()
