import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def analyze_dtw_data(db_path):
    """データの基本統計を分析"""
    conn = sqlite3.connect(db_path)

    # データ番号の範囲を確認
    range_query = """
    SELECT 
        MIN(data_num1) as min_num1,
        MAX(data_num1) as max_num1,
        MIN(data_num2) as min_num2,
        MAX(data_num2) as max_num2,
        COUNT(DISTINCT data_num1) as unique_num1,
        COUNT(DISTINCT data_num2) as unique_num2
    FROM results
    """
    ranges = pd.read_sql_query(range_query, conn)
    print("データ番号の範囲:")
    print(ranges)

    # 基本統計量の取得
    stats_query = """
    SELECT 
        COUNT(*) as total_pairs,
        AVG(distance) as avg_distance,
        MIN(distance) as min_distance,
        MAX(distance) as max_distance,
        AVG(path_quality) as avg_quality,
        MIN(path_quality) as min_quality,
        MAX(path_quality) as max_quality,
        AVG(path_smoothness) as avg_smoothness,
        MIN(path_smoothness) as min_smoothness,
        MAX(path_smoothness) as max_smoothness
    FROM results
    """
    stats = pd.read_sql_query(stats_query, conn)
    print("\n基本統計:")
    print(stats)

    return conn, ranges, stats


def create_distance_matrix(conn, ranges, sample_size=1000):
    """サンプリングしたDTW距離行列を作成"""
    min_num = min(ranges["min_num1"].iloc[0], ranges["min_num2"].iloc[0])
    max_num = max(ranges["max_num1"].iloc[0], ranges["max_num2"].iloc[0])
    size = max_num - min_num + 1

    # サンプリングするインデックスを選択
    if size > sample_size:
        sampled_indices = np.sort(np.random.choice(size, sample_size, replace=False))
        size = sample_size
    else:
        sampled_indices = np.arange(size)

    # 距離行列の初期化
    distance_matrix = np.full((size, size), np.nan)

    # サンプリングしたデータの取得
    indices_list = sampled_indices + min_num
    indices_str = ",".join(map(str, indices_list))
    query = f"""
    SELECT data_num1, data_num2, distance 
    FROM results 
    WHERE data_num1 IN ({indices_str})
    AND data_num2 IN ({indices_str})
    """

    data = pd.read_sql_query(query, conn)

    # 行列の作成
    for _, row in data.iterrows():
        i = np.where(sampled_indices == (row["data_num1"] - min_num))[0]
        j = np.where(sampled_indices == (row["data_num2"] - min_num))[0]
        if len(i) > 0 and len(j) > 0:
            distance_matrix[i[0], j[0]] = row["distance"]
            distance_matrix[j[0], i[0]] = row["distance"]  # 対称行列

    return distance_matrix, sampled_indices + min_num


def plot_distance_heatmap(distance_matrix, indices, min_val=None, max_val=None):
    """距離行列のヒートマップを描画"""
    plt.figure(figsize=(12, 10))

    # 無効な値をマスク
    masked_matrix = np.ma.masked_invalid(distance_matrix)

    # ヒートマップの描画
    sns.heatmap(masked_matrix, cmap="viridis", vmin=min_val, vmax=max_val)

    # 軸ラベルの調整
    num_ticks = 5
    plt.xticks(
        np.linspace(0, len(indices) - 1, num_ticks),
        indices[:: len(indices) // num_ticks],
    )
    plt.yticks(
        np.linspace(0, len(indices) - 1, num_ticks),
        indices[:: len(indices) // num_ticks],
    )

    plt.title("DTW Distance Heatmap (Sampled)")
    plt.xlabel("Sequence Number")
    plt.ylabel("Sequence Number")
    plt.savefig("output/heatmap.png")
    plt.close()


def plot_quality_smoothness(conn, sample_size=10000):
    """Quality-Smoothnessの散布図"""
    query = f"""
    SELECT path_quality, path_smoothness, distance 
    FROM results 
    ORDER BY RANDOM() 
    LIMIT {sample_size}
    """
    data = pd.read_sql_query(query, conn)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        data["path_quality"],
        data["path_smoothness"],
        c=data["distance"],
        cmap="viridis",
        alpha=0.5,
    )
    plt.colorbar(scatter, label="Distance")
    plt.xlabel("Path Quality")
    plt.ylabel("Path Smoothness")
    plt.title("Path Quality vs Smoothness (colored by Distance)")
    plt.savefig("output/quality_smoothness.png")
    plt.close()


def analyze_distance_distribution(conn, sample_size=100000):
    """距離の分布を分析"""
    query = f"""
    SELECT distance 
    FROM results 
    ORDER BY RANDOM() 
    LIMIT {sample_size}
    """
    distances = pd.read_sql_query(query, conn)

    plt.figure(figsize=(10, 6))
    sns.histplot(data=distances, x="distance", bins=50)
    plt.title("Distribution of DTW Distances")
    plt.xlabel("Distance")
    plt.ylabel("Count")
    plt.savefig("output/distance_distribution.png")
    plt.close()

    # 基本統計量を表示
    print("\n距離の分布統計:")
    print(distances.describe())


def analyze_sequence_patterns(conn, ranges, bin_size=100):
    """シーケンス番号ごとの距離パターンを分析"""
    min_num = min(ranges["min_num1"].iloc[0], ranges["min_num2"].iloc[0])
    max_num = max(ranges["max_num1"].iloc[0], ranges["max_num2"].iloc[0])

    # ビンごとの平均距離を計算
    pattern_query = f"""
    SELECT 
        (data_num1 / {bin_size}) * {bin_size} as bin_start,
        AVG(distance) as avg_distance,
        COUNT(*) as pair_count
    FROM results 
    GROUP BY bin_start
    ORDER BY bin_start
    """
    patterns = pd.read_sql_query(pattern_query, conn)

    plt.figure(figsize=(12, 6))
    plt.plot(patterns["bin_start"], patterns["avg_distance"], "b.-")
    plt.title(f"Average Distance per Sequence (bin size: {bin_size})")
    plt.xlabel("Sequence Number")
    plt.ylabel("Average Distance")
    plt.grid(True)
    plt.savefig("output/average_distance_pattern.png")
    plt.close()

    # 特徴的なビンを表示
    print("\n特徴的なパターン:")
    print("最も距離が大きいビン:")
    print(patterns.nlargest(5, "avg_distance"))
    print("\n最も距離が小さいビン:")
    print(patterns.nsmallest(5, "avg_distance"))


def main():
    db_path = "results.db"

    # データ分析の実行
    print("基本統計の分析中...")
    conn, ranges, stats = analyze_dtw_data(db_path)

    # 距離行列の作成と可視化
    print("\n距離行列のサンプリングと可視化中...")
    distance_matrix, sampled_indices = create_distance_matrix(
        conn, ranges, sample_size=500
    )
    plot_distance_heatmap(
        distance_matrix,
        sampled_indices,
        min_val=stats["min_distance"].iloc[0],
        max_val=stats["max_distance"].iloc[0],
    )

    # その他の分析
    print("\nQuality-Smoothnessの関係を分析中...")
    plot_quality_smoothness(conn)

    print("\n距離分布を分析中...")
    analyze_distance_distribution(conn)

    print("\nシーケンスパターンを分析中...")
    analyze_sequence_patterns(conn, ranges)

    conn.close()
    print("\n分析完了。各プロットが保存されました。")


if __name__ == "__main__":
    main()
