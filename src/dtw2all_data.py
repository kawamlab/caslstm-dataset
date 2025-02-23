import pathlib
import sqlite3
import threading
from datetime import datetime
from queue import Queue, Empty
from typing import List, Optional, Set, Tuple

import numpy as np
from pydantic import BaseModel
from tqdm import tqdm

from DTW import dtw


class StoneData(BaseModel):
    x: float
    y: float
    v_x: float
    v_y: float
    omega: float


class ResultData(BaseModel):
    data_num1: int
    data_num2: int
    distance: float
    path_quality: float
    path_smoothness: float


class WorkItem(BaseModel):
    trajectory_idx1: int
    trajectory_idx2: int


class ErrorLog(BaseModel):
    timestamp: datetime
    data_num1: int
    data_num2: int
    error_message: str


class ParallelDTW:
    def __init__(self, data_dir: pathlib.Path, n_workers: int = 8):
        self.data_dir = data_dir
        self.n_workers = n_workers

        # キューの初期化
        self.work_queue: Queue[WorkItem] = Queue()
        self.result_queue: Queue[ResultData] = Queue()
        self.error_queue: Queue[ErrorLog] = Queue()

        # データストレージの初期化
        self.trajectories: List[np.ndarray] = []
        self.total_combinations: int = 0

        # スレッド管理
        self.workers: List[threading.Thread] = []
        self.writer_thread: Optional[threading.Thread] = None
        self.stop_flag = threading.Event()

        # プログレスバー
        self.progress: Optional[tqdm] = None

        # DB接続（writer threadで使用）
        self.db_path = self.data_dir.parent / "results.db"
        self.error_log_path = self.data_dir.parent / "error_log.csv"

    def initialize_database(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS results (
                    data_num1 INTEGER,
                    data_num2 INTEGER,
                    distance REAL,
                    path_quality REAL,
                    path_smoothness REAL,
                    PRIMARY KEY (data_num1, data_num2)
                )
            """)

        # エラーログのヘッダー作成
        if not self.error_log_path.exists():
            with open(self.error_log_path, "w") as f:
                f.write("timestamp,data_num1,data_num2,error_message\n")

    def load_completed_pairs(self) -> Set[Tuple[int, int]]:
        """計算済みのペアを読み込む"""
        completed_pairs = set()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT data_num1, data_num2 FROM results")
            for row in cursor:
                completed_pairs.add((row[0], row[1]))
        return completed_pairs

    def load_data(self) -> int:
        """データの読み込みと前処理"""
        data_files = sorted(self.data_dir.glob("*.csv"))
        for data_file in data_files:
            stone_data = self.load_single_file(data_file)
            trajectory = np.array([[d.x, d.y] for d in stone_data])
            self.trajectories.append(trajectory)

        n_datasets = len(self.trajectories)
        self.total_combinations = n_datasets * (n_datasets - 1) // 2
        return n_datasets

    @staticmethod
    def load_single_file(data_file: pathlib.Path) -> List[StoneData]:
        """単一ファイルの読み込み"""
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

    def generate_work_items(self, n_datasets: int):
        """ワークアイテムの生成とキューへの投入"""
        # 計算済みのペアを取得
        completed_pairs = self.load_completed_pairs()

        # 総組み合わせ数を計算
        total_pairs = n_datasets * (n_datasets - 1) // 2

        # 未計算のペアのみをキューに投入
        remaining_pairs = 0
        for i in range(n_datasets):
            for j in range(i + 1, n_datasets):
                if (i, j) not in completed_pairs:
                    self.work_queue.put(WorkItem(trajectory_idx1=i, trajectory_idx2=j))
                    remaining_pairs += 1

        print(f"Total pairs: {total_pairs}")
        print(f"Completed pairs: {len(completed_pairs)}")
        print(f"Remaining pairs: {remaining_pairs}")

        # 残りのペア数で total_combinations を更新
        self.total_combinations = remaining_pairs

    def worker_function(self, worker_id: int):
        """ワーカースレッドのメイン関数"""
        while not self.stop_flag.is_set():
            try:
                work_item = self.work_queue.get(timeout=1)
            except Empty:
                break

            try:
                # DTW計算の実行
                distance, path, accumulated_cost, path_quality, path_smoothness = dtw(
                    self.trajectories[work_item.trajectory_idx1],
                    self.trajectories[work_item.trajectory_idx2],
                )

                # 結果の作成
                result = ResultData(
                    data_num1=work_item.trajectory_idx1,
                    data_num2=work_item.trajectory_idx2,
                    distance=float(distance),
                    path_quality=float(path_quality),
                    path_smoothness=float(path_smoothness),
                )

                self.result_queue.put(result)
                if self.progress is not None:
                    self.progress.update(1)

            except Exception as e:
                # エラーログの作成
                error_log = ErrorLog(
                    timestamp=datetime.now(),
                    data_num1=work_item.trajectory_idx1,
                    data_num2=work_item.trajectory_idx2,
                    error_message=str(e),
                )
                self.error_queue.put(error_log)

            finally:
                self.work_queue.task_done()

    def writer_function(self):
        """書き込みスレッドのメイン関数"""
        with sqlite3.connect(self.db_path) as conn:
            batch_count = 0
            results_buffer = []

            while not self.stop_flag.is_set() or not (
                self.result_queue.empty() and self.error_queue.empty()
            ):
                # 結果の書き込み
                try:
                    while True:
                        result = self.result_queue.get_nowait()
                        results_buffer.append(
                            (
                                result.data_num1,
                                result.data_num2,
                                result.distance,
                                result.path_quality,
                                result.path_smoothness,
                            )
                        )
                        batch_count += 1

                        if batch_count >= 1000:
                            conn.executemany(
                                """
                                INSERT INTO results 
                                (data_num1, data_num2, distance, path_quality, path_smoothness)
                                VALUES (?, ?, ?, ?, ?)
                                """,
                                results_buffer,
                            )
                            conn.commit()
                            batch_count = 0
                            results_buffer = []

                        self.result_queue.task_done()
                except Empty:
                    pass
                    # # バッファに残っているデータがあれば書き込む
                    # print(f"Writing {len(results_buffer)} results")
                    # if results_buffer:
                    #     conn.executemany(
                    #         """
                    #         INSERT INTO results
                    #         (data_num1, data_num2, distance, path_quality, path_smoothness)
                    #         VALUES (?, ?, ?, ?, ?)
                    #         """,
                    #         results_buffer,
                    #     )
                    #     conn.commit()
                    #     batch_count = 0
                    #     results_buffer = []

                # エラーログの書き込み
                try:
                    while True:
                        error = self.error_queue.get_nowait()
                        with open(self.error_log_path, "a") as f:
                            f.write(
                                f"{error.timestamp},{error.data_num1},{error.data_num2},{error.error_message}\n"
                            )
                        self.error_queue.task_done()
                except Empty:
                    pass

                threading.Event().wait(0.1)  # 短い待機

    def run(self):
        """メイン処理の実行"""
        # データベースの初期化
        self.initialize_database()

        # データの読み込み
        n_datasets = self.load_data()
        print(f"Loaded {n_datasets} datasets")

        # ワークアイテムの生成
        self.generate_work_items(n_datasets)
        print(f"Generated {self.total_combinations} work items")

        # プログレスバーの初期化
        self.progress = tqdm(total=self.total_combinations)

        # スレッドの開始
        self.workers = [
            threading.Thread(target=self.worker_function, args=(i,), name=f"Worker-{i}")
            for i in range(self.n_workers)
        ]
        self.writer_thread = threading.Thread(
            target=self.writer_function, name="Writer"
        )

        # スレッドの開始
        for worker in self.workers:
            worker.start()
        self.writer_thread.start()

        # 完了待ち
        self.work_queue.join()
        self.result_queue.join()
        self.error_queue.join()

        # スレッドの終了
        self.stop_flag.set()
        for worker in self.workers:
            worker.join()
        self.writer_thread.join()

        self.progress.close()
        print("Processing completed")


if __name__ == "__main__":
    root_dir = pathlib.Path(__file__).resolve().parents[1]
    data_dir = root_dir / "Dataset"

    processor = ParallelDTW(data_dir, n_workers=8)
    processor.run()
