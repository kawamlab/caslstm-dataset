import pathlib

import matplotlib.pyplot as plt
import numpy as np
from natsort import natsorted
from pydantic import BaseModel

from dtw import dtw

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


if __name__ == "__main__":
    root_dir = pathlib.Path(__file__).resolve().parents[1]
    # Load the data from the Dataset directory
    data_dir = root_dir / "Dataset"

    data_files = natsorted(list(data_dir.glob("*.csv")))

    # get 2 data
    data1 = load_data(data_files[0])
    data2 = load_data(data_files[1])

    # x,y
    data1_x = np.array([d.x for d in data1])
    data1_y = np.array([d.y for d in data1])

    data2_x = np.array([d.x for d in data2])
    data2_y = np.array([d.y for d in data2])

    alignment = dtw(data1_y, data2_y, keep_internals=True)

    ax = alignment.plot(type="threeway")

    plt.savefig("output.png")

    plt.close()

    fig = plt.figure()

    ax = fig.add_subplot(111)

    ax.plot(data1_y)
    ax.plot(data2_y)

    plt.savefig("output2.png")

    plt.close()
