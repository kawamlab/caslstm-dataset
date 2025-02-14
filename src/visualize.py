import pathlib
import matplotlib.pyplot as plt
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


if __name__ == "__main__":
    root_dir = pathlib.Path(__file__).resolve().parents[1]
    # Load the data from the Dataset directory
    data_dir = root_dir / "Dataset"

    data_files = natsorted(list(data_dir.glob("*.csv")))

    # trajectory plot
    # trajectory と velocityとomegaをplot

    fig = plt.figure(figsize=(10, 12))
    ax_tra = fig.add_subplot(411)
    ax_vel = fig.add_subplot(412)
    ax_omega = fig.add_subplot(413)
    ax_addition = fig.add_subplot(414)

    # trajectory plot
    ax_tra.set_aspect("equal")
    ax_tra.set_xlabel("y [m]")
    ax_tra.set_ylabel("x [m]")

    # velocity plot
    ax_vel_y = ax_vel.twinx()
    ax_vel.set_xlabel("time [s]")
    ax_vel.set_ylabel("x velocity [m/s]")
    ax_vel_y.set_ylabel("y velocity [m/s]")

    # addition plot
    ax_addition_y = ax_addition.twinx()
    ax_addition.set_xlabel("time [s]")
    ax_addition.set_ylabel("x addition [m]")
    ax_addition_y.set_ylabel("y addition [m]")

    # omega plot
    ax_omega.set_ylabel("omega [rad/s]")
    ax_omega.set_xlabel("time [s]")

    # titles
    ax_tra.set_title("Trajectory")
    ax_vel.set_title("Velocity")
    ax_omega.set_title("Omega")

    # grid
    ax_tra.grid()
    ax_vel.grid()
    ax_omega.grid()
    ax_addition.grid()

    for data_file in data_files[:]:
        print(data_file)

        data = load_data(data_file)

        x = [d.x for d in data]
        y = [d.y for d in data]

        v_x = [d.v_x for d in data]
        v_y = [d.v_y for d in data]

        omega = [d.omega for d in data]

        # 0.2秒ごとにデータが取られている
        time = [i * 0.2 for i in range(len(data))]

        ax_tra.plot(y, x, label=data_file.stem, linewidth=0.5)

        # ax_vel.plot(v_x, label=data_file.stem, linewidth=0.5)
        # ax_vel_y.plot(v_y, label=data_file.stem, linewidth=0.5)
        # ax_omega.plot(omega, label=data_file.stem, linewidth=0.5)

        ax_vel.plot(time, v_x, label=data_file.stem, linewidth=0.5)
        ax_vel_y.plot(time, v_y, label=data_file.stem, linewidth=0.5)
        ax_omega.plot(time, omega, label=data_file.stem, linewidth=0.5)

        # addition

        x_addition = [0]
        y_addition = [0]

        for i in range(1, len(data)):
            x_addition.append(x_addition[-1] + v_x[i - 1] * 0.2)
            y_addition.append(y_addition[-1] + v_y[i - 1] * 0.2)

        # ax_addition.plot(x_addition, label=data_file.stem, linewidth=0.5)
        # ax_addition_y.plot(y_addition, label=data_file.stem, linewidth=0.5)
        ax_addition.plot(time, x_addition, label=data_file.stem, linewidth=0.5)
        ax_addition_y.plot(time, y_addition, label=data_file.stem, linewidth=0.5)

    ax_tra.set_ylim(0, 5)

    fig.tight_layout()

    plt.savefig(root_dir / "output" / "output.png", bbox_inches="tight", dpi=300)
    plt.close()
