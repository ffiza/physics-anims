import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd
import os
from movie import create_movie

GRAVITY = 9.8  # m/s^2
LENGTH = 5.0  # m
FINAL_TIME = 20.0  # s
TIME_STEP = 0.01  # s
THETA0 = np.radians(45.0)  # radians
OMEGA0 = 0.0  # 1/s

DPI = 300
FIGSIZE = (15, 7.5)  # in

FRAMES_PATH = "animations/pendulum/frames"
FRAMERATE = int(1 / TIME_STEP)
VIDEO_NAME = "animations/pendulum/pendulum.avi"


# Define the differential equations
def equations(initial_conds, times):
    theta, omega = initial_conds
    f = [omega, - GRAVITY / LENGTH * np.sin(theta)]
    return f


def solve_pendulum(tf, dt, theta0, omega0):
    initial_conds = [theta0, omega0]
    times = np.arange(0, tf, dt)

    # Solve the differential equations
    solution = odeint(equations, initial_conds, times)
    theta = solution[:, 0]
    omega = solution[:, 1]
    gamma = - GRAVITY / LENGTH * np.sin(theta)

    return times, theta, omega, gamma


def create_dataframe(tf, dt, theta0, omega0):
    df = pd.DataFrame()

    times, theta, omega, gamma = solve_pendulum(tf, dt, theta0, omega0)

    df["Time"] = np.round(times, 5)
    df["Angle"] = np.round(theta, 5)
    df["AngularVelocity"] = np.round(omega, 5)
    df["AngularAcceleration"] = np.round(gamma, 5)
    df["TangentialVelocity"] = np.round(LENGTH * df["AngularVelocity"], 5)
    df["RadialAcceleration"] = np.round(- LENGTH * df["AngularVelocity"]**2, 5)
    df["TangentialAcceleration"] = np.round(
        LENGTH * df["AngularAcceleration"], 5)
    df["xPosition"] = np.round(LENGTH * np.cos(df["Angle"]), 5)
    df["yPosition"] = np.round(LENGTH * np.sin(df["Angle"]), 5)

    return df


def run_simulation(tf, dt, theta0, omega0):
    print("Simulating...", end='')
    df = create_dataframe(tf, dt, theta0, omega0)
    df.to_csv("data/pendulum.csv", index=False)
    print(" Done.")


def create_frames():
    df = pd.read_csv("data/pendulum.csv")
    if not os.path.exists(FRAMES_PATH):
        os.makedirs(FRAMES_PATH)
    y1_max = 1.1 * max(df["Angle"])
    y1_min = 1.1 * min(df["Angle"])
    x1_min = min(df["Time"])
    x1_max = max(df["Time"])
    n_frames = len(df)
    n_digits = len(str(n_frames))

    # Create one image for each timestep
    for i in range(n_frames):
        print(f"\rCreating frame {str(i+1).zfill(n_digits)} of {n_frames}...",
              end='')
        present_time = df["Time"].iloc[i]

        # Create a plot for each row in the data frame
        fig, axs = plt.subplots(1, 2, figsize=FIGSIZE)

        axs[1].tick_params(which='major', width=1.0, length=10, labelsize=20)
        axs[1].tick_params(which='minor', width=1.0, length=5, labelsize=10,
                           labelcolor='0.25')
        axs[1].grid(linestyle="-", linewidth=0.5, color='0.25', zorder=-10)

        axs[1].plot(df["Time"][df["Time"] <= present_time],
                    df["Angle"][df["Time"] <= present_time], lw=5)
        axs[1].set_xlabel("Tiempo [s]", fontsize=20)
        axs[1].set_ylabel("Ángulo", fontsize=20)
        axs[1].set_xlim(x1_min, x1_max)
        axs[1].set_ylim(y1_min, y1_max)

        x = df["yPosition"].iloc[i]
        y = -df["xPosition"].iloc[i]

        xvel_pos = df["TangentialVelocity"].iloc[i] * \
            np.sin(df["Angle"].iloc[i])
        yvel_pos = df["TangentialVelocity"].iloc[i] * \
            np.cos(df["Angle"].iloc[i])
        vel_scale = 0.5

        xacc_pos = - df["RadialAcceleration"].iloc[i] *  \
            np.cos(df["Angle"].iloc[i]) + \
            df["TangentialAcceleration"].iloc[i] * \
            np.sin(df["Angle"].iloc[i])
        yacc_pos = df["RadialAcceleration"].iloc[i] * \
            np.sin(df["Angle"].iloc[i]) + \
            df["TangentialAcceleration"].iloc[i] * np.cos(df["Angle"].iloc[i])
        acc_scale = 0.5

        axs[0].set_xlim(-8, 8)
        axs[0].set_ylim(-11, 5)
        axs[0].axis('off')
        axs[0].plot([-2, 2], [0, 0], 'k-', lw=2)
        axs[0].plot([0, 0], [2, -7], 'k--', lw=1, zorder=-5)
        axs[0].plot([0, x], [0, y], 'k-', lw=2)
        axs[0].plot([0], [0], 'ko', ms=15, mew=3, mec="white")
        axs[0].plot([x], [y], 'ko', ms=15, mew=3, mec="white")
        axs[0].arrow(x, y, dx=vel_scale * yvel_pos, dy=vel_scale * xvel_pos,
                     width=0.2, head_width=0.5, color='tab:red')
        axs[0].arrow(x, y, dx=acc_scale * yacc_pos, dy=acc_scale * xacc_pos,
                     width=0.2, head_width=0.5, color='tab:green')
        axs[0].arrow(x=0, y=0, dx=1, dy=0,
                     width=0.2, head_width=0.5, color='tab:purple')
        axs[0].arrow(x=0, y=0, dx=0, dy=-1,
                     width=0.2, head_width=0.5, color='tab:purple')
        axs[0].text(-0.7, -1.5, "$x$", fontsize=20,
                    va='center', ha='center', color='tab:purple')
        axs[0].text(1.5, 0.7, "$y$", fontsize=20,
                    va='center', ha='center', color='tab:purple')

        fig.savefig(f"{FRAMES_PATH}/frame_{str(i).zfill(n_digits)}.png",
                    dpi=DPI, bbox_inches="tight")
        plt.close()

    print(" Done.")


if __name__ == "__main__":
    run_simulation(FINAL_TIME, TIME_STEP, THETA0, OMEGA0)
    create_frames()
    create_movie(FRAMES_PATH, FRAMERATE, VIDEO_NAME, True)
