# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import Image, display

def f(x):      return (x - 1.5)**2 + 0.3
def grad_f(x): return 2*(x - 1.5)

x = 6.0
lr = 0.05
steps = 10          # stop earlier

xs = [x]
for _ in range(steps):
    x = x - lr * grad_f(x)
    xs.append(x)

xgrid = np.linspace(-2, 8, 400)
ygrid = f(xgrid)

fig, ax = plt.subplots(figsize=(6, 3.6))
ax.plot(xgrid, ygrid)
ax.set_xlabel("parameter (x)")
ax.set_ylabel("loss f(x)")
ax.set_title("1D Gradient Descent (follow the slope)")
ax.set_xlim(xgrid.min(), xgrid.max())
ax.set_ylim(ygrid.min() - 0.2, ygrid.max() + 0.2)

point, = ax.plot([], [], marker="o", markersize=8)
tangent, = ax.plot([], [], linewidth=2)
text = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top")

def init():
    point.set_data([], [])
    tangent.set_data([], [])
    text.set_text("")
    return point, tangent, text

def animate(i):
    xi = xs[i]
    yi = f(xi)
    gi = grad_f(xi)

    dx = 1.0
    xline = np.array([xi - dx, xi + dx])
    yline = yi + gi * (xline - xi)

    point.set_data([xi], [yi])
    tangent.set_data(xline, yline)
    text.set_text(f"step={i} | x={xi:.2f} | grad={gi:.2f} | x ← x − lr·grad")
    return point, tangent, text

anim = animation.FuncAnimation(
    fig, animate, init_func=init,
    frames=len(xs), interval=450, blit=True  # slower in notebook
)

gif_path = "gd_1d.gif"
anim.save(gif_path, writer=animation.PillowWriter(fps=1))  # slower playback
plt.close(fig)

display(Image(filename=gif_path))

# %%
