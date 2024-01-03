import matplotlib.pyplot as plt
import numpy as np

array = np.array([[0, 1, 1], [0, 1, 1], [1, 0, 0]])

fig, ax = plt.subplots()
ax.axis("off")

for (j, i), value in np.ndenumerate(array):
    if value == 1:
        rect = plt.Rectangle((i, 3 - j - 1), 1, 1, color="#8946FF")
        ax.add_patch(rect)

ax.set_xlim(0, 3)
ax.set_ylim(0, 3)
ax.set_aspect("equal")

plt.savefig("logo.png", dpi=32, bbox_inches="tight", transparent=True)
# plt.show()
