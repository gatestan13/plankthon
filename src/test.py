import json

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import altair as alt

# with open('data.json', 'r') as f:
#     data = json.load(f)

# num = 1

df = pd.read_csv('station_genus_count.csv')
df = df.loc['dn_20210111']


# df = df.sort_counts(by=['count'])

offset = 1000
padding = 4
cmap = mcolors.LinearSegmentedColormap.from_list("", ["red", "yellow", "green"])

plt.figure(figsize=(18, 14))
plt.tight_layout()
ax = plt.subplot(polar=True)
ax.set_title(f'Station 1', va='bottom', fontsize=20, y=1.0, pad=-200)

plt.axis('off')

# heights = df.count + offset
width = 2 * np.pi / len(df.index)
angles = np.arange(0, 2 * np.pi, width)
bars = ax.bar(x=angles, height=df.count, width=width, bottom=offset, linewidth=4, edgecolor='white', color=cmap(df.count / df.count.max()))

for bar, angle, label in zip(bars, angles, df.index):
    rotation = np.rad2deg(angle)

    if angle >= np.pi / 2 and angle < 3 * np.pi / 2:
        alignment = "right"
        rotation += 180
    else:
        alignment = "left"

    ax.text(x=angle, y=offset + bar.get_height() + padding, s=label, ha=alignment, va='center', rotation=rotation, rotation_mode='anchor')

# ax.set_theta_zero_location('N')

# plt.savefig(f'../../figure_{num}.jpg', bbox_inches='tight')
# plt.savefig(f'figure_{num}.jpg')
plt.show()