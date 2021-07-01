import io
import json

from matplotlib.gridspec import GridSpec
from PIL import Image
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

import utils

num = 1 # filenames & title depend on this
load_json = False # 'True' to load previously generated .json; 'False' to generate new .json

if load_json: # load .json from './json/'
    with open(f'json/body_{num}.json', 'r') as f:
        body = json.load(f)

    with open(f'json/response_{num}.json', 'r') as f:
        resp = json.load(f)
else:
    body, resp = utils.get_jsons(num, batch=20, size=128) # default kwargs (batch=10, size=128)

df = utils.get_df(resp)
df.to_csv(f'csv/data_{num}.csv') # output .csv to './csv/'

# polar bar chart parameters
sum = df['count'].sum()
offset = 400
cmap = mcolors.LinearSegmentedColormap.from_list("", ["red", "yellow", "green"])
padding = 4
width = 2 * np.pi / len(df.index) 
angles = np.arange(0, 2 * np.pi, width)[:len(df.index)]

# canvas
fig = plt.figure(figsize=(22, 16), constrained_layout=True) # (22, 16)

# polar bar chart
gspolarpie = GridSpec(nrows=1, ncols=2)
ax1 = fig.add_subplot(gspolarpie[:, 0], polar=True, zorder=1)
ax1.axis('off')
ax1.set_title(f'Station {num} Sample Counts: {sum}', va='bottom', fontsize=30, x=0.57, y=1.10)
ax1.set_theta_zero_location('N')
ax1.set_theta_direction(-1)

bars = ax1.bar(x=angles, height=np.log(df['count']) * 100, width=width, bottom=offset, linewidth=4, edgecolor='white', color=cmap(np.log(df['count']) / np.log(df['count']).max()))
bars[(df[df['label'] == 'detritus'].iloc[0].name)].set_color('lightsteelblue')
bars[(df[df['label'] == 'detritus'].iloc[0].name)].set_edgecolor('white')

# text rotation
for bar, angle, label in zip(bars, angles, df['label']):
    rotation = -np.rad2deg(angle)

    if angle >= 0 and angle < np.pi:
        alignment = "left"
        rotation += 90
    else:
        alignment = "right"
        rotation += 270

    ax1.text(x=angle, y=offset + bar.get_height() + padding, s=label, ha=alignment, va='center', rotation=rotation, rotation_mode='anchor')

# pie chart parameters
plankton_count = df.groupby('type')['count'].sum().values[1]
detritus_count = df[df['label'] == 'detritus'].iloc[0]['count']
others_count = df.groupby('type')['count'].sum().values[0] - detritus_count
sectors = [plankton_count, others_count, detritus_count]
sector_labels = ["Plankton", "Others", "Detritus"]
sector_colors = ['royalblue', 'dodgerblue', 'lightsteelblue']
radius = -0.0035 * len(df.index) + 0.675

# pie chart
ax2 = fig.add_subplot(gspolarpie[:, 0], zorder=0)
ax2.pie(sectors, radius=radius, autopct='%1.0f%%', startangle=90, labeldistance=0, counterclock=False, pctdistance=0.5, colors=sector_colors, textprops={'fontsize': 16})
ax2.legend(sector_labels, fontsize=14, loc='upper left') # 10 x 128 =  0.48, 5 x 128 = 0.51

# plankton image column parameters
plankton_1 = df.iloc[0]
plankton_2 = df.iloc[1]
plankton_3 = df.iloc[2]
detritus = df[df['label'] == 'detritus'].iloc[0]
img_byte_1 = utils.get_image(body, plankton_1['sample'])
img_byte_2 = utils.get_image(body, plankton_2['sample'])
img_byte_3 = utils.get_image(body, plankton_3['sample'])
img_byte_4 = utils.get_image(body, detritus['sample'])

with Image.open(io.BytesIO(img_byte_1)) as f:
    img_1 = f.resize((224, 224), resample=Image.NEAREST)

with Image.open(io.BytesIO(img_byte_2)) as f:
    img_2 = f.resize((224, 224), resample=Image.NEAREST)

with Image.open(io.BytesIO(img_byte_3)) as f:
    img_3 = f.resize((224, 224), resample=Image.NEAREST)

with Image.open(io.BytesIO(img_byte_4)) as f:
    img_4 = f.resize((224, 224), resample=Image.NEAREST)

# plankton image column
gsimg = GridSpec(nrows=6, ncols=5)
ax31 = fig.add_subplot(gsimg[1:2, 2])
ax31.imshow(img_1)
ax31.set_title(f"{plankton_1['label']}\n{plankton_1['count']} counts ({round(plankton_1['count']/sum * 100, 2)}%)", fontsize=10)
ax31.axis('off')

ax32 = fig.add_subplot(gsimg[2:3, 2])
ax32.imshow(img_2)
ax32.set_title(f"{plankton_2['label']}\n{plankton_2['count']} counts ({round(plankton_2['count']/sum * 100, 2)}%)", fontsize=10)
ax32.axis('off')

ax33 = fig.add_subplot(gsimg[3:4, 2])
ax33.imshow(img_3)
ax33.set_title(f"{plankton_3['label']}\n{plankton_3['count']} counts ({round(plankton_3['count']/sum * 100, 2)}%)", fontsize=9)
ax33.axis('off')

ax34 = fig.add_subplot(gsimg[4:5, 2])
ax34.imshow(img_4)
ax34.set_title(f"{detritus['label']}\n{detritus['count']} counts ({round(detritus['count']/sum * 100, 2)}%)", fontsize=9)
ax34.axis('off')

plt.savefig(f'charts/station_{num}.jpg', bbox_inches='tight') # output .jpg to './charts/'
plt.show()
plt.clf()