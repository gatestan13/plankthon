offset = 250
padding = 4
cmap = mcolors.LinearSegmentedColormap.from_list("", ["red", "yellow", "green"])
gspolar = GridSpec(nrows=1, ncols=2)
gs = GridSpec(nrows=7, ncols=4)
# hfont = {'fontname':'Tex Gyre Heros'}

# fig, ax = plt.subplots(1, figsize=(20, 20), subplot_kw=dict(polar=True, zorder=1))
fig = plt.figure(figsize=(15, 15), constrained_layout=True)
ax = fig.add_subplot(gspolar[0, :], polar=True)
ax.axis('off')
ax.set_title(f'Station 5 Sample Counts: {sumcounts}', va='bottom', fontsize=30, y=1.05, x=0.5)

sizes = [size1[1] / sum(size1), size1[0] / sum(size1)]
labels = ['Plankton', 'Others']

# heights = df.count + offset
width = 2 * np.pi / len(station5.index) 
angles = np.arange(0, 2 * np.pi, width)
bars = ax.bar(x=angles, height=station5['count'] * 100, width=width, bottom=offset, linewidth=4, edgecolor='white', color=cmap(station5['count'] / station5['count'].max()))

for bar, angle, label in zip(bars, angles, station5['genus']):
    rotation = -np.rad2deg(angle)

    # if angle >= np.pi / 2 and angle < 3 * np.pi / 2:
    #     alignment = "right"
    #     rotation += 180
    # else:
    #     alignment = "left"
    if angle >= 0 and angle < np.pi:
        alignment = "left"
        rotation += 90
    else:
        alignment = "right"
        rotation += 270

    ax.text(x=angle, y=offset + bar.get_height() + padding, s=label, ha=alignment, va='center', rotation=rotation, rotation_mode='anchor')

ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
# autopct='%1.2f%%'

# axp = fig.add_subplot(111, label='pie axes', zorder=0)
axp = fig.add_subplot(gspolar[0, :], label='pie axes', zorder=0)
axp.pie(sizes, radius=0.29, autopct='%1.0f%%', startangle=90, labeldistance=0, counterclock=False, pctdistance=0.5, colors=['dodgerblue', 'lightsteelblue'], textprops={'fontsize': 14})
axp.legend(labels, fontsize=15)
axp.set(aspect="equal")
# plt.savefig(f'../../figure_{num}.jpg', bbox_inches='tight')

im = Image.open('Cyanobacteria.jpg').resize((224, 224), resample=Image.NEAREST)
im2 = Image.open('Neoceratium_pentagonum.jpg').resize((224, 224), resample=Image.NEAREST)
im3 = Image.open('Steenstrupiella.jpg').resize((224, 224), resample=Image.NEAREST)
im4 = Image.open('Acantharia.jpg').resize((224, 224), resample=Image.NEAREST)
# aximg = fig.add_subplot(122, label='image')
# aximg.imshow(img)



# newax = fig.add_axes([1, 0.6, 0.2, 0.2], anchor='NE', zorder=-1)
# newax = fig.add_subplot(171, anchor='NE', zorder=-1)
newax = fig.add_subplot(gs[6, 0])
newax.imshow(im)
newax.set_title(f"Cyanobacteria\n {cyanocount} counts ({round(cyanocount/sumcounts * 100, 2)}%)", fontsize=10)
newax.axis('off')

# newax2 = fig.add_axes([1.4, 0.6, 0.2, 0.2], anchor='NE', zorder=-1)
# newax2 = fig.add_subplot(1, 7, (2, 2), anchor='NE', zorder=-1)
newax2 = fig.add_subplot(gs[6, 1])
newax2.imshow(im2)
newax2.set_title(f"Neoceratium pentagonum\n {neocount} counts ({round(neocount/sumcounts * 100, 2)}%)", fontsize=10)
newax2.axis('off')

# newax3 = fig.add_axes([1, 0.2, 0.2, 0.2], anchor='NE', zorder=-1)
# newax3 = fig.add_subplot(3, 7, 1, anchor='S', zorder=-1)
newax3 = fig.add_subplot(gs[6, 2])
newax3.imshow(im3)
newax3.set_title(f"Steenstrupiella\n {steecount} counts ({round(steecount/sumcounts * 100, 2)}%)", fontsize=10)
newax3.axis('off')

# newax4 = fig.add_axes([1.4, 0.2, 0.2, 0.2], anchor='NE', zorder=-1)
# newax4 = fig.add_subplot(3, 7, 2, anchor='S', zorder=-1)
newax4 = fig.add_subplot(gs[6, 3])
newax4.imshow(im4)
newax4.set_title(f"Acantharia\n {acancount} counts ({round(acancount/sumcounts * 100, 2)}%)", fontsize=10)
newax4.axis('off')
plt.savefig('Station_1.jpg')
plt.show()
# plt.savefig('Station_1.jpg')
plt.clf()