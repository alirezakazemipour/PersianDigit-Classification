from matplotlib import pyplot as plt
import glob

fnames = glob.glob("/home/alireza/Desktop/*.png")
fnames.sort()

fig = plt.figure()
plt.subplots_adjust(wspace=0.1, hspace=-0.4)
for i, fname in enumerate(fnames):
    img = plt.imread(fname)
    ax = fig.add_subplot(2, 5, i + 1)
    # fig.subplots_adjust(top=0.85)
    ax.set_title(str(i), y=-0.3, style="italic", fontsize=12)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    # plt.axis("off")
    plt.imshow(img, cmap="gray")
fig.savefig("Numbers.png")





