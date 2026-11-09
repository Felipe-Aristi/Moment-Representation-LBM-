# Exporter from turbulent-single-component-jet-lbm/post/exportAndCrop.py
import matplotlib.pyplot as plt


def export_and_crop(fig, file_name, vertical_crop=True, dpi=600, transparent=False, pad_inches=0.01):
    fig.savefig(file_name, dpi=dpi, transparent=transparent)
    if vertical_crop:
        fig.savefig(file_name, dpi=dpi, transparent=transparent,
                    bbox_inches='tight', pad_inches=pad_inches)
    plt.close(fig)
