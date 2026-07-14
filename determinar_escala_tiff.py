#%%
import tifffile
from glob import glob
from os.path import join, basename

#%%
dir_imgs = "copias_mejoradas"

files = sorted(glob(join(dir_imgs, "*.tif")))

escalas = []

for file in files:

    try:
        with tifffile.TiffFile(file) as tif:

            md = tif.pages[0].tags[34682].value

            px_w = md["Scan"]["PixelWidth"] * 1e9   # nm/px
            px_h = md["Scan"]["PixelHeight"] * 1e9  # nm/px

            hfw = md["Scan"]["HorFieldsize"] * 1e6  # um
            vfw = md["Scan"]["VerFieldsize"] * 1e6  # um

            escalas.append(px_w)

            print(
                f'{basename(file):25s} | '
                f'Pixel = {px_w:.4f} x {px_h:.4f} nm/px | '
                f'HFW = {hfw:.3f} um | '
                f'VFW = {vfw:.3f} um'
            )

    except Exception as e:
        print(f'{basename(file):25s} | ERROR: {e}')

#%%
escalas_unicas = sorted(set(round(x, 6) for x in escalas))

print('\n' + '-'*80)

if len(escalas_unicas) == 1:
    print(f'Todas las imágenes tienen la misma escala: {escalas_unicas[0]:.6f} nm/px')
else:
    print('Se encontraron múltiples escalas:')
    for e in escalas_unicas:
        print(f'  {e:.6f} nm/px')