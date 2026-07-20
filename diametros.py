#%%
import os
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm, lognorm

# %%
res_M1 = glob("copias_mejoradas/medidas/M1*.csv")
res_M2 = glob("copias_mejoradas/medidas/M2*.csv")
res_M3 = glob("copias_mejoradas/medidas/M3*.csv")
res_M6 = glob("copias_mejoradas/medidas/M6*.csv")
res_M9 = glob("copias_mejoradas/medidas/M9*.csv")

res_M1.sort()
res_M2.sort()
res_M3.sort()
res_M6.sort()
res_M9.sort()
for r in [  res_M1, res_M2, res_M3, res_M6, res_M9]:
    print(r)

# %%
#%%
def leer_feret(lista_archivos):

    feret = np.array([])

    for archivo in lista_archivos:

        df = pd.read_csv(archivo)

        # eliminar la fila correspondiente a la imagen completa
        df = df[df['Label'].str.contains(':', na=False)]

        feret = np.concatenate((feret, df['Feret'].to_numpy()))

    return feret


feret_M1 = leer_feret(res_M1)
feret_M2 = leer_feret(res_M2)
feret_M3 = leer_feret(res_M3)
feret_M6 = leer_feret(res_M6)
feret_M9 = leer_feret(res_M9)
# %%
#%%

fig, ax = plt.subplots(figsize=(8,5), constrained_layout=True)
bins = np.arange(150, 450, 10)

for feret, nombre in zip([feret_M1, feret_M2, feret_M3, feret_M6, feret_M9],
    ['M1', 'M2', 'M3', 'M6', 'M9']):

    ax.hist(feret,bins=bins,density=False,
        linewidth=2,
        alpha=1,
        label=f'{nombre} (N={len(feret)})',
        histtype='bar',rwidth=0.9)

    ax.set_xlabel('Diámetro Feret (nm)')
    ax.set_ylabel('Densidad')
    ax.legend()

    plt.show()
#%%
from scipy.stats import lognorm

shape, loc, scale = lognorm.fit(feret_M1, floc=0)
#%%
def lector_resultados(path):

    data = pd.read_csv(
        path,
        usecols=['Label', 'Feret']
    )

    # eliminar la fila correspondiente a la imagen completa
    data = data[data['Label'].str.contains(':', na=False)]

    Feret = data['Feret'].to_numpy(dtype=float)

    return Feret

# %%
