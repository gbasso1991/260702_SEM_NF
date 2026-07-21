#%% Distribuicion de diámetros Feret 
import os
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm
from uncertainties import ufloat, unumpy
#%%
def leer_feret(lista_archivos):
    feret = np.array([])
    for archivo in lista_archivos:
        df = pd.read_csv(archivo)
        df = df[df['Label'].str.contains(':', na=False)] # eliminar la fila correspondiente a la imagen completa
        feret = np.concatenate((feret, df['Feret'].to_numpy()))
    return feret
#%%
def ajuste_lognormal_bootstrap(datos, n_boot=1000):
    """
    Ajusta una distribución lognormal a un conjunto de diámetros mediante
    máxima verosimilitud (MLE) y estima las incertidumbres del diámetro medio
    y la desviación estándar mediante bootstrap.

    El ajuste se realiza sobre los datos individuales (no sobre el histograma),
    fijando el parámetro de localización en cero (`floc=0`). Las incertidumbres
    se estiman mediante remuestreo bootstrap con reemplazo.

    Parameters
    ----------
    datos : array_like
        Array unidimensional con los diámetros (por ejemplo, diámetros Feret)
        de las partículas.
    n_boot : int, optional
        Número de remuestras bootstrap utilizadas para estimar las
        incertidumbres. El valor por defecto es 1000.

    Returns
    -------
    dict
        Diccionario con los parámetros del ajuste:

        - 'shape' : float
            Parámetro de forma (σ) de la distribución lognormal.
        - 'loc' : float
            Parámetro de localización (igual a 0 al usar ``floc=0``).
        - 'scale' : float
            Parámetro de escala (= exp(μ)).
        - 'media' : uncertainties.ufloat
            Diámetro medio de la distribución lognormal con su incertidumbre
            bootstrap.
        - 'std' : uncertainties.ufloat
            Desviación estándar de la distribución lognormal con su
            incertidumbre bootstrap.
        - 'mediana' : float
            Mediana de la distribución ajustada.
        - 'moda' : float
            Moda de la distribución ajustada.

    Notes
    -----
    El bootstrap consiste en generar ``n_boot`` conjuntos de datos del mismo
    tamaño que el original mediante remuestreo con reemplazo. Cada conjunto se
    ajusta independientemente y la dispersión de los parámetros obtenidos se
    utiliza como estimación de su incertidumbre estadística.

    Este procedimiento es independiente del binning del histograma y resulta
    más robusto que estimar las incertidumbres a partir de un ajuste realizado
    sobre los histogramas.
    """
    shape, loc, scale = lognorm.fit(datos, floc=0)
    media = lognorm.mean(shape, loc, scale)
    std = lognorm.std(shape, loc, scale)
    mediana = lognorm.median(shape, loc, scale)
    moda = scale*np.exp(-shape**2)

    medias = np.empty(n_boot)
    stds = np.empty(n_boot)

    N = len(datos)

    for i in range(n_boot):
        muestra = np.random.choice(datos, N, replace=True)
        s, l, sc = lognorm.fit(muestra, floc=0)
        medias[i] = lognorm.mean(s, l, sc)
        stds[i] = lognorm.std(s, l, sc)

    return {'shape': shape,
        'loc': loc,
        'scale': scale,
        'media': ufloat(media, medias.std(ddof=1)),
        'std': ufloat(std, stds.std(ddof=1)),
        'mediana': mediana,
        'moda': moda}

#%%  Tablas de resultados
res_M1 = glob("copias_mejoradas/medidas/M1_*.csv")
res_M2 = glob("copias_mejoradas/medidas/M2_*.csv")
res_M3 = glob("copias_mejoradas/medidas/M3_*.csv")
res_M6 = glob("copias_mejoradas/medidas/M6_*.csv")
res_M9 = glob("copias_mejoradas/medidas/M9_*.csv")
res_M10 = glob("copias_mejoradas/medidas/M10_*.csv")

res_M1.sort()
res_M2.sort()
res_M3.sort()
res_M6.sort()
res_M9.sort()
res_M10.sort()
for r in [res_M1, res_M2, res_M3, res_M6, res_M9,res_M10]:
    print(r)
#%%
feret_M1 = leer_feret(res_M1)
feret_M2 = leer_feret(res_M2)
feret_M3 = leer_feret(res_M3)
feret_M6 = leer_feret(res_M6)
feret_M9 = leer_feret(res_M9)
feret_M10 = leer_feret(res_M10)

#%% Distribuciones de diámetros Feret

fig, axs = plt.subplots(3, 2, figsize=(10, 10), constrained_layout=True)
axs = axs.ravel()

for ax, feret, nombre in zip(axs,
    [feret_M1, feret_M2, feret_M3, feret_M6, feret_M9,feret_M10],
    ['M1', 'M2', 'M3', 'M6', 'M9','M10']):

    # Bins óptimos (Freedman-Diaconis)
    bins = np.histogram_bin_edges(feret, bins='fd')
    ax.hist(feret,bins=bins,density=True,edgecolor='k',alpha=0.6)

    ax.set_title(f'{nombre}   N={len(feret)}  bins = {len(bins)-1}', loc='left')
    ax.set_xlabel('Diámetro Feret (nm)')
    ax.set_ylabel('Densidad')
    ancho_bin = np.diff(bins)[0]

    print(f'{nombre}:')
    print(f'  Nº bins = {len(bins)-1}')
    print(f'  Ancho bin = {ancho_bin:.2f} nm')
# Eliminar el sexto subplot vacío
#fig.delaxes(axs[-1])
plt.suptitle('Distribución de diámetros Feret', fontsize=16)
plt.show()

#%% Ajuste lognormal de cada muestra

ajustes = {}
for feret, nombre in zip(
    [feret_M1, feret_M2, feret_M3, feret_M6, feret_M9,feret_M10],
    ['M1', 'M2', 'M3', 'M6', 'M9','M10']):

    ajustes[nombre] = ajuste_lognormal_bootstrap(feret)

    print('-'*50)
    print(nombre)

    print(f'N        = {len(feret)}')
    print(f'Media    = {ajustes[nombre]["media"]:.2uP} nm')
    print(f'Std      = {ajustes[nombre]["std"]:.2uP} nm')
    print(f'Mediana  = {ajustes[nombre]["mediana"]:.2f} nm')
    print(f'Moda     = {ajustes[nombre]["moda"]:.2f} nm')

#%% Histogramas + ajuste lognormal

fig, axs = plt.subplots(3,2,figsize=(10,10),constrained_layout=True)
axs = axs.ravel()

for ax, feret, nombre,c in zip(axs,
    [feret_M1, feret_M2, feret_M3, feret_M6, feret_M9,feret_M10],
    ['M1', 'M2', 'M3', 'M6', 'M9','M10'],['C0', 'C1', 'C2', 'C3', 'C4','C5']):

    bins = np.histogram_bin_edges(feret, bins='fd')

    ax.hist(feret,bins=bins,
        density=True,alpha=0.5,color=c,
        edgecolor='k',label='Datos')

    x = np.linspace(bins[0], bins[-1], 600)

    ax.plot(x,lognorm.pdf(x,ajustes[nombre]['shape'],ajustes[nombre]['loc'],ajustes[nombre]['scale']),
        'r',
        lw=1.5,
        label='Lognormal')

    ax.text(0.97,
        0.97,f'⟨d⟩ = {ajustes[nombre]["media"]:.1uS} nm\n'
        f'σ = {ajustes[nombre]["std"]:.1uS} nm',
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.9))

    ax.set_title(f'{nombre}   N={len(feret)}', loc='left')
    ax.set_xlabel('Diámetro Feret (nm)')
    ax.set_ylabel('Densidad')
    ax.legend(loc='upper left')
 
plt.suptitle('Distribuciones de diámetros Feret y ajuste lognormal',
             fontsize=16)

plt.show()
#%%