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

    q16, q84 = lognorm.ppf([0.16, 0.84], shape, loc=loc, scale=scale)
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
        'moda': moda,
        'q16':q16,
        'q84':q84}

#%%  Tablas de resultados
res_M1 = glob("copias_mejoradas/medidas/M1_*.csv")
res_M2 = glob("copias_mejoradas/medidas/M2_*.csv")
res_M3 = glob("copias_mejoradas/medidas/M3_*.csv")
res_M4 = glob("copias_mejoradas/medidas/M4_*.csv")
res_M5 = glob("copias_mejoradas/medidas/M5_*.csv")
res_M6 = glob("copias_mejoradas/medidas/M6_*.csv")
res_M7 = glob("copias_mejoradas/medidas/M7_*.csv")
res_M9 = glob("copias_mejoradas/medidas/M9_*.csv")
res_M10 = glob("copias_mejoradas/medidas/M10_*.csv")

res_M1.sort()
res_M2.sort()
res_M3.sort()
res_M4.sort()
res_M5.sort()
res_M6.sort()
res_M7.sort()
res_M9.sort()
res_M10.sort()
for r in [res_M1, res_M2, res_M3,res_M4, res_M5, res_M6,res_M7, res_M9,res_M10]:
    print(r)
#%%
ferets = {
    'M1': leer_feret(res_M1),
    'M2': leer_feret(res_M2),
    'M3': leer_feret(res_M3),
    'M4': leer_feret(res_M4),
    'M5': leer_feret(res_M5),
    'M6': leer_feret(res_M6),
    'M7': leer_feret(res_M7),
    'M9': leer_feret(res_M9),
    'M10': leer_feret(res_M10),}

#%% Distribuciones de diámetros Feret

fig, axs = plt.subplots(3, 3, figsize=(12, 12), constrained_layout=True)
axs = axs.ravel()

colores = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5','C6', 'C7', 'C8', 'C9']

for ax, (nombre, feret), c in zip(axs, ferets.items(), colores):

    bins = np.histogram_bin_edges(feret, bins='fd')

    ax.hist(feret,bins=bins,density=False,edgecolor='k',alpha=0.6,color=c)

    ax.set_title(f'{nombre}   N={len(feret)}   bins={len(bins)-1}',loc='left')

    ancho_bin = np.diff(bins)[0]

    print(f'{nombre}:')
    print(f'  Nº bins   = {len(bins)-1}')
    print(f'  Δbin      = {ancho_bin:.2f} nm\n')
    
axs[0].set_ylabel('Cuentas')
axs[3].set_ylabel('Cuentas')
axs[6].set_ylabel('Cuentas')
axs[6].set_xlabel('Diámetro Feret (nm)')
axs[7].set_xlabel('Diámetro Feret (nm)')
axs[8].set_xlabel('Diámetro Feret (nm)')

plt.suptitle('Distribución de diámetros Feret', fontsize=16)
plt.savefig('Distribuciones_Feret_M1_M2_M3_M4_M5_M6_M7_M9_M10.png', dpi=300)
plt.show()
#%% Ajuste lognormal de cada muestra
ajustes = {}
for nombre, feret in ferets.items():
    ajustes[nombre] = ajuste_lognormal_bootstrap(feret)
    print('-'*50)
    print(nombre)
    print(f'N        = {len(feret)}')
    print(f'Media    = {ajustes[nombre]["media"]:.2uP} nm')
    print(f'Std      = {ajustes[nombre]["std"]:.2uP} nm')
    print(f'Mediana  = {ajustes[nombre]["mediana"]:.2f} nm')
    print(f'Moda     = {ajustes[nombre]["moda"]:.2f} nm')
#%% Histogramas + ajuste lognormal

#%% Histogramas + ajuste lognormal

fig, axs = plt.subplots(3, 3, figsize=(13, 13), constrained_layout=True)
axs = axs.ravel()

colores = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']

for ax, (nombre, feret), c in zip(axs, ferets.items(), colores):

    bins = np.histogram_bin_edges(feret, bins='fd')

    # Histograma
    ax.hist(feret,bins=bins,density=True,
        alpha=0.5,color=c,edgecolor='k',label=f'{nombre}   N={len(feret)}'    )

    # Curva ajustada
    x = np.linspace(bins[0], bins[-1], 1000)

    pdf = lognorm.pdf(x,
        ajustes[nombre]['shape'],
        ajustes[nombre]['loc'],
        ajustes[nombre]['scale'])

    ax.axvspan(
        ajustes[nombre]['q16'],
        ajustes[nombre]['q84'],
        color=c,
        alpha=0.2,zorder=-1)

    # Curva lognormal
    ax.plot(x,pdf,color='k',
        lw=1.5,
        label=f'⟨d⟩ = {ajustes[nombre]["media"]:.1uS} nm\n'
              f'σ = {ajustes[nombre]["std"]:.1uS} nm'    )

    # Diámetro medio
    ax.axvline(
        ajustes[nombre]['media'].n,
        color=c,
        ls='--',
        lw=2)

    ax.legend(loc='upper left',shadow=True,frameon=True )

# Etiquetas
for i in [0, 3, 6]:
    axs[i].set_ylabel('Densidad (nm$^{-1}$)')

for i in [6, 7, 8]:
    axs[i].set_xlabel('Diámetro Feret (nm)')

plt.suptitle(
    'Distribuciones de diámetros Feret y ajuste lognormal',
    fontsize=16)

plt.savefig(
    'Distribuciones_Feret_lognormal_M1_M2_M3_M4_M5_M6_M7_M9_M10.png',
    dpi=300)

plt.show()
#%%

#%% Distribución y ajuste lognormal individual por muestra

colores = ['C0','C1','C2','C3','C4','C5','C6','C7','C8']

for (nombre, feret), c in zip(ferets.items(), colores):

    fig, ax = plt.subplots(figsize=(7,4), constrained_layout=True)

    # Histograma
    bins = np.histogram_bin_edges(feret, bins='fd')

    ax.hist(
        feret,
        bins=bins,
        density=True,
        alpha=0.5,
        color=c,
        edgecolor='k',
        label=f'{nombre}   N={len(feret)}'
    )

    # Curva ajustada
    x = np.linspace(bins[0], bins[-1], 1000)

    pdf = lognorm.pdf(
        x,
        ajustes[nombre]['shape'],
        ajustes[nombre]['loc'],
        ajustes[nombre]['scale']
    )

    # Intervalo central 68 %
    ax.axvspan(
        ajustes[nombre]['q16'],
        ajustes[nombre]['q84'],
        color=c,
        alpha=0.2,
        zorder=-1
    )

    # Ajuste lognormal
    ax.plot(
        x,
        pdf,
        color='k',
        lw=2,
        label=f'⟨d⟩ = {ajustes[nombre]["media"]:.1uS} nm\n'
              f' σ = {ajustes[nombre]["std"]:.1uS} nm'
    )

    # Diámetro medio
    ax.axvline(
        ajustes[nombre]['media'].n,
        color=c,
        ls='--',
        lw=2
    )

    ax.set_xlabel('Diámetro Feret (nm)')
    ax.set_ylabel(r'Densidad')
    ax.set_title(f'{nombre}')
    ax.grid(alpha=0.3)

    ax.legend(loc='upper left', frameon=True, shadow=True)

    plt.savefig(f'{nombre}_Distribucion_Feret_lognormal.png', dpi=300)
    plt.show()
#%% Guardar resultados del ajuste

resultados = []

for nombre, feret in ferets.items():

    aj = ajustes[nombre]

    resultados.append({
        'Muestra': nombre,
        'N': len(feret),
        'Media (nm)': aj['media'].n,
        'u(Media) (nm)': aj['media'].s,
        'Std (nm)': aj['std'].n,
        'u(Std) (nm)': aj['std'].s,
        'Mediana (nm)': aj['mediana'],
        'Moda (nm)': aj['moda'],
        'shape': aj['shape'],
        'scale': aj['scale']
    })

df_resultados = pd.DataFrame(resultados)

print(df_resultados)

df_resultados.to_csv(
    'Resultados_lognormal_Feret.txt',
    sep='\t',
    index=False,
    float_format='%.4f'
)

print('\nResultados guardados en:')
print('Resultados_lognormal_Feret.txt')


#%% Comparación de todas las muestras

fig, axs = plt.subplots(2, 1, figsize=(10,8), sharex=True, constrained_layout=True)
colores = ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9']

ax_hist = axs[0]
ax_fit = axs[1]


xmin = min(np.min(f) for f in ferets.values())
xmax = max(np.max(f) for f in ferets.values())
x = np.linspace(xmin, xmax, 1000)

for (nombre, feret), c in zip(ferets.items(), colores):

    bins = np.histogram_bin_edges(feret, bins='fd')

    ax_hist.hist(feret,bins=bins,density=False,alpha=0.5,color=c,edgecolor='None',label=f'{nombre}')

    pdf = lognorm.pdf(x,ajustes[nombre]['shape'],ajustes[nombre]['loc'],ajustes[nombre]['scale'])

    ax_fit.plot(x, pdf,color=c,lw=2,label=f'{nombre}: {ajustes[nombre]["media"]:.1uS}')
    ax_fit.axvline(ajustes[nombre]['media'].n,color=c,ls='--',lw=1.2,alpha=0.8)

ax_hist.set_ylabel('Frecuencia')
ax_fit.set_ylabel('Densidad (nm$^{-1}$)')
ax_fit.set_xlabel('Diámetro Feret (nm)')

ax_hist.set_title('Histogramas')
ax_fit.set_title('Ajustes lognormales')

ax_hist.legend(ncol=1,frameon=True,shadow=True)
ax_fit.legend(title='Diámetro medio (nm)' , ncol=1,frameon=True,shadow=True)

plt.suptitle('Comparación de las distribuciones de diámetros Feret',
             fontsize=16)
plt.savefig('Comparacion_Feret_lognormal_M1_M2_M3_M4_M5_M6_M7_M9_M10.png', dpi=300)
plt.show()
# %%
#%% Diámetro medio por muestra

fig, ax = plt.subplots(figsize=(6,3), constrained_layout=True)

muestras = list(ajustes.keys())

media = [ajustes[m]['media'].n for m in muestras]
dmedia = [ajustes[m]['media'].s for m in muestras]

colores = [f'C{i}' for i in range(len(muestras))]

for i, (m, y, dy, c) in enumerate(zip(muestras, media, dmedia, colores)):
    ax.errorbar(
        i, y,
        yerr=dy,
        fmt='.',
        ms=8,
        capsize=5,
        color=c,
        ecolor=c,
        elinewidth=2,
        label=f'{y:.1f}')

ax.set_xticks(range(len(muestras)))
ax.set_xticklabels(muestras)
# ax.legend(frameon=True)
ax.set_ylabel('Diámetro medio (nm)')
ax.set_xlabel('Muestra')
# ax.set_title('Diámetro medio obtenido del ajuste lognormal')

ax.grid(axis='y', alpha=0.3)

plt.savefig('Diametro_medio_por_muestra.png', dpi=300)
plt.show()
# %%
#%% Comparación del diámetro medio y la dispersión

fig, axs = plt.subplots(
    2, 1,
    figsize=(7, 5),
    sharex=True,
    constrained_layout=True
)

colores = ['C0','C1','C2','C3','C4','C5','C6','C7','C8']

muestras = list(ajustes.keys())

media = [ajustes[m]['media'].n for m in muestras]
dmedia = [ajustes[m]['media'].s for m in muestras]

std = [ajustes[m]['std'].n for m in muestras]
dstd = [ajustes[m]['std'].s for m in muestras]

# ------------------------
# Diámetro medio
# ------------------------

for i, (m, c) in enumerate(zip(muestras, colores)):
    axs[0].errorbar(
        i,
        media[i],
        yerr=dmedia[i],
        fmt='.',
        ms=8,
        color=c,
        capsize=4
    )

axs[0].set_ylabel(r'$\langle d\rangle$ (nm)')
axs[0].set_title('Diámetro medio ⟨d⟩ ',loc='left')
axs[0].grid(alpha=0.3)

# ------------------------
# Desviación estándar
# ------------------------

for i, (m, c) in enumerate(zip(muestras, colores)):
    axs[1].errorbar(
        i,
        std[i],
        yerr=dstd[i],
        fmt='.',
        ms=8,
        color=c,
        capsize=4
    )

axs[1].set_ylabel(r'$\sigma$ (nm)')
axs[1].set_title('Dispersión de tamaños σ   ',loc='left')
axs[1].grid(alpha=0.3)

axs[1].set_xticks(range(len(muestras)))
axs[1].set_xticklabels(muestras)

#plt.suptitle('Comparación de las distribuciones lognormales', fontsize=16)

plt.savefig('Diametro_medio_y_dispersion.png',dpi=300)

plt.show()
# %% Para plotear en funcion del diametro

# Diámetros medios (Feret) con incerteza del ajuste lognormal [1, 2]
# Formato: ufloat(media, error_ajuste)
diam_medio = np.array([
    ufloat(250.6, 3.6),   # M1
    ufloat(225.9, 3.6),   # M2
    ufloat(184.2, 1.8),   # M3
    ufloat(88.1, 1.8),    # M4
    ufloat(109.8, 1.1),   # M5
    ufloat(162.8, 1.5),   # M9
    ufloat(85.8, 0.6)     # M10
])

# ESAR (W/g) a 300 kHz y ~58 kA/m [3-9]
# Nota: La incerteza para M1-M5 y M9-M10 es la desviación estándar de las 3 repeticiones.
esar = np.array([
    ufloat(210, 13),      # M1
    ufloat(720, 31),      # M2
    ufloat(1235, 11),     # M3
    ufloat(1179, 47),     # M4
    ufloat(1327, 14),     # M5
    ufloat(360,  10),     # M9 
    ufloat(1258, 48)      # M10
])

# Tiempo de relajación tau (ns) [3-9]
tau = np.array([
    ufloat(37, 8),        # M1
    ufloat(75, 3),        # M2
    ufloat(56.3, 0.6),    # M3
    ufloat(137, 15),      # M4
    ufloat(116, 6),       # M5
    ufloat(47.0, 2.0),    # M9
    ufloat(115.7, 3.1)    # M10
])

# Campo coercitivo dinámico Hc (kA/m) [3-9]
hc = np.array([
    ufloat(7.5, 0.6),     # M1
    ufloat(12.01, 0.14),  # M2
    ufloat(9.79, 0.11),   # M3
    ufloat(18.7, 0.9),    # M4
    ufloat(16.9, 0.3),    # M5
    ufloat(8.5, 0.2),     # M9
    ufloat(16.33, 0.35)   # M10
])

wr = np.array([
    ufloat(0.26, 0.02),   # M1
    ufloat(2.21, 0.10),   # M2
    ufloat(4.44, 0.21),   # M3
    ufloat(1.80, 0.18),   # M4   
    ufloat(4.31, 0.16),   # M5
    ufloat(1.07, 0.05),   # M9   
    ufloat(6.0, 0.33)    # M10
])
# %%
#%% ESAR vs diámetro

from uncertainties import unumpy as unp

muestras = ['M1', 'M2', 'M3', 'M4', 'M5', 'M9', 'M10']
colores = ['C0','C1','C2','C3','C4','C5','C6']

# diámetro medio obtenido del ajuste lognormal
diam = np.array([ajustes[m]['media'].n for m in muestras])
ddiam = np.array([ajustes[m]['media'].s for m in muestras])

fig, ax = plt.subplots(figsize=(6,4), constrained_layout=True)

for i, (m, c) in enumerate(zip(muestras, colores)):
    ax.errorbar(
        diam[i],
        unp.nominal_values(esar)[i],
        xerr=ddiam[i],
        yerr=unp.std_devs(esar)[i],
        fmt='.',
        color=c,
        ms=8,
        capsize=4
    )

    ax.annotate(
        m,
        (diam[i], unp.nominal_values(esar)[i]),
        xytext=(5,5),
        textcoords='offset points',
        color=c
    )

ax.set_xlabel('Diámetro medio (nm)')
ax.set_ylabel('ESAR (W/g)')
ax.set_title('ESAR vs diámetro')
ax.grid(alpha=0.3)

plt.show()
# %%
#%% ESAR vs diámetro

from uncertainties import unumpy as unp

muestras = ['M1', 'M2', 'M3', 'M4', 'M5', 'M9', 'M10']
colores = ['C0','C1','C2','C3','C4','C5','C6']

# diámetro medio obtenido del ajuste lognormal
diam = np.array([ajustes[m]['media'].n for m in muestras])
ddiam = np.array([ajustes[m]['media'].s for m in muestras])

fig1, ax = plt.subplots(figsize=(6,3), constrained_layout=True)

for i, (m, c) in enumerate(zip(muestras, colores)):
    ax.errorbar(
        diam[i],
        unp.nominal_values(esar)[i],
        xerr=ddiam[i],
        yerr=unp.std_devs(esar)[i],
        fmt='.',
        color=c,
        ms=8,
        capsize=4
    )

    ax.annotate(
        m,
        (diam[i], unp.nominal_values(esar)[i]),
        xytext=(5,5),
        textcoords='offset points',
        color=c
    )

ax.set_xlabel('Diámetro medio (nm)')
ax.set_ylabel('ESAR (W/g)')
ax.set_title('ESAR vs diámetro')
ax.grid(alpha=0.3)

plt.show()
# %%
fig2, ax = plt.subplots(figsize=(6,3), constrained_layout=True)

for i, (m, c) in enumerate(zip(muestras, colores)):
    ax.errorbar(
        diam[i],
        unp.nominal_values(tau)[i],
        xerr=ddiam[i],
        yerr=unp.std_devs(tau)[i],
        fmt='.',
        color=c,
        ms=8,
        capsize=4
    )

    ax.annotate(
        m,
        (diam[i], unp.nominal_values(tau)[i]),
        xytext=(5,5),
        textcoords='offset points',
        color=c
    )

ax.set_xlabel('Diámetro medio (nm)')
ax.set_ylabel('tau (ns)')
ax.set_title('tau vs diámetro')
ax.grid(alpha=0.3)

plt.show()
# %%
fig3, ax = plt.subplots(figsize=(6,3), constrained_layout=True)

for i, (m, c) in enumerate(zip(muestras, colores)):
    ax.errorbar(
        diam[i],
        unp.nominal_values(hc)[i],
        xerr=ddiam[i],
        yerr=unp.std_devs(hc)[i],
        fmt='.',
        color=c,
        ms=8,
        capsize=4
    )

    ax.annotate(
        m,
        (diam[i], unp.nominal_values(hc)[i]),
        xytext=(5,5),
        textcoords='offset points',
        color=c
    )

ax.set_xlabel('Diámetro medio (nm)')
ax.set_ylabel('Hc (kA/m)')
ax.set_title('Hc vs diámetro')
ax.grid(alpha=0.3)

plt.show()
# %%
fig1.savefig('ESAR_vs_diam.png',dpi=300)
fig2.savefig('tau_vs_diam.png',dpi=300)
fig3.savefig('Hc_vs_diam.png',dpi=300)


#%% Grafico barras tau

valores = np.array([t.n for t in tau])
errores = np.array([t.s for t in tau])

# Figura tau

labels = ['M1','M2','M3','M4','M5','M9','M10']
colores = ['C0','C1','C2','C3','C4','C5','C6']
fig, ax = plt.subplots(figsize=(11,6), constrained_layout=True)

x = np.arange(len(labels))
for i, (label, valor, error, color) in enumerate(zip(labels, valores, errores, colores)):
    ax.bar(x=x[i], height=valor, yerr=error, color=color, edgecolor='k',linewidth=0,capsize=6)

# Etiquetas arriba de cada barra
for xi, yi, ei, t in zip(x, valores, errores, tau):
    ax.text(xi,yi + ei + 2,
        f'${t:.2uS}$',ha='center',
        va='bottom',fontsize=13)

ax.set_ylabel('Tau (ns)', fontsize=15)
ax.set_ylim(0, 165)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=13)
ax.grid(axis='y', ls='-', alpha=0.5)

# Agrupaciones
# Línea inferior NF@cit
ax.plot([x[0]-0.05, x[5]+0.05], [-16,-16], color='k', clip_on=False,lw=1.0)
ax.text((x[0]+x[5])/2,-18,'NF@cit',ha='center',va='top',fontsize=16)

# Línea inferior M10
ax.plot([x[6]-0.35, x[6]+0.35], [-16,-16], color='k', clip_on=False,lw=1)

ax.text(x[6],-18,'NF@PAA',
    ha='center',va='top',fontsize=15)

# Título
fig.suptitle(r'tiempos de relajación $\tau$ (ns)',
    fontsize=20)

plt.savefig('tau_vs_muestra.png',dpi=300)
plt.show()

# %% Grafico barras ESAR
valores = np.array([e.n for e in esar])
errores = np.array([e.s for e in esar])

labels = ['M1','M2','M3','M4','M5','M9','M10']
colores = ['C0','C1','C2','C3','C4','C5','C6']

fig, ax = plt.subplots(figsize=(11,6), constrained_layout=True)

x = np.arange(len(labels))

for i, (label, valor, error, color) in enumerate(zip(labels, valores, errores, colores)):
    ax.bar(
        x=x[i],
        height=valor,
        yerr=error,
        color=color,
        edgecolor='k',
        linewidth=0,
        capsize=6
    )

for xi, yi, ei, e in zip(x, valores, errores, esar):
    ax.text(
        xi,
        yi + ei + 20,
        f'${e:.2uS}$',
        ha='center',
        va='bottom',
        fontsize=13
    )

ax.set_ylabel('ESAR (W/g)', fontsize=15)
ax.set_ylim(0, 1450)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=13)

ax.grid(axis='y', ls='-', alpha=0.5)


ax.plot([x[0]-0.05, x[5]+0.05], [-140,-140],
        color='k', clip_on=False, lw=1)

ax.text((x[0]+x[5])/2, -170,
        'NF@cit',
        ha='center',
        va='top',
        fontsize=16)

ax.plot([x[6]-0.35, x[6]+0.35], [-140,-140],
        color='k', clip_on=False, lw=1)

ax.text(x[6], -170,
        'NF@PAA',
        ha='center',
        va='top',
        fontsize=15)

# ==========================================================
# Título
# ==========================================================

fig.suptitle('ESAR (W/g)', fontsize=20)

plt.savefig('ESAR_vs_muestra.png', dpi=300)
plt.show()
#%% Figura Campo Coercitivo


valores = np.array([e.n for e in hc])
errores = np.array([e.s for e in hc])
labels = ['M1','M2','M3','M4','M5','M9','M10']
colores = ['C0','C1','C2','C3','C4','C5','C6']

fig, ax = plt.subplots(figsize=(11,6), constrained_layout=True)

x = np.arange(len(labels))

for i, (valor, error, color) in enumerate(zip(valores, errores, colores)):
    ax.bar(
        x=x[i],
        height=valor,
        yerr=error,
        color=color,
        edgecolor='k',
        linewidth=0,
        capsize=6    )

for xi, yi, ei, h in zip(x, valores, errores, hc):
    ax.text(
        xi,
        yi + ei + 0.2,
        f'${h:.1uS}$',
        ha='center',
        va='bottom',
        fontsize=13
    )

ax.set_ylabel(r'$H_c$ (kA/m)', fontsize=15)
ax.set_ylim(0, max(valores + errores)*1.15)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=13)

ax.grid(axis='y', ls='-', alpha=0.5)

# y_line = 0
# y_text = -10

# ax.plot([x[0]-0.05, x[5]+0.05], [y_line, y_line],
#         color='k', clip_on=False, lw=1)

# ax.text((x[0]+x[5])/2, y_text,
#         'NF@cit',
#         ha='center',
#         va='top',
#         fontsize=16)

# ax.plot([x[6]-0.35, x[6]+0.35], [y_line, y_line],
#         color='k', clip_on=False, lw=1)

# ax.text(x[6], y_text,
#         'NF@PAA',
#         ha='center',
#         va='top',
#         fontsize=15)

# Agrupaciones
# Línea inferior NF@cit
ax.plot([x[0]-0.05, x[5]+0.05], [-1.5,-1.5], color='k', clip_on=False,lw=1.0)
ax.text((x[0]+x[5])/2,-2,'NF@cit',ha='center',va='top',fontsize=16)

# Línea inferior M10
ax.plot([x[6]-0.35, x[6]+0.35], [-1.5,-1.5], color='k', clip_on=False,lw=1)
ax.text(x[6],-2,'NF@PAA',ha='center',va='top',fontsize=15)

fig.suptitle(r'Campo coercitivo $H_c$', fontsize=20)
plt.savefig('Hc_vs_muestra.png', dpi=300)
plt.show()
#%% Diametros y dispersiones
# ==========================================================
diametros = [ufloat(d,e) for d,e in zip(diam,ddiam)]
valores = np.array([d.n for d in diametros])
errores = np.array([d.s for d in diametros])

desviacion = [ufloat(s,ds) for s,ds in zip(std,dstd)]
valores2 = np.array([des.n for des in desviacion])
errores2 = np.array([des.s for des in desviacion])

labels = ['M1','M2','M3','M4','M5','M9','M10']
colores = ['C0','C1','C2','C3','C4','C5','C6']

fig, (ax,ax2) = plt.subplots(2,1,figsize=(11,9), sharex=True,constrained_layout=True)

x = np.arange(len(labels))

for i, (valor, error, color) in enumerate(zip(valores, errores, colores)):
    ax.bar(
        x=x[i],
        height=valor,
        yerr=error,
        color=color,
        edgecolor='k',
        linewidth=0,
        capsize=6
    )
for xi, yi, ei, d in zip(x, valores, errores, diametros):
    ax.text(
        xi,
        yi + ei + 2,
        f'${d:.1uS}$',
        ha='center',
        va='bottom',
        fontsize=13
    )

for i, (valor, error, color) in enumerate(zip(valores2, errores2, colores)):
    ax2.bar(
        x=x[i],
        height=valor,
        yerr=error,
        color=color,
        edgecolor='k',
        linewidth=0,
        capsize=6
    )
for xi, yi, ei, d in zip(x, valores2, errores2, desviacion):
    ax2.text(
        xi,
        yi + ei+0.5 ,
        f'${d:.1uS}$',
        ha='center',
        va='bottom',
        fontsize=13
    )


ax.set_ylabel('Diámetro medio (nm)', fontsize=15)
ax2.set_ylabel('Dispersion de tamaños (nm)', fontsize=15)

ax.set_ylim(0, max(valores + errores)*1.15)
ax2.set_ylim(0, max(valores2 + errores2)*1.15)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=13)

ax.grid(axis='y', ls='-', alpha=0.5)
ax2.grid(axis='y', ls='-', alpha=0.5)

y_line = -5
y_text = -6

ax2.plot([x[0]-0.05, x[5]+0.05], [y_line, y_line],
        color='k', clip_on=False, lw=1)

ax2.text((x[0]+x[5])/2, y_text,
        'NF@cit',
        ha='center',
        va='top',
        fontsize=16)

ax2.plot([x[6]-0.35, x[6]+0.35], [y_line, y_line],
        color='k', clip_on=False, lw=1)

ax2.text(x[6], y_text,
        'NF@PAA',
        ha='center',
        va='top',
        fontsize=15)
plt.savefig('diametro_medio_vs_muestra.png', dpi=300)
plt.show()
# %% Warming Rate

valores = np.array([d.n for d in wr])
errores = np.array([d.s for d in wr])
fig, ax = plt.subplots(figsize=(11,6), constrained_layout=True)

for i, (valor, error, color) in enumerate(zip(valores, errores, colores)):
    ax.bar(
        i,
        valor,
        yerr=error,
        color=color,
        edgecolor='k',
        linewidth=0,
        capsize=6
    )

for xi, yi, ei, w in zip(range(len(muestras)), valores, errores, wr):
    ax.text(
        xi,
        yi + ei + 0.1,
        f'${w:.1uS}$',
        ha='center'
    )

ax.set_ylabel('Warming rate (°C/s)')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=13)
ax.set_ylim(0, max(valores + errores)*1.15)


y_line = -0.5
y_text = -0.8
ax.grid(axis='y', ls='-', alpha=0.5)
ax.plot([x[0]-0.05, x[5]+0.05], [y_line, y_line],
        color='k', clip_on=False, lw=1)

ax.text((x[0]+x[5])/2, y_text,
        'NF@cit',
        ha='center',
        va='top',
        fontsize=16)

ax.plot([x[6]-0.35, x[6]+0.35], [y_line, y_line],
        color='k', clip_on=False, lw=1)

ax.text(x[6], y_text,
        'NF@PAA',
        ha='center',
        va='top',
        fontsize=15)
plt.title('Warming rate (°C/s)', fontsize=20)
plt.savefig('warming_rate_vs_muestra.png', dpi=300)
plt.show()s


# %%
