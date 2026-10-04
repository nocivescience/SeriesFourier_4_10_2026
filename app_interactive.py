"""
Dibuja una curva con el mouse y mira cómo PyTorch ajusta una serie de Fourier en vivo.

Uso:  python dibujar.py

Controles:
  - Mantén el botón izquierdo y dibuja un trazo. Al soltar, empieza el ajuste.
  - c        : borrar
  - r        : reajustar el mismo trazo (otra inicialización)
  - + / -    : más / menos armónicos (N) y reajusta
  - q        : salir
"""
import math

import matplotlib.pyplot as plt
import numpy as np
import torch

M = 128          # puntos con los que se resamplea tu trazo
T = 300          # puntos con los que se dibuja la serie
PASOS = 400      # pasos de Adam por ajuste
REFRESCO = 20    # cada cuántos pasos se redibuja la ventana
SUAVIDAD = 1e-4  # penalización de frecuencias altas

estado = {"N": 15, "trazo": [], "ocupado": False}


# ------------------------------------------------------------------ modelo
def curva_fourier(coef, t, N):
    """coef: (1, K, 2) con (re, im) -> puntos (1, T, 2)."""
    ks = torch.arange(-N, N + 1).float()
    ang = t[None, :, None] * ks[None, None, :]
    c, s = torch.cos(ang), torch.sin(ang)
    re, im = coef[..., 0][:, None, :], coef[..., 1][:, None, :]
    x = (c * re - s * im).sum(-1)
    y = (s * re + c * im).sum(-1)
    return torch.stack([x, y], -1)


def chamfer(a, b):
    d = torch.cdist(a, b)
    return d.min(-1).values.pow(2).mean() + d.min(-2).values.pow(2).mean()


def resamplear(pts, n):
    """Reparte n puntos a igual distancia a lo largo del trazo."""
    pts = np.asarray(pts, dtype=np.float32)
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    if len(pts) < 5 or d[-1] < 1e-3:
        return None
    s = np.linspace(0, d[-1], n)
    return np.stack([np.interp(s, d, pts[:, 0]), np.interp(s, d, pts[:, 1])], 1)


# ------------------------------------------------------------------ interfaz
fig, ax = plt.subplots(figsize=(7, 7))
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_aspect("equal")
ax.grid(alpha=0.3)
linea_trazo, = ax.plot([], [], "k--", lw=1.5, label="tu dibujo")
linea_fourier, = ax.plot([], [], "r", lw=2.5, label="serie de Fourier")
ax.legend(loc="upper right")


def titulo(txt):
    ax.set_title(f"N = {estado['N']} armónicos   |   {txt}")
    fig.canvas.draw_idle()


def ajustar():
    pts = resamplear(estado["trazo"], M)
    if pts is None:
        titulo("trazo demasiado corto, dibuja de nuevo")
        return
    estado["ocupado"] = True
    N = estado["N"]

    centro = pts.mean(0)
    radio = np.linalg.norm(pts - centro, axis=1).max() + 1e-8
    objetivo = torch.from_numpy(((pts - centro) / radio).astype(np.float32))[None]

    t = torch.linspace(0, 2 * math.pi, T)
    coef = torch.nn.Parameter(torch.randn(1, 2 * N + 1, 2) * 0.01)
    ks = torch.arange(-N, N + 1).float()
    opt = torch.optim.Adam([coef], lr=0.03)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, PASOS)

    for paso in range(PASOS + 1):
        opt.zero_grad()
        pred = curva_fourier(coef, t, N)
        pen = (ks ** 2 * (coef ** 2).sum(-1)).mean()
        perdida = chamfer(pred, objetivo) + SUAVIDAD * pen
        perdida.backward()
        opt.step()
        sched.step()
        if paso % REFRESCO == 0:
            p = pred.detach()[0].numpy() * radio + centro               # de vuelta a pantalla
            linea_fourier.set_data(p[:, 0], p[:, 1])
            titulo(f"ajustando... paso {paso}/{PASOS}  pérdida {perdida.item():.5f}")
            plt.pause(0.001)

    titulo(f"listo, pérdida {perdida.item():.5f}  (c borrar, r reajustar, +/- armónicos)")
    estado["ocupado"] = False


def al_presionar(e):
    if e.inaxes != ax or e.button != 1 or estado["ocupado"]:
        return
    estado["trazo"] = [(e.xdata, e.ydata)]
    linea_trazo.set_data([e.xdata], [e.ydata])
    linea_fourier.set_data([], [])
    titulo("dibujando...")


def al_mover(e):
    if e.inaxes != ax or e.button != 1 or estado["ocupado"] or not estado["trazo"]:
        return
    estado["trazo"].append((e.xdata, e.ydata))
    xs, ys = zip(*estado["trazo"])
    linea_trazo.set_data(xs, ys)
    fig.canvas.draw_idle()


def al_soltar(e):
    if e.button != 1 or estado["ocupado"] or len(estado["trazo"]) < 5:
        return
    ajustar()


def al_teclear(e):
    if estado["ocupado"]:
        return
    if e.key == "c":
        estado["trazo"] = []
        linea_trazo.set_data([], [])
        linea_fourier.set_data([], [])
        titulo("dibuja un trazo")
    elif e.key == "r" and estado["trazo"]:
        ajustar()
    elif e.key in ("+", "=", "up"):
        estado["N"] = min(estado["N"] + 5, 60)
        titulo("N cambiado")
        if estado["trazo"]:
            ajustar()
    elif e.key in ("-", "_", "down"):
        estado["N"] = max(estado["N"] - 5, 2)
        titulo("N cambiado")
        if estado["trazo"]:
            ajustar()
    elif e.key == "q":
        plt.close(fig)


fig.canvas.mpl_connect("button_press_event", al_presionar)
fig.canvas.mpl_connect("motion_notify_event", al_mover)
fig.canvas.mpl_connect("button_release_event", al_soltar)
fig.canvas.mpl_connect("key_press_event", al_teclear)

titulo("mantén el botón izquierdo y dibuja un trazo")
plt.show()