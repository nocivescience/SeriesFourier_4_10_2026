"""
Dibujar curvas con series de Fourier usando PyTorch.

Modos:
  python fourier_curvas.py single --curva corazon   # ajusta UNA curva por descenso de gradiente
  python fourier_curvas.py net                      # entrena una red que predice coeficientes

La curva es z(t) = sum_k c_k * exp(i k t), k = -N..N, con c_k = re + i*im (parámetros).
Todo es diferenciable, así que PyTorch optimiza los coeficientes (o la red que los produce).
"""
import argparse
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------ utilidades
def frecuencias(N):
    return torch.arange(-N, N + 1, device=device).float()          # (K,)


def curva_fourier(coef, t, N):
    """coef: (B, K, 2) con (re, im); t: (T,). Devuelve puntos (B, T, 2)."""
    ks = frecuencias(N)
    ang = t[None, :, None] * ks[None, None, :]                      # (1, T, K)
    c, s = torch.cos(ang), torch.sin(ang)
    re = coef[..., 0][:, None, :]                                   # (B, 1, K)
    im = coef[..., 1][:, None, :]
    x = (c * re - s * im).sum(-1)                                   # parte real de z
    y = (s * re + c * im).sum(-1)                                   # parte imaginaria de z
    return torch.stack([x, y], dim=-1)


def normalizar(p):
    """Centra y escala para que el radio máximo sea 1. p: (..., T, 2)."""
    p = p - p.mean(dim=-2, keepdim=True)
    r = p.norm(dim=-1).amax(dim=-1)[..., None, None]
    return p / (r + 1e-8)


def chamfer(a, b):
    """Distancia Chamfer simétrica. a: (B,Ta,2), b: (B,Tb,2)."""
    d = torch.cdist(a, b)                                           # (B, Ta, Tb)
    return d.min(-1).values.pow(2).mean() + d.min(-2).values.pow(2).mean()


def penalizacion_suavidad(coef, N):
    """sum k^2 |c_k|^2: castiga frecuencias altas -> curvas más suaves."""
    ks = frecuencias(N)
    return (ks ** 2 * (coef ** 2).sum(-1)).mean()


# ------------------------------------------------------------- curvas objetivo
def corazon(T=400):
    t = torch.linspace(0, 2 * math.pi, T, device=device)
    x = 16 * torch.sin(t) ** 3
    y = 13 * torch.cos(t) - 5 * torch.cos(2 * t) - 2 * torch.cos(3 * t) - torch.cos(4 * t)
    return normalizar(torch.stack([x, y], -1))


def estrella(T=400, puntas=5):
    t = torch.linspace(0, 2 * math.pi, T, device=device)
    r = 1 + 0.5 * torch.cos(puntas * t)
    return normalizar(torch.stack([r * torch.cos(t), r * torch.sin(t)], -1))


def cuadrado(T=400):
    lado = T // 4
    a = torch.linspace(-1, 1, lado, device=device)
    o, u = torch.ones_like(a), -torch.ones_like(a)
    pts = torch.cat([
        torch.stack([a, u], -1), torch.stack([o, a], -1),
        torch.stack([-a, o], -1), torch.stack([u, -a], -1)])
    return normalizar(pts)


CURVAS = {"corazon": corazon, "estrella": estrella, "cuadrado": cuadrado}


def graficar(pares, archivo):
    fig, axs = plt.subplots(1, len(pares), figsize=(4.5 * len(pares), 4.5))
    axs = [axs] if len(pares) == 1 else axs
    for ax, (titulo, objetivo, pred) in zip(axs, pares):
        ax.plot(*objetivo.cpu().T, "k--", lw=1, label="objetivo")
        ax.plot(*pred.cpu().T, "r", lw=2, label="Fourier")
        ax.set_title(titulo)
        ax.set_aspect("equal")
        ax.legend()
    plt.tight_layout()
    plt.savefig(archivo, dpi=120)
    print("Guardado:", archivo)


# ------------------------------------------------- MODO 1: ajustar una curva
def modo_single(args):
    N = args.N
    objetivo = CURVAS[args.curva]()[None]                           # (1, T, 2)
    t = torch.linspace(0, 2 * math.pi, 400, device=device)

    coef = nn.Parameter(torch.randn(1, 2 * N + 1, 2, device=device) * 0.01)
    opt = torch.optim.Adam([coef], lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.pasos)

    for paso in range(args.pasos + 1):
        opt.zero_grad()
        pred = curva_fourier(coef, t, N)
        perdida = chamfer(pred, objetivo) + args.suavidad * penalizacion_suavidad(coef, N)
        perdida.backward()
        opt.step()
        sched.step()
        if paso % 500 == 0:
            print(f"paso {paso:5d}  pérdida {perdida.item():.6f}")

    with torch.no_grad():
        pred = curva_fourier(coef, t, N)[0]
    graficar([(f"{args.curva} (N={N})", objetivo[0], pred)], f"ajuste_{args.curva}.png")


# --------------------------------- MODO 2: red que predice los coeficientes
def curvas_sinteticas(B, N, M, K_verdad=8):
    """Curvas aleatorias suaves (amplitud ~ 1/k^1.5) muestreadas en M puntos, normalizadas."""
    ks = torch.arange(-K_verdad, K_verdad + 1, device=device).float()
    amp = 1.0 / (1 + ks.abs()) ** 1.5
    coef = torch.randn(B, len(ks), 2, device=device) * amp[None, :, None]
    # reutilizamos curva_fourier con N=K_verdad
    t = torch.linspace(0, 2 * math.pi, M, device=device)
    return normalizar(curva_fourier(coef, t, K_verdad))


class RedCoef(nn.Module):
    """Puntos del contorno (M,2) -> coeficientes de Fourier (K,2)."""
    def __init__(self, M, N, ancho=512):
        super().__init__()
        self.N = N
        self.net = nn.Sequential(
            nn.Linear(M * 2, ancho), nn.ReLU(),
            nn.Linear(ancho, ancho), nn.ReLU(),
            nn.Linear(ancho, (2 * N + 1) * 2))

    def forward(self, pts):
        return self.net(pts.flatten(1)).view(-1, 2 * self.N + 1, 2)


def modo_net(args):
    N, M = args.N, 128
    red = RedCoef(M, N).to(device)
    opt = torch.optim.Adam(red.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.pasos)
    t = torch.linspace(0, 2 * math.pi, 200, device=device)

    for paso in range(args.pasos + 1):
        datos = curvas_sinteticas(128, N, M)
        coef = red(datos)
        pred = curva_fourier(coef, t, N)
        perdida = chamfer(pred, datos) + args.suavidad * penalizacion_suavidad(coef, N)
        opt.zero_grad()
        perdida.backward()
        opt.step()
        sched.step()
        if paso % 500 == 0:
            print(f"paso {paso:5d}  pérdida {perdida.item():.6f}")

    # Prueba con formas que NO estaban en el entrenamiento
    red.eval()
    pares = []
    with torch.no_grad():
        for nombre, f in CURVAS.items():
            tt = torch.linspace(0, 2 * math.pi, M, device=device)
            obj = f(M)[None]
            pred = curva_fourier(red(obj), t, N)[0]
            pares.append((nombre, obj[0], pred))
    graficar(pares, "red_coeficientes.png")


# ------------------------------------------------------------------------ main
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("modo", choices=["single", "net"])
    p.add_argument("--curva", default="corazon", choices=list(CURVAS))
    p.add_argument("--N", type=int, default=20, help="armónicos (frecuencias -N..N)")
    p.add_argument("--pasos", type=int, default=3000)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--suavidad", type=float, default=1e-4, help="peso de la penalización k^2|c|^2")
    args = p.parse_args()
    {"single": modo_single, "net": modo_net}[args.modo](args)