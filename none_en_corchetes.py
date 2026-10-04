import torch

x = torch.zeros(128, 2)          # una curva: 128 puntos (x, y)
print(x.shape)                   # torch.Size([128, 2])
print(x[None].shape)             # torch.Size([1, 128, 2])  -> eje nuevo al inicio
print(x[:, None].shape)          # torch.Size([128, 1, 2])  -> eje nuevo en medio
print(x[:, :, None].shape)       # torch.Size([128, 2, 1])  -> eje nuevo al final

# Caso real 1: cdist exige lotes, pero una de las curvas no lo tiene
a = torch.randn(1, 300, 2)       # con eje de lote
b = torch.randn(128, 2)          # sin eje de lote

try:
    torch.cdist(a, b)
except RuntimeError as e:
    print("Sin None -> error:", e)

d = torch.cdist(a, b[None])      # con None funciona
print("Con None ->", d.shape)    # torch.Size([1, 300, 128])

# Caso real 2: distancia entre todos los pares de puntos (sin cdist)
p = torch.randn(5, 2)
q = torch.randn(7, 2)
dif = p[:, None, :] - q[None, :, :]   # (5,1,2) - (1,7,2) -> (5,7,2) por broadcasting
dist = dif.norm(dim=-1)               # (5, 7)
print("Todas las distancias:", dist.shape)