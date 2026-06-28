"""
Schémas numériques pour l'équation de transport (advection 1D)

    du/dt + c du/dx = 0,   c > 0

On résout cette EDP par différences finies sur une grille [0, 1], et on compare
quatre schémas classiques sur trois conditions initiales différentes. La solution
exacte est simplement la condition initiale translatée : u(x, t) = u0(x - c t).

Schémas implémentés :
  - Euler upwind     : décentré amont, stable sous CFL, mais diffusif
  - Euler centré     : centré en espace, non dissipatif -> instable pour l'advection
  - Lax-Wendroff     : ordre 2, ajoute un terme de diffusion correctif
  - Lax-Friedrichs   : moyenne des voisins, stable mais très diffusif

Condition de stabilité (CFL) : lambda = c*dt/dx <= 1.

Auteur : projet d'analyse numérique (MAM).
"""

import os
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Les quatre schémas
# Chacun avance la solution de nt pas de temps et renvoie u au temps final.
# left_bc(t) fournit la valeur au bord gauche (entrée, c > 0) : on y injecte
# la solution exacte, ce qui marche pour les trois conditions initiales.
# ---------------------------------------------------------------------------

def euler_upwind(u0, c, dx, dt, nt, left_bc):
    u = u0.copy().astype(float)
    lam = c * dt / dx
    for n in range(nt):
        un = u.copy()
        u[1:] = un[1:] - lam * (un[1:] - un[:-1])
        u[0] = left_bc((n + 1) * dt)
    return u


def euler_centre(u0, c, dx, dt, nt, left_bc):
    u = u0.copy().astype(float)
    lam = c * dt / dx
    for n in range(nt):
        un = u.copy()
        # intérieur : différence centrée
        u[1:-1] = un[1:-1] - (lam / 2) * (un[2:] - un[:-2])
        # bord droit : upwind (extrapolation amont)
        u[-1] = un[-1] - lam * (un[-1] - un[-2])
        u[0] = left_bc((n + 1) * dt)
    return u


def lax_wendroff(u0, c, dx, dt, nt, left_bc):
    u = u0.copy().astype(float)
    lam = c * dt / dx
    for n in range(nt):
        un = u.copy()
        advection = (lam / 2) * (un[2:] - un[:-2])
        diffusion = (lam ** 2 / 2) * (un[2:] - 2 * un[1:-1] + un[:-2])
        u[1:-1] = un[1:-1] - advection + diffusion
        u[-1] = un[-1] - lam * (un[-1] - un[-2])
        u[0] = left_bc((n + 1) * dt)
    return u


def lax_friedrichs(u0, c, dx, dt, nt, left_bc):
    u = u0.copy().astype(float)
    lam = c * dt / dx
    for n in range(nt):
        un = u.copy()
        u[1:-1] = 0.5 * (un[2:] + un[:-2]) - 0.5 * lam * (un[2:] - un[:-2])
        u[-1] = un[-1] - lam * (un[-1] - un[-2])
        u[0] = left_bc((n + 1) * dt)
    return u


SCHEMES = {
    "Euler upwind": euler_upwind,
    "Euler centre": euler_centre,
    "Lax-Wendroff": lax_wendroff,
    "Lax-Friedrichs": lax_friedrichs,
}


# ---------------------------------------------------------------------------
# Conditions initiales et leur solution exacte (profil translaté de c*t)
# ---------------------------------------------------------------------------

def make_initial_conditions():
    """Renvoie un dict {nom: u0_func} où u0_func(x) donne le profil à t=0.

    La solution exacte au temps t est u0_func(x - c*t).
    """
    def sinusoide(x):
        return np.sin(2 * np.pi * x)

    def creneau(x):
        return ((x >= 0.4) & (x <= 0.6)).astype(float)

    def gaussienne(x):
        alpha, x0 = 200.0, 0.3
        return np.exp(-alpha * (x - x0) ** 2)

    return {
        "Sinusoide": sinusoide,
        "Creneau": creneau,
        "Gaussienne": gaussienne,
    }


# ---------------------------------------------------------------------------
# Comparaison d'un schéma (ou de tous) sur une condition initiale donnée
# ---------------------------------------------------------------------------

def run_case(u0_func, scheme_fn, c, dx, dt, nt, x):
    """Fait tourner un schéma et renvoie (u_num, u_exact, erreur_max)."""
    T = nt * dt
    u0 = u0_func(x)
    # bord gauche = solution exacte injectée en entrée
    left_bc = lambda t: float(u0_func(np.array([x[0] - c * t]))[0])
    u_num = scheme_fn(u0, c, dx, dt, nt, left_bc)
    u_exact = u0_func(x - c * T)
    err = np.max(np.abs(u_num - u_exact))
    return u_num, u_exact, err


def compare_all_schemes(ic_name, u0_func, c=1.0, Nx=200, cfl=0.8, nt=100,
                        outdir="figures", show=False):
    """Compare les 4 schémas sur une condition initiale et trace le résultat."""
    x = np.linspace(0, 1, Nx)
    dx = x[1] - x[0]
    dt = cfl * dx / c
    T = nt * dt

    fig, ax = plt.subplots(figsize=(10, 6))
    # solution exacte de référence
    ax.plot(x, u0_func(x - c * T), color="black", lw=2, label="Exacte")

    print(f"\n=== {ic_name}  (CFL = {cfl}, t = {T:.3f}) ===")
    for name, fn in SCHEMES.items():
        u_num, _, err = run_case(u0_func, fn, c, dx, dt, nt, x)
        print(f"  {name:<16s} erreur max = {err:.4f}")
        ax.plot(x, u_num, "--", lw=1.5, label=f"{name} (err={err:.2f})")

    ax.set_title(f"Equation de transport - {ic_name} (CFL={cfl})")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_ylim(-0.8, 1.8)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"comparaison_{ic_name.lower()}.png")
    fig.savefig(path, dpi=130)
    print(f"  -> figure enregistree : {path}")
    if show:
        plt.show()
    plt.close(fig)


def cfl_stability_demo(outdir="figures", show=False):
    """Illustre la condition CFL : upwind reste stable a CFL<1, le schema
    centre diverge. On compare aussi upwind a CFL=0.8 et la même IC."""
    x = np.linspace(0, 1, 200)
    dx = x[1] - x[0]
    c = 1.0
    nt = 100
    u0_func = make_initial_conditions()["Sinusoide"]

    fig, ax = plt.subplots(figsize=(10, 6))
    T_ref = nt * (0.8 * dx / c)
    ax.plot(x, u0_func(x - c * T_ref), color="black", lw=2, label="Exacte")

    for cfl, style in [(0.8, "--"), (0.95, ":")]:
        dt = cfl * dx / c
        u_num, _, err = run_case(u0_func, euler_upwind, c, dx, dt, nt, x)
        ax.plot(x, u_num, style, label=f"Upwind CFL={cfl} (err={err:.2f})")

    ax.set_title("Influence de la condition CFL (schema upwind)")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "cfl_demo.png")
    fig.savefig(path, dpi=130)
    print(f"\n-> demo CFL enregistree : {path}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Programme principal
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ics = make_initial_conditions()

    # Comparaison des 4 schemas sur chaque condition initiale
    compare_all_schemes("Sinusoide", ics["Sinusoide"], cfl=0.8, nt=100)
    compare_all_schemes("Creneau", ics["Creneau"], Nx=500, cfl=0.8, nt=200)
    compare_all_schemes("Gaussienne", ics["Gaussienne"], cfl=0.8, nt=80)

    # Demonstration de la condition de stabilite CFL
    cfl_stability_demo()

    print("\nTermine. Les figures sont dans le dossier 'figures/'.")
