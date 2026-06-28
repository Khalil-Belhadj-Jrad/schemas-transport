# Schémas numériques pour l'équation de transport

Projet d'analyse numérique sur la résolution d'EDP, centré sur l'équation de transport (advection) `∂u/∂t + c ∂u/∂x = 0`. On ne cherche pas une solution analytique mais une approximation par différences finies, en discrétisant le temps et l'espace sur une grille.

Le fil rouge du projet : comprendre quand un schéma numérique est stable, et pourquoi certains explosent alors que d'autres restent sages.

## Ce qu'on étudie

- **Le schéma d'Euler upwind** — un schéma décentré qui « regarde » dans le sens d'où vient l'information, ce qui le rend naturellement adapté au transport.
- **La condition CFL** — le point central : pour que le schéma soit stable, le pas de temps ne peut pas être choisi librement, il doit respecter `c·dt/dx ≤ 1`. Si on dépasse cette limite, la solution numérique diverge. Le projet illustre concrètement ce qui se passe de part et d'autre de ce seuil.
- **Comparaison à la solution exacte** — comme on connaît la solution analytique (une onde qui se translate sans se déformer), on peut mesurer précisément l'erreur du schéma.

Le rapport explore aussi les effets numériques typiques de ces schémas, comme la diffusion artificielle qui « lisse » la solution au fil des pas de temps.

## Contenu

- `rapport.pdf` — le rapport complet avec le code, les schémas, les courbes et l'analyse de stabilité

## Stack

Python, NumPy, Matplotlib.
