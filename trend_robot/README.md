# TSMOM Research Robot

Robot de **recherche et validation** d'une stratégie *Time-Series Momentum* (TSMOM) multi-actifs sur ETF, en Python. L'accent est mis sur un **harnais de validation rigoureux** (séparation train/test verrouillée, walk-forward, validation croisée purgée, Deflated Sharpe) plutôt que sur la sophistication du signal.

> ⚠️ **Avertissement — ceci n'est PAS un conseil financier ni un système de trading prêt à déployer.**
> Le périmètre est *recherche + validation*. L'exécution réelle / le trading live sont volontairement **hors périmètre**.
> Sur les données testées, le **verdict de validation (§6.5) est actuellement `REJECT`** : la stratégie ne satisfait pas le critère de robustesse et **ne doit pas être déployée**. Le mode *paper-trading* fourni est un **`--dry-run` de répétition d'ingénierie**, pas un signal de déploiement.

---

## Aperçu

Le pipeline opérationnalise :

- **Couche de données abstraite** (`DataProvider`) — yfinance (clôtures ajustées) + cache parquet, avec un fournisseur **synthétique déterministe** en repli (Yahoo est souvent rate-limité, HTTP 429).
- **Signal TSMOM** — moyenne des signes de rendement sur plusieurs horizons (`[21, 63, 126, 252]` jours), borné dans `[-1, 1]`.
- **Construction de portefeuille** — ciblage de volatilité par actif puis au niveau portefeuille, Kelly fractionnaire, plafond de levier brut.
- **Moteur de backtest réaliste** — **zéro look-ahead** (poids décalés d'une barre), coûts de transaction sur le turnover, mark-to-market.
- **Métriques honnêtes** — CAGR, vol, Sharpe, Sortino, Calmar/MAR, max drawdown + durée, profit factor, hit rate, turnover, exposition, **Deflated Sharpe** (Bailey & López de Prado), attribution de P&L par actif.
- **Harnais de validation** — split verrouillé 70/30, walk-forward glissant, **validation croisée purgée + embargo** (López de Prado 2018), compteur de *trials* pour la correction tests-multiples.
- **Tests de sensibilité aux coûts** + **rapport de validation final** évaluant le critère §6.5.
- **Dry-run paper-trading** (`run_live.py`) — calcule et **affiche** les ordres du jour sans rien envoyer, structuré pour brancher l'API **Alpaca paper**.

**85 tests** (pytest), tous verts, exécutés sur données synthétiques (aucun accès réseau requis).

---

## Structure du projet

```
trend_robot/
├── config.yaml                 # tous les paramètres (aucune valeur de marché en dur dans le code)
├── requirements.txt
├── run_research.py             # point d'entrée : backtest de bout en bout + rapport
├── run_validation.py           # point d'entrée : stress coûts + rapport de validation §6.5
├── run_live.py                 # point d'entrée : dry-run paper-trading (aperçu d'ordres)
├── trend_robot/
│   ├── config.py               # dataclass Config typée + chargement/validation YAML + seed global
│   ├── data/
│   │   ├── provider.py         # Protocol DataProvider + cache parquet (CachedProvider)
│   │   ├── yfinance_provider.py
│   │   └── synthetic_provider.py
│   ├── signals/tsmom.py        # signal TSMOM (pur)
│   ├── portfolio/sizing.py     # ciblage de vol, pondérations, levier (pur)
│   ├── backtest/
│   │   ├── engine.py           # moteur de backtest (no look-ahead, mark-to-market)
│   │   └── costs.py            # modèle de coûts
│   ├── metrics/
│   │   ├── performance.py      # métriques classiques + attribution P&L
│   │   └── deflated_sharpe.py  # Deflated Sharpe Ratio
│   ├── validation/
│   │   ├── splits.py           # split train/test verrouillé + walk-forward
│   │   ├── purged_cv.py        # validation croisée purgée + embargo
│   │   ├── trials.py           # compteur de configurations testées (n_trials)
│   │   ├── stress.py           # sensibilité aux coûts
│   │   └── final_report.py     # verdict §6.5 sur le jeu de test verrouillé
│   ├── reporting/report.py     # graphiques + tableaux (présentation uniquement)
│   └── live/                   # paper-trading (dry-run)
│       ├── broker.py           # Protocol Broker + LocalPaperBroker + AlpacaBroker (lazy)
│       ├── live_data.py        # prix "as of" (réutilise la couche de données)
│       ├── target.py           # poids cibles du jour (réutilise signal + sizing)
│       ├── executor.py         # poids cibles → ordres (diff vs positions, garde-fous)
│       └── state.py            # persistance / idempotence
└── tests/                      # 85 tests pytest (données synthétiques, hors réseau)
```

---

## Installation

Requiert **Python 3.11+** (développé et validé sous 3.14).

```bash
git clone <url-du-repo>
cd trend_robot

python3 -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Utilisation

Toutes les commandes se lancent depuis la racine du projet (`trend_robot/`).

### 1. Backtest de recherche + rapport

```bash
python run_research.py                      # données synthétiques (déterministe, reproductible)
python run_research.py --live               # données réelles via yfinance (+ cache)
python run_research.py --live --end 2026-06-17   # date de fin épinglée (reproductibilité stricte)
```

Produit la courbe d'equity, le drawdown, l'exposition, la contribution par actif et un tableau de métriques dans `outputs/`.

### 2. Stress coûts + rapport de validation final (§6.5)

```bash
python run_validation.py            # synthétique
python run_validation.py --live     # données réelles
```

Affiche la table de sensibilité aux coûts (rejeu aux niveaux `[2, 5, 10]` bps) et le verdict **RETAIN / REJECT** calculé sur le jeu de test **verrouillé** (Deflated Sharpe + stabilité walk-forward).

### 3. Dry-run paper-trading (aperçu d'ordres — n'envoie rien)

```bash
python run_live.py --dry-run                     # synthétique
python run_live.py --dry-run --live              # prix réels via yfinance
python run_live.py --dry-run --asof 2026-06-17   # à une date donnée
```

Exemple de sortie :

```
TSMOM PAPER-TRADING -- ORDER PREVIEW
  MODE       : DRY-RUN (preview only, nothing sent)
  asof       : 2026-06-17   (orders to place NEXT session)
  broker     : local (paper)   data source: yfinance
  equity     : 2,000.00
SYMBOL    TARGET_W  CURRENT_W  SIDE     QTY    EST_PX    NOTIONAL  REASON
EEM         0.1071     0.0000   buy   3.0000    68.56      205.68  rebalance
EFA         0.1686     0.0000   buy   3.0000   103.78      311.34  rebalance
TLT         0.3389     0.0000   buy   7.0000    86.33      604.31  rebalance
(in target, no order: SPY, GLD)
  orders=3  gross_exposure=0.8857  buy_notional=1,121.33  est_cost=0.22
```

### 4. Tests

```bash
pytest -q                       # suite complète (85 tests, ~quelques minutes)
pytest tests/test_live.py -q    # un module
```

---

## Configuration

Tous les paramètres vivent dans [`config.yaml`](config.yaml) et sont chargés/validés dans une dataclass `Config` typée. **Aucune valeur de marché n'est codée en dur** dans le code.

| Paramètre | Défaut | Rôle |
|---|---|---|
| `initial_capital` | `2000` | Capital de départ (échelle de recherche). |
| `universe` | `[SPY, EFA, EEM, TLT, IEF, GLD, DBC]` | Panier multi-actifs (actions, obligations, or, matières premières). |
| `direction` | `long_short` | `long_short` (alpha de crise) ou `long_only`. |
| `rebalance` | `weekly` | `daily` / `weekly` / `monthly`. |
| `lookbacks` | `[21, 63, 126, 252]` | Horizons TSMOM (1/3/6/12 mois), moyennés. |
| `vol_window` | `60` | Fenêtre d'estimation de la volatilité ex-ante. |
| `asset_vol_target` | `0.10` | Cible de vol annualisée par actif. |
| `portfolio_vol_target` | `0.10` | Cible de vol annualisée du portefeuille. |
| `max_gross_leverage` | `2.0` | Plafond d'exposition brute (`1.0` = pas de levier). |
| `kelly_fraction` | `1.0` | Scalaire de risque global (`<1` = plus conservateur). |
| `cost_bps_per_side` | `2` | Coût aller en points de base. |
| `cost_stress_levels` | `[5, 10]` | Niveaux de coûts pour les tests de sensibilité. |
| `periods_per_year` | `252` | Annualisation. |
| `train_test_ratio` | `0.70` | 30 % de l'historique verrouillé en test out-of-sample. |
| `wf_train_years` / `wf_test_years` / `wf_step_years` | `5 / 1 / 1` | Fenêtres de walk-forward. |
| `cv_embargo` | `0.01` | Embargo (fraction d'échantillons) pour la CV purgée. |
| `seed` | `42` | Graine globale (reproductibilité). |

> Le choix final des paramètres relève d'un **jugement humain** : à relire et ajuster avant tout run sérieux.

---

## Données

`yfinance` (clôtures ajustées) est la source par défaut en mode `--live`, mise en cache sur disque (parquet, dossier `.cache/`). Yahoo étant fréquemment **rate-limité (HTTP 429)**, un **`SyntheticProvider` déterministe** (mouvement brownien géométrique seedé) sert de repli et garantit des runs reproductibles **sans réseau** — c'est aussi ce qu'utilisent tous les tests. Un futur fournisseur payant n'a qu'à implémenter la même interface `DataProvider`.

---

## Protocole de validation (la partie critique)

1. **Train/test verrouillé** — les 30 % les plus récents sont réservés en *out-of-sample*, intouchés jusqu'à la fin.
2. **Walk-forward** — fenêtres glissantes (entraînement 5 ans / test 1 an / pas 1 an) ; la stabilité d'une fenêtre à l'autre est le signal de robustesse.
3. **Validation croisée purgée + embargo** — purge des échantillons d'entraînement chevauchant la fenêtre de test, embargo pour bloquer la fuite d'information.
4. **Correction tests-multiples** — un compteur de *trials* alimente le **Deflated Sharpe** : plus on teste de configs, plus le seuil de significativité monte.
5. **Critère §6.5** — une variante n'est retenue que si, **sur le jeu de test verrouillé**, le Deflated Sharpe est nettement positif **et** la performance walk-forward est stable. Sinon → `REJECT` (on ne re-règle jamais sur le jeu de test).

---

## Paper trading — du dry-run à Alpaca

L'étape actuelle est un **dry-run** (aperçu d'ordres) — répétition d'ingénierie, pas un déploiement. Pour soumettre réellement à **Alpaca paper** (jalon suivant) :

```bash
export APCA_API_KEY_ID=...        # clés d'un compte Alpaca *paper*
export APCA_API_SECRET_KEY=...
python run_live.py --no-dry-run --broker alpaca --live
```

Garde-fous : `--dry-run` est le défaut et n'appelle jamais `submit_order` ; `--no-dry-run --broker local` est refusé. Restent à construire pour une vraie boucle paper : ordonnancement hebdomadaire, réconciliation des positions/fills réels, gestion de la disponibilité à l'emprunt pour la vente à découvert, monitoring/kill-switch. **N'utilisez jamais d'argent réel sans validation §6.5 positive.**

---

## Licence

À définir par le propriétaire du dépôt. Fourni « tel quel », sans aucune garantie ; usage de recherche uniquement.
