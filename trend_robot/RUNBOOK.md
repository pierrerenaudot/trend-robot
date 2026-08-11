# Runbook — lancer, tester et analyser le robot TSMOM

Guide pas-à-pas pour piloter le robot de A à Z et **interpréter les résultats**.
Toutes les commandes se lancent depuis la racine du projet (`trend_robot/`), venv activé.

> ⚠️ Rappel de cadrage : ceci est un système de **recherche**. Le mode paper-trading
> est une **répétition d'ingénierie**, pas un feu vert. La décision finale de retenir
> une variante est un **jugement humain** — voir l'étape 6.

---

## Carte des points d'entrée

| Commande | À quoi ça sert |
|---|---|
| `run_research.py` | Backtest de bout en bout + rapport (graphiques + métriques) |
| `run_validation.py` | Stress coûts + verdict §6.5 sur le jeu de test verrouillé |
| `run_live.py` | Paper-trading : aperçu d'ordres (`--dry-run`) ou soumission (`--no-dry-run`) |
| `run_holdout.py` | Lecture §6.5 sur le hold-out **forward pristine** (le vrai juge OOS) |
| `pytest -q` | Suite de tests (126) |

---

## 0. Installation (une fois)

```bash
cd /chemin/vers/trend_robot
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Vérifier que tout est sain :

```bash
pytest -q          # attendu : 126 passed (quelques minutes)
```

---

## 1. Backtest de recherche

```bash
python run_research.py --live --end 2026-06-17      # données réelles (yfinance + cache)
# ou, sans réseau / 100% reproductible :
python run_research.py
```

**Ce qui est produit** (dossier `outputs/`) :

| Fichier | Contenu |
|---|---|
| `equity_curve.png` | Courbe de capital nette de coûts |
| `drawdown.png` | Pertes depuis le plus haut (sous l'eau) |
| `exposure.png` | Exposition brute dans le temps |
| `contribution.png` | Contribution par actif |
| `metrics_table.csv` / `.html` | Toutes les métriques |

### Comment lire les métriques (résumé imprimé + `metrics_table`)

| Métrique | Lecture rapide |
|---|---|
| **CAGR** | Rendement annualisé composé. |
| **Annual vol** | Doit être proche de la cible (10 %). |
| **Sharpe** | Rendement/risque. < 0,5 faible · 0,5–1 correct · > 1 bon (sur données réelles, méfiance si > 2). |
| **Sortino** | Comme Sharpe mais ne pénalise que la baisse. |
| **Calmar / MAR** | CAGR ÷ |max drawdown|. > 0,5 plutôt sain. |
| **Max drawdown (+ durée)** | Pire perte et sa durée en barres. La question : « est-ce que je tiendrais ça ? » |
| **Profit factor** | Gains ÷ pertes. > 1 = profitable. |
| **Hit rate** | % de barres positives (souvent ~50 % en trend-following, c'est normal). |
| **Avg ann turnover** | Rotation annuelle → proxy des coûts. |
| **Avg gross expo** | Exposition brute moyenne (≤ `max_gross_leverage`). |
| **Total cost** | Coût cumulé. |
| **Deflated Sharpe** | Proba que le Sharpe soit vraiment > 0 après correction tests-multiples + non-normalité. **Ici à `n_trials=1`** (non corrigé du nombre de configs essayées — voir étape 2). |
| **Per-asset P&L** | ⚠️ Si un seul actif porte tout → **fragilité** (concentration). |

---

## 2. Validation (le juge) — coûts + §6.5

```bash
python run_validation.py --live --end 2026-06-17 --n-trials 6
```

`--n-trials` = nombre **honnête** de configurations essayées pour aboutir à la variante
courante (plus on a cherché, plus la barre monte). Pour la config actuelle, 6.

### A. Table de sensibilité aux coûts
Rejoue le backtest à 2 / 5 / 10 bps. **Critère** : la stratégie doit rester décente à
**10 bps**. Si le Sharpe s'effondre dès 5 bps → fragile.

### B. Rapport §6.5 (jeu de test verrouillé)
Deux jambes, **les deux** requises pour un `RETAIN` :

| Jambe | Critère de réussite |
|---|---|
| **(a) Deflated Sharpe** | DSR **> 0,60** (au `n_trials` honnête). |
| **(b) Stabilité walk-forward** | Fenêtres positives **≥ 50 %** ET dispersion (cv) **≤ 2,0**. |

→ Verdict **RETAIN** seulement si (a) ET (b). Sinon **REJECT (ne pas déployer)**.

> ⚠️ Le verdict de `run_validation.py` est calculé sur des données **déjà vues** pendant
> la recherche → à considérer comme indicatif. Le verdict propre vient de l'étape 6.
> **Ne jamais re-régler les paramètres en regardant ce verdict** (sur-apprentissage).

---

## 3. Paper-trading — aperçu (n'envoie rien)

```bash
python run_live.py --dry-run --live
```

**Comment lire la table d'ordres** :

```
SYMBOL  TARGET_W  CURRENT_W  SIDE   QTY   EST_PX   NOTIONAL  REASON
EEM       0.1071    0.0000   buy  3.0000  68.56     205.68   rebalance
(in target, no order: SPY, GLD)
  orders=3  gross_exposure=0.8857  buy_notional=1,121.33  est_cost=0.22
```

- `TARGET_W` = poids visé · `CURRENT_W` = poids actuel · `SIDE/QTY` = ordre à passer.
- `REASON` : `rebalance` (ajustement) ou `close` (sortie à plat).
- `(in target, no order: …)` : actifs visés mais dont l'ordre arrondit à 0 part.
- `gross_exposure` doit rester ≤ `max_gross_leverage` (2.0).

---

## 4. Paper-trading — soumission réelle (paper)

Voir le guide dédié **[ALPACA_SETUP.md](ALPACA_SETUP.md)** (compte, clés, env, planificateur).

```bash
python run_live.py --status --broker alpaca           # test de connexion (n'envoie rien)
python run_live.py --no-dry-run --broker alpaca --live  # soumet (1 trade/mois, idempotent)
```

Sûr à exécuter quotidiennement : garde-fous idempotence (1 trade/mois), anti-dérive de
config, marché ouvert, capture des résultats dans `live_state/`.

---

## 5. Tester dans la durée (accumulation forward)

La cadence est **mensuelle** : le job se déclenche, passe quelques ordres, et c'est tout.
Rien à surveiller au quotidien. Le temps calendaire est le seul coût (voir étape 6).

`live_state/` contient l'historique des runs (ordres, statuts, période). C'est la trace
d'audit à consulter après chaque rebalancement.

---

## 6. Analyser le verdict propre — hold-out forward pristine

C'est **le** juge sans contamination : il n'évalue que les barres postérieures à la
date de décision gelée (`decision_record.json`).

```bash
python run_holdout.py --live --end <aujourd'hui>                 # forward (pristine)
python run_holdout.py --live --mode retrospective --retrospective-months 13   # indicatif (NON-pristine)
```

**Comment lire** :
- **Bannière PRISTINE FORWARD** = preuve OOS réelle. **NON-PRISTINE/RETROSPECTIVE** = indicatif seulement.
- `Bars : X/252` = progression de l'accumulation. Tant que `< 252` → `INSUFFICIENT`
  (« accrue more data ») : **c'est normal**, pas un échec.
- Une fois suffisant : DSR à `n_trials=1` (test pré-enregistré unique) **et** `n_trials=6`
  (conservateur). Le verdict `RETAIN` exige DSR > 0,60 **et** stabilité (si évaluable).

**Calendrier** : ~**1 an** de paper-trading mensuel → première lecture pristine (DSR).
**2–3 ans** → confiance. Effort humain ≈ nul.

---

## 7. Cadre de décision (jugement humain — §11)

Avant de considérer une variante comme « validée » :

- ✅ DSR > 0,60 au `n_trials` honnête, **sur le hold-out forward** (pas seulement rétrospectif).
- ✅ Walk-forward stable (≥ 50 % de fenêtres positives, dispersion ≤ 2,0).
- ✅ Survit aux coûts à 10 bps.
- ✅ P&L pas porté par un seul actif (pas de concentration).
- ✅ Max drawdown supportable psychologiquement.
- 🚩 Drapeaux rouges : Sharpe « trop beau » (> 2 sur données réelles), une seule fenêtre
  qui porte tout, effondrement dès 5 bps, dérive de config depuis le gel.

**Règles d'or** : ne jamais re-régler sur le jeu de test ; compter honnêtement `n_trials` ;
à chaque ajout de complexité, se demander si les données le justifient.

---

## 8. Dépannage

| Symptôme | Cause / solution |
|---|---|
| `HTTP 429` / pas de données yfinance | Yahoo rate-limite. Le cache (`.cache/`) et le repli synthétique prennent le relais ; réessayer plus tard ou enlever `--live`. |
| Suite de tests lente (~7 min) | Normal (backtests walk-forward). Pour aller vite : cibler un fichier, ex. `pytest tests/test_live.py -q`. |
| `run_holdout` dit « insufficient » | Normal tant que < 252 barres forward. Laisser le paper-trading accumuler. |
| `AlpacaBroker requires API credentials` | Variables `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY` non définies (voir ALPACA_SETUP.md). |
| Soumission live refusée « CONFIG DRIFT » | `config.yaml` ne correspond plus au `decision_record.json` gelé. Re-geler la décision ou restaurer la config. |
| Soumission live refusée « DATA-INTEGRITY GATE » | Données synthétiques (Yahoo en panne/429) ou périmées : le robot refuse de trader dessus. Rien à faire — le job quotidien réessaiera avec des données fraîches. |
| Run GitHub **rouge** « RECONCILIATION ANOMALY » | La période est marquée tradée mais le book broker est à plat/décalé (reset du compte, fills rejetés, liquidation manuelle). Vérifier le compte Alpaca, puis relancer le workflow avec la case **force** cochée (ou `--force` en local) pour réparer. |

---

## Aide-mémoire

```bash
pytest -q                                              # tests
python run_research.py --live                          # backtest + rapport (outputs/)
python run_validation.py --live --n-trials 6           # coûts + §6.5 (indicatif)
python run_live.py --dry-run --live                    # aperçu d'ordres
python run_live.py --no-dry-run --broker alpaca --live # paper (réel)
python run_holdout.py --live                           # verdict pristine forward
```
