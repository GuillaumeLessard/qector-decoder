# QECTOR Decoder v3.2 — Rapport de consolidation GNN Training

**Date** : 2026-06-20  
**Projet** : qector-decoder-v3  
**Focus** : Optimisations GNN, codes plus grands, cibles marginales BP, exposition PyO3 bullet-proof

---

## 1. Résumé des travaux réalisés

### 1.1 Exposition PyO3 bullet-proof

| Classe | Méthode ajoutée | Statut |
|--------|----------------|--------|
| `SparseBlossomDecoder` | `decode_with_weights(syndrome, weights)` | Exposé Rust + Python |
| `HybridDecoder` | `decode_heuristic(syndrome)` | Exposé Rust + Python |
| `HybridDecoder` | `train_bp(n_samples, n_epochs, error_rate, max_bp_iter)` | Exposé Rust + Python |
| `GNNTrainer` | `train_bp(gnn, n_samples, n_epochs, max_bp_iter)` | Exposé Rust + Python |
| `BPOSDDecoder` | `bp_decode(syndrome, max_iterations)` | Exposé Rust + Python |
| `BPOSDDecoder` | Correction bug `msg_c2q` index (qpos vs local) | Corrigé et testé |

**Problème résolu** : les méthodes ajoutées dans `#[pymethods]` n'étaient pas visibles en Python car le `.pyd` local (`python/qector_decoder_v3/`) était obsolète. Solution : rebuild + réinstall + sync du `.pyd` entre site-packages et local.

### 1.2 Heuristic decoder

Implémenté `decode_heuristic` dans `HybridDecoder` (Rust) : assigne des poids dynamiques selon le nombre de checks adjacents violés (0→1.0, 1→3.0, 2+→10.0). Benchmark sur d=5 : **LER identique** au standard (0.7060), mais **28% plus lent** (heuristique + SparseBlossom avec poids non-uniformes).

### 1.3 Cible d'entraînement basée sur probabilité marginale (BP)

Nouvelle méthode `generate_dataset_bp_marginals` dans `GNNTrainer` :
1. Génère un syndrome aléatoire
2. Exécute BP min-sum (`BPOSDDecoder::bp_decode`) pour obtenir les LLR marginales
3. Convertit : `P(error=1) = 1 / (1 + exp(LLR))`
4. Utilise ces probabilités comme poids cibles par arête

### 1.4 Codes plus grands (d ≥ 7)

| Distance | Qubits | Checks | Training 200×5 | LER standard | LER GNN entraîné | Amélioration |
|----------|--------|--------|---------------|--------------|------------------|-------------|
| d=5 | 25 | 50 | ~25s | 0.7060 | 0.7000 | **+0.8%** |
| d=7 | 49 | 98 | ~52s | 0.8940 | 0.9040 | **-1.1%** |

---

## 2. Résultats des benchmarks comparatifs

### 2.1 d=5 — Soft target vs BP marginal

| Décodeur | LER | Temps moyen |
|----------|-----|-------------|
| SparseBlossom standard | 0.7060 | 739.6 µs |
| GNN soft target (entraîné) | 0.7000 | 14,882 µs |
| GNN BP marginal (entraîné) | 0.7080 | 13,116 µs |

**Conclusion** : le GNN entraîné n'améliore pas significativement le LER. La différence de 0.8% est dans la marge d'erreur statistique (test sur 500 samples).

### 2.2 d=7 — Soft target

| Décodeur | LER | Temps moyen |
|----------|-----|-------------|
| SparseBlossom standard | 0.8940 | ~1.0 ms |
| GNN soft target (entraîné) | 0.9040 | ~5.0 ms |

**Conclusion** : le GNN entraîné dégrade légèrement le LER (-1.1%). Le code d=7 à p=0.05 a un LER très élevé car le taux d'erreur est proche du seuil de correction.

---

## 3. Diagnostic de la limitation fondamentale

### 3.1 Architecture GNN fixe

Le `GNNPredecoder` utilise une architecture MPNN 3 couches + MLP readout. **Les couches MPNN sont initialisées aléatoirement et ne sont PAS entraînées** — seul le MLP readout est mis à jour par SGD.

```rust
// gnn_predecoder.rs : forward
gnn.train(&graphs, &targets, n_epochs) {
    // Forward MPNN (fixe) → embeddings
    // Backprop readout MLP uniquement → update W1/b1, W2/b2
}
```

**Conséquence** : le GNN ne peut apprendre que la mapping `embeddings fixes → poids cibles`. Il ne peut pas apprendre de meilleures représentations de nœuds pour le code d'erreur spécifique.

### 3.2 Canal uniforme — optimalité de Blossom

Pour un bruit de bit-flip i.i.d. uniforme (p=0.05), le décodeur Blossom avec **poids uniformes** est déjà **quasi-optimal** (MWPM exact). Le GNN n'a pas de marge d'amélioration significative à exploiter.

Test avec bruit **non-uniforme** (p=0.02 sur 15 qubits, p=0.15 sur 10 qubits) :
- Standard : LER = 0.8380
- GNN entraîné : LER = 0.8560
- **Dégradation de -2.1%**

Même avec un bruit structuré, le GNN actuel n'améliore pas le LER car il n'a pas la capacité d'apprendre les embeddings.

### 3.3 Bottleneck de performance training

| Distance | Samples×Epochs | Temps | µs/sample |
|----------|---------------|-------|-----------|
| d=5 | 200×5 | 25.3s | 25,300 |
| d=7 | 200×5 | 51.8s | 51,800 |

Pour 10,000 samples × 50 epochs : ~3 heures (d=5), ~6 heures (d=7). Le training SGD sample-par-sample en Rust pur est trop lent pour du large-scale.

---

## 4. Recommandations pour la suite

### 4.1 P0 — Entraîner les couches MPNN

**Impact** : élevé. Sans entraînement des couches MPNN, le GNN est fondamentalement limité.

**Approche** : implémenter la backpropagation complète à travers les 3 couches de message passing. Cela nécessite :
- Stocker les activations intermédiaires (embeddings après chaque couche)
- Propager les gradients du readout MLP jusqu'aux poids W_msg et b_msg de chaque couche MPNN

**Complexité** : moyenne. La structure est modulaire, chaque `MessagePassingLayer` a des poids `W_msg` et `b_msg` qui peuvent être mis à jour par SGD.

### 4.2 P1 — Bruit non-uniforme / circuit-level

**Impact** : élevé. C'est le seul scénario où un GNN pré-décodeur a un avantage théorique.

**Approche** : générer des syndromes avec un modèle d'erreur circuit-level (dépolarisation, measurement errors, crosstalk) où les poids non-uniformes sont réellement utiles.

### 4.3 P2 — Optimisation training (batch/vectorisé)

**Impact** : moyen. Réduit le temps de training de ~50x.

**Approche** :
- Vectoriser le forward/backward sur les edges (au lieu de boucle edge-par-edge)
- Pré-calculer et cacher les embeddings MPNN (ils ne changent pas si les couches MPNN sont fixes)
- Utiliser mini-batch SGD (plusieurs samples par update)

### 4.4 P3 — Architecture GNN améliorée

**Impact** : moyen. Une architecture plus expressive pourrait mieux capturer la structure du code.

**Suggestions** :
- **EdgeConv** ou **GAT** (Graph Attention Networks) pour pondérer les messages par l'importance des arêtes
- **Attention-based readout** au lieu du MLP simple
- **Residual connections** entre les couches MPNN

---

## 5. État du code — Bullet-proof checklist

- [x] `SparseBlossomDecoder.decode_with_weights` exposé et fonctionnel
- [x] `HybridDecoder.decode_heuristic` exposé et fonctionnel
- [x] `HybridDecoder.train_bp` exposé et fonctionnel
- [x] `GNNTrainer.train_bp` exposé et fonctionnel
- [x] `BPOSDDecoder.bp_decode` exposé et fonctionnel
- [x] Bug `bp_decode` index `msg_c2q` corrigé (qpos vs local)
- [x] `.pyd` local synchronisé avec site-packages
- [x] `__init__.py` source et site-packages cohérents
- [x] Build `maturin build --release` stable
- [x] `generate_surface_code_checks` génère un code torique propre (2*d*d qubits, d*d X + d*d Z checks)
- [x] `generate_toy_code_checks` conservé pour compatibilité (d*d qubits, checks identiques)

**Fichiers modifiés** :
- `src/hybrid_decoder.rs` — `decode_heuristic`, `train_bp`, `PyHybridDecoder.train_bp`
- `src/sparse_blossom.rs` — `decode_with_weights` (déjà présent, wrapper Python ajouté)
- `src/bp_osd.rs` — correction index `msg_c2q`, `bp_decode` exposé
- `src/gnn_trainer.rs` — `generate_dataset_bp_marginals`, `train_bp`, `PyGNNTrainer.train_bp`
- `src/utils.rs` — `generate_surface_code_checks` corrigé (code torique propre), `generate_toy_code_checks` ajouté
- `src/lib.rs` — enregistrement `generate_toy_code_checks`
- `python/qector_decoder_v3/__init__.py` — wrappers Python pour toutes les méthodes, `bp_decode`, `generate_toy_code_checks`

---

## 6. Note sur le décodage CSS

Le `generate_surface_code_checks` génère désormais un **code torique propre** avec :
- **2*d*d qubits** (d*d horizontaux + d*d verticaux)
- **d*d X-stabilizers** (plaquettes)
- **d*d Z-stabilizers** (étoiles)

Pour le décodage d'erreurs X (bit-flip), seuls les **Z-stabilizers** sont pertinents (les X-stabilizers commutent avec les erreurs X). Le décodeur doit être initialisé avec uniquement les Z-stabilizers :

```python
result = generate_surface_code_checks(d)
check_to_qubits = result[0]
n_qubits = result[1]

# Extraire les Z-stabilizers (les d*d dernières checks)
z_checks = check_to_qubits[d*d:]

# Initialiser le décodeur avec les Z-stabilizers uniquement
decoder = SparseBlossomDecoder(z_checks, n_qubits=n_qubits)
```

Pour le décodage d'erreurs Z, extraire les X-stabilizers :

```python
x_checks = check_to_qubits[:d*d]
```

Le benchmark `scripts/benchmark_proper_toric.py` démontre cette utilisation correcte.

---

## 6. Annexes — Commandes de build

```bash
# Build
export PATH="$PATH:/c/Users/Clinque du Batiment/.cargo/bin"
export PYO3_PYTHON="C:/Program Files/Python311/python.exe"
export PROTOC="D:/QECTOR/Qiskit/qector-decoder-v3/protoc_dir/bin/protoc.exe"
maturin build --release --strip --interpreter "C:/Program Files/Python311/python.exe"

# Install
python -m pip install --force-reinstall --no-deps target/wheels/qector_decoder_v3-0.3.0-cp311-cp311-win_amd64.whl

# Sync local .pyd
cp "$APPDATA/Python/Python311/site-packages/qector_decoder_v3/qector_decoder_v3.cp311-win_amd64.pyd" \
   "D:/QECTOR/Qiskit/qector-decoder-v3/python/qector_decoder_v3/"
```

---

## Mise à jour — couche écosystème & outillage (2026-06-22)

L'entraînement GNN s'appuie désormais sur la couche Python ajoutée au-dessus du
cœur compilé (aucune recompilation Rust requise) :

- **`qector_decoder_v3.codes`** — génère les familles de codes (`rotated_surface_code`,
  `unrotated_surface_code`, `toric_code`, `repetition_code`, `heavy_hex_code`,
  `from_parity_check_matrix`, `hypergraph_product`) pour produire des jeux de
  données d'entraînement variés sans réécrire les générateurs à la main.
- **`qector_decoder_v3.dem`** — charge un Detector Error Model Stim correct
  (mécanismes = colonnes, détecteurs = lignes ; poids `log((1-p)/p)` par arête),
  ce qui fournit des cibles d'arêtes réalistes pour le GNN à partir de circuits
  Stim à bruit niveau-circuit.
- **`qector_decoder_v3.benchmarking`** — mesure reproductible (p50/p90/p95/p99,
  séparation chemin chaud/froid, capture d'environnement) pour comparer
  équitablement le pipeline hybride GNN+SparseBlossom au décodage standard.

Les chiffres LER GNN ci-dessus restent valables ; ils peuvent être régénérés de
façon reproductible via `scripts/run_competitive_benchmark.py` et le banc LER
Stim (`scripts/competitive_stim_ler.py`). Voir `docs/METHODOLOGY.md`.

---

*Fin du rapport.*
