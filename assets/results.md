| Environment | Steps | Ours (final eval return) | SB3 PPO (final eval return) | Ours train time / run |
|---|---|---|---|---|
| CartPole-v1 | 500k | 500 ± 0 | 445 ± 71 | 0.6 min |
| LunarLander-v3 | 1000k | 272 ± 8 | 272 ± 3 | 1.8 min |
| HalfCheetah-v5 | 1000k | 1856 ± 139 | 2404 ± 1314 | 6.7 min |
| Hopper-v5 | 1000k | 3321 ± 505 | 3054 ± 619 | 5.5 min |

Per-seed final evaluation returns (20 deterministic episodes each):

- CartPole-v1: ours [500, 500, 500, 500, 500]; SB3 [500, 371, 490, 363, 500]
- LunarLander-v3: ours [260, 276, 279, 267, 279]; SB3 [274, 273, 268, 275, 272]
- HalfCheetah-v5: ours [1824, 1844, 1998, 1647, 1965]; SB3 [1780, 1716, 4750, 1836, 1936]
- Hopper-v5: ours [3537, 3524, 3345, 3747, 2454]; SB3 [3316, 2911, 3667, 3321, 2055]

**LunarLander-v3** (5 seeds each; final deterministic-evaluation return, mean ± std across seeds; *p* is an exact two-sided permutation test against the default; "Train AUC" is the mean training return over the whole run, a measure of sample efficiency):

| Variant | Eval return | Δ vs default | p | Train AUC | p (AUC) | Per-seed eval |
|---|---|---|---|---|---|---|
| default | 272 ± 8 | – | – | 142 ± 18 | – | [260, 276, 279, 267, 279] |
| no obs. normalisation | 261 ± 8 | -11 | 0.063 | 156 ± 11 | 0.183 | [258, 261, 252, 264, 272] |
| no reward scaling | 174 ± 19 | -98 | 0.008 | 56 ± 2 | 0.008 | [143, 186, 192, 176, 174] |
| no advantage normalisation | 233 ± 21 | -39 | 0.008 | 62 ± 5 | 0.008 | [251, 228, 202, 232, 254] |
| no value-loss clipping | 274 ± 6 | +1 | 0.786 | 142 ± 14 | 0.905 | [279, 264, 277, 275, 273] |
| default (Kaiming) init | 270 ± 7 | -2 | 0.627 | 142 ± 23 | 1.000 | [259, 273, 273, 268, 277] |
| no time-limit bootstrap | 275 ± 2 | +3 | 0.476 | 147 ± 22 | 0.651 | [275, 278, 276, 276, 273] |
| no reward scaling *and* no value-loss clipping | 267 ± 4 | -5 | 0.230 (vs. no reward scaling alone: 0.008) | 140 ± 14 | 0.913 | [272, 262, 264, 267, 272] |

**HalfCheetah-v5** (5 seeds each; final deterministic-evaluation return, mean ± std across seeds; *p* is an exact two-sided permutation test against the default; "Train AUC" is the mean training return over the whole run, a measure of sample efficiency):

| Variant | Eval return | Δ vs default | p | Train AUC | p (AUC) | Per-seed eval |
|---|---|---|---|---|---|---|
| default | 1856 ± 139 | – | – | 1326 ± 41 | – | [1824, 1844, 1998, 1647, 1965] |
| no obs. normalisation | 1855 ± 526 | -1 | 1.000 | 933 ± 134 | 0.008 | [1602, 1367, 1965, 1626, 2715] |
| no reward scaling | 1756 ± 157 | -99 | 0.286 | 1160 ± 74 | 0.008 | [1810, 1931, 1839, 1530, 1672] |
| no advantage normalisation | 1851 ± 64 | -4 | 0.929 | 1294 ± 35 | 0.238 | [1949, 1841, 1819, 1779, 1866] |
| no value-loss clipping | 2060 ± 628 | +204 | 0.794 | 1306 ± 220 | 0.690 | [1873, 3167, 1839, 1826, 1595] |
| default (Kaiming) init | 2862 ± 1509 | +1007 | 0.373 | 1678 ± 623 | 0.397 | [4531, 1842, 1833, 4494, 1612] |
| no time-limit bootstrap | 2695 ± 1344 | +840 | 0.413 | 1638 ± 692 | 0.444 | [1818, 1776, 3979, 4336, 1568] |
