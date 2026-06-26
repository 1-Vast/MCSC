# Same-Protocol Frontier Evidence Report

Scope: same-protocol reproduced-frontier evidence under this repository's exact cold-split protocol.
This report does not use paper-table comparisons and does not support claims beyond reproduced/adapted local baselines.

## Frontier Table

| split | MCSC frozen alpha R2 | DeepDTA delta | GraphDTA compact delta | MolTrans compact delta | XGBoost mean margin | decision |
|---|---:|---|---|---|---:|---|
| DAVIS/target-cold | 0.4938 | +0.0271 [0.0102, 0.042], 7/8 | +0.1820 [0.1551, 0.2067], 8/8 | +0.1425 [0.1178, 0.1661], 8/8 | 0.0101 | PASS |
| DAVIS/family-cold | 0.2915 | +0.0375 [0.0048, 0.0673], 7/8 | +0.0568 [0.0295, 0.0827], 7/8 | +0.0895 [0.0442, 0.1382], 7/8 | 0.0079 | PASS |
| KIBA/target-cold | 0.5168 | +0.1020 [0.0855, 0.1166], 8/8 | +0.1801 [0.1448, 0.215], 8/8 | +2.7343 [0.1957, 7.7546], 8/8 | 0.0496 | PASS |
| KIBA/cluster-cold | 0.3722 | +0.0711 [0.0365, 0.1112], 8/8 | +0.0998 [0.0465, 0.1517], 7/8 | +0.5927 [0.1358, 1.452], 8/8 | 0.0257 | PASS |

## Mechanism Evidence

| split | delta vs prior | delta vs full refiner | harmful-correction reduction | worst-group delta |
|---|---|---|---|---|
| DAVIS/target-cold | +0.0398 [0.0362, 0.043], 8/8 | +0.0052 [0.002, 0.009], 7/8 | +0.0194 [0.0166, 0.0222], 8/8 | +0.0142 [0.0065, 0.0206], 7/8 |
| DAVIS/family-cold | +0.0360 [0.0217, 0.0538], 8/8 | +0.0092 [-0.0317, 0.0481], 5/8 | +0.0183 [0.0154, 0.0211], 8/8 | +0.0544 [-0.0023, 0.1182], 6/8 |
| KIBA/target-cold | +0.0537 [0.0415, 0.0658], 8/8 | +0.0147 [0.0015, 0.0273], 5/8 | +0.0360 [0.032, 0.0412], 8/8 | +0.0934 [0.0478, 0.1514], 7/8 |
| KIBA/cluster-cold | +0.0267 [0.0131, 0.0403], 7/8 | +0.0262 [0.0102, 0.041], 7/8 | +0.0285 [0.0253, 0.0319], 8/8 | +0.1155 [0.0294, 0.2331], 6/8 |

## Claim Boundary

- Supported: MCSC frozen split-level residual alpha outperforms the reproduced local frontier: DeepDTA, compact GraphDTA, compact MolTrans, and XGBoost mean references on all four cold splits.
- Supported mechanism claim: dataset-adaptive target representation plus validation-frozen residual shrinkage reduces the observed refiner self-harm under these splits.
- Not supported: global SOTA claims, paper-table comparisons, or superiority over paper-faithful official GraphDTA/MolTrans/DrugBAN reproductions.
