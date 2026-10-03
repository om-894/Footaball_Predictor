# Results

Full backtest tables behind the summary in the [README](../README.md).

Each test season is scored by models trained on every season before it, with the season
just before held back for early stopping. There are six test seasons, 2020/21 to 2025/26.
The mirror stops in September 2025, so the last one only has its first 40 matches. Lower is
better for every metric. Bold marks the best log score for each target.

Made with `footy build` then `footy evaluate` on the Premier League mirror.

## Forecast mode

Minutes come from the minutes model, as they would for a real forecast. The minutes model
is off by 18.6 minutes on average.

#### Sh

| model | log score | CRPS | Poisson deviance | MAE | ECE |
|---|---|---|---|---|---|
| GlobalMean | 1.0124 | 0.4542 | 1.5850 | 0.8167 | 0.0667 |
| PositionMean | 0.9565 | 0.4247 | 1.3523 | 0.7477 | 0.0763 |
| ShrunkCareerRate | 0.8297 | 0.3493 | 0.9642 | 0.5818 | 0.0647 |
| NaivePer90EWMA | 0.9591 | 0.3666 | 1.2643 | 0.5741 | 0.0443 |
| PlayerEWMA | 0.9437 | 0.3446 | 1.1984 | 0.5419 | 0.0466 |
| PoissonGLM | 0.9622 | 0.4166 | 1.3265 | 0.7129 | 0.0744 |
| NegBinGLM | 0.9615 | 0.4217 | 1.3385 | 0.7206 | 0.0702 |
| **PoissonGBM** | 0.7477 | 0.3248 | 0.7724 | 0.5173 | 0.0493 |
| NegBinMLP | 0.7553 | 0.3278 | 0.8155 | 0.5163 | 0.0304 |

#### SoT

| model | log score | CRPS | Poisson deviance | MAE | ECE |
|---|---|---|---|---|---|
| GlobalMean | 0.5209 | 0.1690 | 0.7802 | 0.3379 | 0.0367 |
| PositionMean | 0.4836 | 0.1599 | 0.6775 | 0.3177 | 0.0301 |
| ShrunkCareerRate | 0.4270 | 0.1426 | 0.5438 | 0.2689 | 0.0259 |
| NaivePer90EWMA | 0.5596 | 0.1529 | 0.8268 | 0.2631 | 0.0306 |
| PlayerEWMA | 0.5510 | 0.1458 | 0.7984 | 0.2544 | 0.0313 |
| PoissonGLM | 0.4826 | 0.1579 | 0.6681 | 0.3011 | 0.0300 |
| NegBinGLM | 0.4828 | 0.1587 | 0.6697 | 0.3024 | 0.0297 |
| **PoissonGBM** | 0.3951 | 0.1373 | 0.4770 | 0.2480 | 0.0181 |
| NegBinMLP | 0.4020 | 0.1387 | 0.4940 | 0.2423 | 0.0167 |

#### Fls

| model | log score | CRPS | Poisson deviance | MAE | ECE |
|---|---|---|---|---|---|
| GlobalMean | 1.1821 | 0.4831 | 1.2937 | 0.7810 | 0.0695 |
| PositionMean | 1.1304 | 0.4574 | 1.1727 | 0.7237 | 0.0513 |
| **ShrunkCareerRate** | 1.1124 | 0.4464 | 1.1294 | 0.7177 | 0.0546 |
| NaivePer90EWMA | 1.3755 | 0.5207 | 1.7784 | 0.8336 | 0.0794 |
| PlayerEWMA | 1.3419 | 0.4774 | 1.6341 | 0.7628 | 0.0530 |
| PoissonGLM | 1.1689 | 0.4763 | 1.2604 | 0.7647 | 0.0690 |
| NegBinGLM | 1.1635 | 0.4827 | 1.2925 | 0.7847 | 0.0533 |
| PoissonGBM | 1.1148 | 0.4480 | 1.1315 | 0.7213 | 0.0634 |
| NegBinMLP | 1.1274 | 0.4552 | 1.1649 | 0.7387 | 0.0473 |

#### Fld

| model | log score | CRPS | Poisson deviance | MAE | ECE |
|---|---|---|---|---|---|
| GlobalMean | 1.1685 | 0.4859 | 1.3620 | 0.7968 | 0.0691 |
| PositionMean | 1.1298 | 0.4633 | 1.2546 | 0.7503 | 0.0519 |
| **ShrunkCareerRate** | 1.0865 | 0.4364 | 1.1419 | 0.7169 | 0.0592 |
| NaivePer90EWMA | 1.3229 | 0.4947 | 1.6945 | 0.7923 | 0.0661 |
| PlayerEWMA | 1.2938 | 0.4591 | 1.5833 | 0.7391 | 0.0614 |
| PoissonGLM | 1.1395 | 0.4687 | 1.2770 | 0.7608 | 0.0529 |
| NegBinGLM | 1.1388 | 0.4757 | 1.2899 | 0.7704 | 0.0536 |
| PoissonGBM | 1.0916 | 0.4394 | 1.1468 | 0.7215 | 0.0659 |
| NegBinMLP | 1.0974 | 0.4440 | 1.1740 | 0.7238 | 0.0369 |

#### CrdY

| model | log score | CRPS | Poisson deviance | MAE | ECE |
|---|---|---|---|---|---|
| GlobalMean | 0.3943 | 0.1131 | 0.5316 | 0.2216 | 0.0270 |
| PositionMean | 0.3902 | 0.1121 | 0.5234 | 0.2193 | 0.0197 |
| ShrunkCareerRate | 0.3898 | 0.1117 | 0.5226 | 0.2222 | 0.0188 |
| NaivePer90EWMA | 0.6227 | 0.1390 | 1.0238 | 0.2591 | 0.0659 |
| PlayerEWMA | 0.6062 | 0.1248 | 0.9612 | 0.2361 | 0.0568 |
| PoissonGLM | 0.3919 | 0.1126 | 0.5269 | 0.2268 | 0.0160 |
| NegBinGLM | 0.3919 | 0.1131 | 0.5268 | 0.2259 | 0.0151 |
| **PoissonGBM** | 0.3858 | 0.1108 | 0.5146 | 0.2295 | 0.0149 |
| NegBinMLP | 0.3916 | 0.1116 | 0.5199 | 0.2258 | 0.0185 |

#### Tkl

| model | log score | CRPS | Poisson deviance | MAE | ECE |
|---|---|---|---|---|---|
| GlobalMean | 1.4857 | 0.7111 | 1.7117 | 1.1056 | 0.0893 |
| PositionMean | 1.3978 | 0.6506 | 1.4587 | 1.0106 | 0.0645 |
| **ShrunkCareerRate** | 1.3647 | 0.6220 | 1.3581 | 0.9641 | 0.0699 |
| NaivePer90EWMA | 1.6432 | 0.7205 | 2.0954 | 1.1142 | 0.0530 |
| PlayerEWMA | 1.6016 | 0.6557 | 1.8976 | 1.0092 | 0.0504 |
| PoissonGLM | 1.4435 | 0.6803 | 1.5972 | 1.0449 | 0.0644 |
| NegBinGLM | 1.4260 | 0.6928 | 1.6461 | 1.0746 | 0.0414 |
| PoissonGBM | 1.3780 | 0.6240 | 1.3472 | 0.9582 | 0.0898 |
| NegBinMLP | 1.3741 | 0.6314 | 1.3963 | 0.9880 | 0.0485 |

## Known-minutes mode

The same models given the minutes each player actually played, which isolates how good the
per-90 rate models are. LightGBM wins every target here.

| target | best model | log score | NegBinMLP | PlayerEWMA (benchmark) | best v benchmark |
|---|---|---|---|---|---|
| Sh | PoissonGBM | 0.681 | 0.709 | 0.883 | 23% better |
| SoT | PoissonGBM | 0.374 | 0.386 | 0.531 | 30% better |
| Fls | PoissonGBM | 1.053 | 1.072 | 1.286 | 18% better |
| Fld | PoissonGBM | 1.010 | 1.035 | 1.220 | 17% better |
| CrdY | PoissonGBM | 0.382 | 0.389 | 0.596 | 36% better |
| Tkl | PoissonGBM | 1.275 | 1.289 | 1.514 | 16% better |

#### Sh

| model | log score | CRPS | Poisson deviance | MAE | ECE |
|---|---|---|---|---|---|
| GlobalMean | 0.9897 | 0.4426 | 1.5043 | 0.7595 | 0.0674 |
| PositionMean | 0.9117 | 0.3990 | 1.2199 | 0.6728 | 0.0396 |
| ShrunkCareerRate | 0.7724 | 0.3139 | 0.8344 | 0.5068 | 0.0241 |
| NaivePer90EWMA | 0.8966 | 0.3136 | 1.0946 | 0.4789 | 0.0280 |
| PlayerEWMA | 0.8828 | 0.2993 | 1.0521 | 0.4597 | 0.0241 |
| PoissonGLM | 0.9166 | 0.3898 | 1.2010 | 0.6409 | 0.0391 |
| NegBinGLM | 0.9140 | 0.3909 | 1.1993 | 0.6422 | 0.0353 |
| **PoissonGBM** | 0.6805 | 0.2813 | 0.6381 | 0.4401 | 0.0148 |
| NegBinMLP | 0.7093 | 0.2943 | 0.6950 | 0.4458 | 0.0369 |

#### SoT

| model | log score | CRPS | Poisson deviance | MAE | ECE |
|---|---|---|---|---|---|
| GlobalMean | 0.5114 | 0.1668 | 0.7548 | 0.3177 | 0.0312 |
| PositionMean | 0.4644 | 0.1538 | 0.6318 | 0.2868 | 0.0161 |
| ShrunkCareerRate | 0.4063 | 0.1346 | 0.5004 | 0.2394 | 0.0104 |
| NaivePer90EWMA | 0.5390 | 0.1397 | 0.7720 | 0.2304 | 0.0250 |
| PlayerEWMA | 0.5313 | 0.1354 | 0.7522 | 0.2259 | 0.0211 |
| PoissonGLM | 0.4635 | 0.1513 | 0.6247 | 0.2721 | 0.0191 |
| NegBinGLM | 0.4632 | 0.1517 | 0.6245 | 0.2726 | 0.0191 |
| **PoissonGBM** | 0.3742 | 0.1284 | 0.4351 | 0.2200 | 0.0146 |
| NegBinMLP | 0.3861 | 0.1315 | 0.4566 | 0.2167 | 0.0271 |

#### Fls

| model | log score | CRPS | Poisson deviance | MAE | ECE |
|---|---|---|---|---|---|
| GlobalMean | 1.1520 | 0.4629 | 1.2226 | 0.7053 | 0.0645 |
| PositionMean | 1.0844 | 0.4290 | 1.0743 | 0.6571 | 0.0332 |
| ShrunkCareerRate | 1.0582 | 0.4132 | 1.0188 | 0.6435 | 0.0281 |
| NaivePer90EWMA | 1.3162 | 0.4621 | 1.5716 | 0.7094 | 0.0905 |
| PlayerEWMA | 1.2860 | 0.4330 | 1.4769 | 0.6653 | 0.0643 |
| PoissonGLM | 1.1110 | 0.4395 | 1.1307 | 0.6861 | 0.0459 |
| NegBinGLM | 1.1100 | 0.4406 | 1.1277 | 0.6876 | 0.0604 |
| **PoissonGBM** | 1.0529 | 0.4109 | 1.0077 | 0.6423 | 0.0295 |
| NegBinMLP | 1.0720 | 0.4194 | 1.0289 | 0.6546 | 0.0442 |

#### Fld

| model | log score | CRPS | Poisson deviance | MAE | ECE |
|---|---|---|---|---|---|
| GlobalMean | 1.1253 | 0.4618 | 1.2526 | 0.7158 | 0.0514 |
| PositionMean | 1.0651 | 0.4265 | 1.1061 | 0.6740 | 0.0228 |
| ShrunkCareerRate | 1.0121 | 0.3936 | 0.9880 | 0.6290 | 0.0185 |
| NaivePer90EWMA | 1.2457 | 0.4325 | 1.4809 | 0.6737 | 0.0607 |
| PlayerEWMA | 1.2195 | 0.4086 | 1.4071 | 0.6402 | 0.0422 |
| PoissonGLM | 1.0690 | 0.4272 | 1.1143 | 0.6742 | 0.0318 |
| NegBinGLM | 1.0646 | 0.4271 | 1.1078 | 0.6732 | 0.0321 |
| **PoissonGBM** | 1.0103 | 0.3932 | 0.9843 | 0.6313 | 0.0169 |
| NegBinMLP | 1.0353 | 0.4061 | 1.0226 | 0.6370 | 0.0459 |

#### CrdY

| model | log score | CRPS | Poisson deviance | MAE | ECE |
|---|---|---|---|---|---|
| GlobalMean | 0.3936 | 0.1125 | 0.5302 | 0.2100 | 0.0250 |
| PositionMean | 0.3891 | 0.1114 | 0.5213 | 0.2073 | 0.0258 |
| ShrunkCareerRate | 0.3874 | 0.1107 | 0.5179 | 0.2090 | 0.0264 |
| NaivePer90EWMA | 0.6115 | 0.1278 | 0.9808 | 0.2297 | 0.0608 |
| PlayerEWMA | 0.5964 | 0.1180 | 0.9361 | 0.2152 | 0.0538 |
| PoissonGLM | 0.3889 | 0.1115 | 0.5209 | 0.2129 | 0.0230 |
| NegBinGLM | 0.3886 | 0.1117 | 0.5202 | 0.2117 | 0.0230 |
| **PoissonGBM** | 0.3822 | 0.1097 | 0.5074 | 0.2150 | 0.0187 |
| NegBinMLP | 0.3886 | 0.1106 | 0.5135 | 0.2118 | 0.0247 |

#### Tkl

| model | log score | CRPS | Poisson deviance | MAE | ECE |
|---|---|---|---|---|---|
| GlobalMean | 1.4193 | 0.6623 | 1.5264 | 1.0150 | 0.0430 |
| PositionMean | 1.3206 | 0.5939 | 1.2683 | 0.8860 | 0.0325 |
| ShrunkCareerRate | 1.2775 | 0.5591 | 1.1597 | 0.8442 | 0.0242 |
| NaivePer90EWMA | 1.5472 | 0.6220 | 1.7716 | 0.9333 | 0.0635 |
| PlayerEWMA | 1.5139 | 0.5792 | 1.6480 | 0.8650 | 0.0440 |
| PoissonGLM | 1.3717 | 0.6238 | 1.3903 | 0.9330 | 0.0655 |
| NegBinGLM | 1.3527 | 0.6171 | 1.3571 | 0.9161 | 0.0730 |
| **PoissonGBM** | 1.2746 | 0.5543 | 1.1402 | 0.8350 | 0.0306 |
| NegBinMLP | 1.2892 | 0.5656 | 1.1703 | 0.8555 | 0.0363 |
