# Medium no-gold experiment

This directory records the medium-size CUDA experiment used as the current presentation baseline.

## Setup

- Device: CUDA / NVIDIA GeForce RTX 4060 Laptop GPU
- Dataset: CIFAR-100, grouped into CIFAR-100 Super20 labels
- Forget class: `69`
- Forget train samples: `500`
- Retain train samples: `5000`
- Forget validation samples: `100`
- Retain validation samples: `2000`
- Gold model: skipped

Command:

```powershell
python run_experiment.py --skip-gold --out-dir outputs\medium_no_gold --batch-size 128 --base-epochs 3 --amnesiac-epochs 2 --badteacher-epochs 1 --max-forget-train 500 --max-retain-train 5000 --max-forget-valid 100 --max-retain-valid 2000 --force-retrain
```

## Main result

| Method | Forget Acc ↓ | Retain Acc ↑ | MIA ↓ | ZRF ↑ |
| --- | ---: | ---: | ---: | ---: |
| Original | 78.00 | 15.07 | 0.000 | 0.494 |
| Amnesiac | 3.00 | 24.04 | 0.012 | 0.700 |
| Bad Teacher | 68.00 | 15.92 | 0.000 | 0.762 |
| Bad Teacher + Retain (Ours) | 56.00 | 20.31 | 0.000 | 0.734 |

## Conclusion

Compared with the standard Bad Teacher baseline, the retain-aware variant reduces Forget Acc from `68.00%` to `56.00%` and improves Retain Acc from `15.92%` to `20.31%`. This supports the presentation claim that adding a retain-set constraint can improve the trade-off between forgetting the target class and preserving non-forgotten knowledge.

Amnesiac gives the lowest Forget Acc in this run, but it is a stronger retraining-style baseline and has a small non-zero MIA value. For the presentation, the main comparison should focus on `Bad Teacher` versus `Bad Teacher + Retain (Ours)`.
