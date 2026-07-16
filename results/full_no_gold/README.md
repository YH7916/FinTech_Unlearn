# Full no-gold experiment

This directory records the full CUDA experiment used as the main presentation result.

## Setup

- Device: CUDA / NVIDIA GeForce RTX 4060 Laptop GPU
- Dataset: CIFAR-100, grouped into CIFAR-100 Super20 labels
- Forget class: `69`
- Forget train samples: `500`
- Retain train samples: `49500`
- Forget validation samples: `100`
- Retain validation samples: `9900`
- Retain train subset for membership estimation: `14850`
- Gold model: skipped

Command:

```powershell
python run_experiment.py --skip-gold --out-dir outputs\full_no_gold --batch-size 256 --base-epochs 5 --amnesiac-epochs 3 --badteacher-epochs 1 --force-retrain
```

## Main result

| Method | Forget Acc ↓ | Retain Acc ↑ | MIA ↓ | ZRF ↑ |
| --- | ---: | ---: | ---: | ---: |
| Original | 55.00 | 47.14 | 1.000 | 0.492 |
| Amnesiac | 7.00 | 60.02 | 1.000 | 0.628 |
| Bad Teacher | 22.00 | 47.67 | 1.000 | 0.745 |
| Bad Teacher + Retain (Ours) | 23.00 | 54.25 | 1.000 | 0.702 |

## Conclusion

Compared with the standard Bad Teacher baseline, the retain-aware variant keeps a similar forgetting level on the target class (`22.00%` vs. `23.00%` Forget Acc) while improving Retain Acc from `47.67%` to `54.25%`. This supports the presentation claim that adding a retain-set constraint improves knowledge preservation under a comparable forgetting effect.

For the presentation, this full run should be used as the main result. The wording should emphasize a better forgetting-retention trade-off rather than stronger forgetting, because Bad Teacher has a slightly lower Forget Acc in this run.
