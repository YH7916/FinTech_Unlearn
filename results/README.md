# 实验结果

`python run_experiment.py` 默认把所有产物写到 `outputs/`（已被 `.gitignore` 忽略，避免大文件进仓库）。

跑完后，请把 **最终的结果表与图** 拷到本目录 `results/` 并提交，供报告引用与复现：

- `unlearning_results.csv` —— 五模型 × {Forget, Retain, MIA, ZRF} 对比表
- `fig_accuracy_comparison.png` —— 准确率对比图
- `fig_privacy_metrics.png` —— MIA / ZRF 指标对比图
- `experiment_meta.json` —— 运行配置与环境信息

模型权重 `*.pt`（base / gold / amnesiac / badteacher / retain_aware）体积大，**留在 `outputs/` 不入库**。

拷贝示例：

```bash
cp outputs/unlearning_results.csv outputs/fig_*.png outputs/experiment_meta.json results/
git add results && git commit -m "exp: 实验结果" && git push
```
