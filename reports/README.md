# 正式实验结果

每个目录对应一次完整运行，按 `reports/<模型目录>/<运行名>/` 组织。三份文件来自同一个运行目录。

| 文件 | 内容 |
| --- | --- |
| `run_config.json` | 模型、数据版本、题目 ID 和实验参数。 |
| `metrics.csv` | 各方法的准确率、有效答案率和推理开销。 |
| `summary.json` | 逐组纠正与退化、检索和辩论统计、提前停止与工具调用统计。 |

在项目根目录执行以下命令。将 `run_path` 改为实际运行路径中 `results/` 之后的部分。

```bash
run_path="model-directory/test-full-suite"
mkdir -p "reports/$run_path" &&
  cp -- "results/$run_path"/{run_config.json,metrics.csv,summary.json} "reports/$run_path/" &&
  git add -- "reports/$run_path"/{run_config.json,metrics.csv,summary.json}
```

原始输出 `predictions.jsonl` 和 `checkpoints/` 留在 `results/`，供续跑和逐题分析。更换模型、数据或参数时使用新的运行目录，便于保留历次正式结果。
