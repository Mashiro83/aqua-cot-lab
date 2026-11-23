# AQuA 思维链推理实验

本项目对应课程大作业选题二。研究问题是，在同一个生成模型和同一批 AQuA 数学应用题上，多路径采样能否提高选项准确率，算式校验能否改进投票，以及在什么条件下可以少采样几条路径。

[选题说明](<选题二 基于大语言模型的思维链推理.docx>) · [作业提交与评分标准](课程大作业及评分标准.docx)

## 快速运行

实验客户端使用 Python 3.10 或更高版本，仅依赖标准库。启动或接入一个提供兼容 Chat Completions 接口的模型服务，然后在项目根目录配置以下参数。

| 配置 | 用途 |
| --- | --- |
| `COT_MODEL` 或 `--model` | 模型服务对外提供的模型 ID。 |
| `COT_BASE_URL` 或 `--base-url` | 接口前缀，默认 `http://127.0.0.1:8080/v1`。客户端会追加 `/chat/completions`。 |
| `COT_API_KEY` | 接口需要鉴权时使用的 Bearer 密钥。 |

下面的命令适用于云端 Linux。将模型 ID、接口前缀和结果目录替换为实际配置。

```bash
python3 -m cot_lab --help

export COT_MODEL="served-model-id"
export COT_BASE_URL="http://127.0.0.1:8080/v1"
model_results="results/model-directory"

# 用 5 道开发集题目验证完整套件
python3 -m cot_lab --suite full --split dev --limit 5 --output "$model_results/dev-smoke"

# 冻结参数后评测 254 道官方测试题
python3 -m cot_lab --suite full --split test --output "$model_results/test-full-suite"

# 独立执行提前停止，记录实际请求量
python3 -m cot_lab --split dev --limit 6 --mode adaptive --output "$model_results/dev-adaptive"
```

`--suite full` 运行四项实验和 Bonus，共 12 组对照。`--suite core` 是默认选项，只运行基础方法。模型答复从 `choices[0].message.content` 读取，开发集试跑后可查看 `metrics.csv` 的有效答案率，以及 `predictions.jsonl` 中的原始答复。

首次运行会将固定版本的官方数据下载到 `.cache/aqua/`。离线运行可用 `--data-file` 指向对应数据划分的本地 JSONL。模型请求包含格式示例、待测题目和选项，待测题目的标准答案独立用于评分。[AQuA 官方数据](https://github.com/google-deepmind/AQuA/tree/26b64ed1f22208c6742f47df547dfd65088ec30d)

默认设置为 `--paths 5 --min-paths 3 --temperature 0.7 --seed 42`。服务支持并发时，可按其容量设置 `--workers`。端点不接受请求中的 `seed` 时可加 `--no-api-seed`。种子相同仍可能因后端版本、硬件或采样实现产生差异。完整套件还支持 `--debate-rounds`、`--agent-steps` 和 `--knowledge-file`。运行清单记录参数和公式库的 SHA-256。

重复同一条命令可从逐题检查点续跑。模型、参数、数据或选题变更时使用新的输出目录。

## 项目结构

```text
cot_lab/
├── __main__.py                  # 命令行入口
├── experiment.py                # 评测调度、评分、结果写入和断点续跑
├── extended.py                  # 完整套件的实验编排
├── data.py                      # 数据集加载
├── model.py                     # 模型接口与消息格式
├── inference.py                 # 推理调用、路径记录和用量统计
├── reasoning.py                 # 答案解析、投票和算式求值
├── knowledge.py                 # 检索实验和工具 Agent 共用的公式索引
├── math_knowledge.json
└── experiments/
    ├── self_consistency/        # 多路径采样和等权投票
    ├── step_verifier/           # 算式校验、加权投票和提前停止
    ├── retrieval_cot/            # 一次检索和交替检索
    ├── multi_agent_debate/       # 独立、自审和辩论对照
    └── agentic_harness/         # Bonus 工具 Agent 和 Harness

results/                          # 运行时创建
└── <模型目录>/
    └── <运行名>/
```

实验目录按方法组织。Self-Consistency 和算式校验在同一次评测中复用同一批采样路径，保证投票对照只改变聚合规则。切换模型时沿用同一套实验代码，修改 `--model` 或 `COT_MODEL`，按需设置兼容端点，并用 `--output results/<模型目录>/<运行名>` 保存本次结果。模型 ID 如果包含斜杠，可以为它选一个不含斜杠的目录名。

## 四项实验与 Bonus

完整套件使用同一模型和 AQuA 题集，比较以下五部分。默认路径数为 5，辩论包含独立初稿和 1 轮同步修订，每个工具 Agent 最多获得 4 次模型调用。

| 部分 | 方法和对照 |
| --- | --- |
| 1. Self-Consistency | 直接回答、单条思维链、多路径等权投票。 |
| 2. 算式校验器 | 对同一批路径比较等权和加权投票，另统计提前停止的成本与准确率。 |
| 3. 检索增强 CoT | 比较无检索 CoT、一次检索 `retrieval_cot` 和交替检索 `interleaved_cot`。交替检索先读两张卡片，生成一个中间推理步骤和下一条检索式，再融合第二轮结果作答。 |
| 4. 多 Agent 辩论 | 两个角色先独立作答，再分别用同样的预算自审或交换上一轮草稿后修订。使用相同的等权投票规则评估独立、无同伴自审和辩论三个条件。 |
| Bonus. Agentic Harness | 对比 `agent_basic` 和 `agent_harness`。两组共享首个模型动作、工具与模型轮数上限。Agent 可调用精确算式计算器和公式检索，至少成功调用一次工具后才能提交选项。 |

检索使用独立维护的 24 张通用英文数学公式卡，索引和排序使用 BM25。卡片由公式与建模提示组成，不含 AQuA 的题目、答案或解析。两种检索方法最多给最终回答提供四张卡片。二次检索命中不足时用首轮排序补足，原始排序、检索式、二次命中及回填均写入逐题记录。交替检索在 AQuA 上实现一次推理引导的二次检索。[公式库](cot_lab/math_knowledge.json)

辩论角色分别侧重方程建模和替代解法、单位检查。每轮双方只读取完整的上一轮，自审和辩论共享初稿。Harness 在工具接口之外增加重复动作检测、无效动作反馈和剩余预算提示。普通 Agent 遇到无效动作即终止，Harness 可用剩余轮数纠错。两组都记录工具输入、观察、拒绝和终止原因。逐题检查点负责实验中断后的恢复。

## 基础方法

| 方法 | 每题流程 |
| --- | --- |
| `direct` | 温度为 0，直接输出选项。 |
| `cot` | 温度为 0，输出推导、算式和最终选项。 |
| `self_consistency` | 以指定温度采样 K 条 CoT 路径，对有效选项做等权投票。 |
| `verified_vote` | 复用同一批 K 条路径，先校验算式，再按路径权重投票。 |
| `adaptive_vote` | 满足提前停止条件时，使用已生成的路径加权投票。其余题采样到 K 条。 |

直接回答和 CoT 共用一道手写题作为格式示例，分别展示选项和推导的输出格式。普通和加权投票遇到平分时，优先采用更早出现的选项。最终答案须出现在独立末行，例如 `FINAL: C`。缺失、截断或格式不合规的输出记为无效，无效路径不投票，无效题目计入准确率分母。

算式校验只读取模型标为 `CALC:` 的行，以有理数检查四则运算及有限范围的整数幂。例如 `CALC: 1 / 3 ~= 0.33` 使用末位半个单位的舍入容差。路径出现可判定的错误算式时权重为 1。有通过的算式且无可判定错误时权重为 5。其余路径权重为 4。遇到超出支持范围的符号或语法时，该算式标为 `unsupported`。提前停止要求已采样路径数达到下限、所有已采样路径都支持同一个有效选项、至少有一条路径通过算式校验，且没有路径出现可判定的错误算式。

算式正确仅能说明局部数值运算成立。题意建模、推导步骤的承接和算出的数值是否对应所选选项仍需通过逐题记录分析。多条路径也可能因共享同一个生成模型而重复同一错误。建议在 `dev` 上确定实验参数，再冻结参数并运行全量 `test`，同时比较准确率、校验纠正与退化的题数和推理成本。

## 结果文件

| 文件 | 用途 |
| --- | --- |
| `run_config.json` | 记录数据版本、SHA-256、题目 ID、模型和采样参数。 |
| `predictions.jsonl` | 逐题保存标准答案、各组预测、原始输出、算式校验状态和调用统计。 |
| `metrics.csv` | 各方法的准确率、有效答案率、平均采样量、请求数、token 数和调用耗时。 |
| `summary.json` | 记录逐组纠正与退化、采样路径命中率上界、校验覆盖及提前停止统计。 |
| `checkpoints/` | 每题完成后写入的检查点，用于断点续跑。 |

`compare` 模式把采样路径共享给三个多路径方法。提前停止的成本按对应前缀回放计算，实际调用量看 `summary.json` 的 `total`。`adaptive` 模式满足条件即停止发出请求。模型耗时统计为客户端请求时间之和，包含成功生成前的重试。端点未提供 token 用量时，对应聚合值留空。

完整套件另外记录每个方法的平均模型调用数、二次检索命中、辩论一致率和 Agent 终止原因。共享初稿和共享首个动作在方法成本里按该方法独立运行计量，在 `total` 里只记一次实际生成。并行时累计请求耗时可大于整个实验的墙钟耗时。

Git 忽略实验结果、数据缓存、模型权重和本地环境配置。组内汇总时将 `run_config.json`、`metrics.csv` 和 `summary.json` 一并保存，便于把统计结果对应到模型、数据版本和实验参数。

## 参考

- [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171)
- [Making Large Language Models Better Reasoners with Step-Aware Verifier](https://aclanthology.org/2023.acl-long.291/)
- [Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions](https://aclanthology.org/2023.acl-long.557/)
- [Improving Factuality and Reasoning in Language Models through Multiagent Debate](https://arxiv.org/abs/2305.14325)
- [Learn Harness Engineering](https://github.com/walkinglabs/learn-harness-engineering)
