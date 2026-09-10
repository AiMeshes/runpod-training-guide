# RunPod 最低成本训练指南

> 面向个人独立研究者。覆盖范围：64M–1B 从头预训练、1B–27B midtraining、64M–27B 微调/LoRA、后训练（RLHF / RLAIF / GRPO / OPSD）。
>
> 数据采集于 2026-09。RunPod 价格随供需浮动，**实际下单前务必在 console 核对**。

---

## 0. TL;DR

1. **成本 ≈ 6 × N × D / 有效FLOPS × 单价**。有效 FLOPS（MFU）是第一杠杆，单价是第二。
2. **≤1.5B 用 8×RTX 4090 Spot + 纯 DDP**，比 H100 便宜约 2.3 倍。这是本研究范围的主力配置。
3. **≥3B 必须上 A100/H100**，因为优化器状态放不下 24GB。这是成本断崖。
4. **LoRA 省的是显存，不是算力**。前反向 FLOPs 与全参相同，只在优化器步省。别指望 LoRA 减少计算成本。
5. **后训练的成本在生成，不在训练**。PPO/GRPO 中 70–90% GPU 时间是 rollout。vLLM 吞吐 >> 训练 MFU。
6. **Spot 只在 spot 成本 > $200 时才划算**（见 §7）。小模型实验、微调、LoRA 全部用 **Secure 按需**，不搞编排。
7. **不需要 GitHub Actions**（见 §8.5）。独立研究者无 CI 需求；唯一必须外部完成的事是"发现 Pod 被抢占"，本地 cron 或 $5 VPS 就够。
8. **不要在 Mac 本地 build 镜像**。Apple Silicon 编译 amd64 的 flash-attn 会走 qemu，极慢甚至失败。
9. **最大的两个钱坑**：volume disk 停止后涨到 $0.20/GB/月；忘记 terminate 的 Pod。
10. **先花 $5 校准**，再外推总成本。不要相信本文任何理论数字，包括下面这些表。

---

## 1. 成本模型

```
成本(美元) = 6 × N × D / (GPU数 × 单卡有效FLOPS × 并行效率) × 单价
```

- `N` = 模型参数量
- `D` = 训练 token 数
- `6` = 前向 2 + 反向 4（FLOPs per token per param）
- 单卡有效 FLOPS = 峰值 FLOPS × MFU

等价形式（更好用）：先算 **GPU-小时**，再乘单价。

```
GPU-小时 = 6 × N × D / (有效FLOPS × 3600)
```

### 三个杠杆的量级对比

| 杠杆 | 典型改善幅度 | 备注 |
|---|---|---|
| MFU（25% → 40%） | **-37% 成本** | 靠 FA2 / bf16 / fused optimizer / 合适 batch |
| 单价（Secure → Spot） | **-50% 成本** | 需要抗抢占工程 |
| 卡型（H100 → 4090） | **-57% 成本** | 受限于显存和互联，不总可用 |
| 并行效率（DDP → FSDP over PCIe） | **+40~70% 成本** | 能 DDP 就别 FSDP |

结论：**先把 MFU 做上去，再谈省钱**。一个 MFU 25% 的 H100 配置，比 MFU 40% 的 4090 配置还贵。

---

## 2. GPU 选型表

假设：bf16、FlashAttention-2、真实可达的 MFU（非宣传值）。Spot / Community Cloud 价格。

| GPU | Spot 价 | BF16 峰值 | 实际 MFU | 有效 TFLOPS | 成本效率指数¹ | 显存 | 适用 |
|---|---|---|---|---|---|---|---|
| **RTX 4090** | $0.34 | 330 | 36% | ~120 | **100** | 24G | ≤1.5B DDP |
| RTX 5090 | $0.69 | 420 | 35% | ~145 | 60 | 32G | ≤3B |
| RTX 3090 | $0.22 | 142 | 30% | ~45 | 58 | 24G | 小模型、便宜试错 |
| H100 PCIe | $1.99 | 756 | 40% | ~302 | 43 | 80G | 3B+ |
| H100 SXM | $2.69 | 990 | 40% | ~396 | 42 | 80G | 3B+、多机 |
| A100 SXM | $1.39 | 312 | 40% | ~125 | 25 | 80G | 仅当 H100 无货 |

¹ 每美元有效 FLOPs，归一化到 4090 = 100。数值越高越省钱。

**4090 便宜 2.3 倍，但有两个硬约束：**

- **24GB 显存** → 优化器状态放不下时直接出局（见 §2 显存速算）
- **无 NVLink，PCIe 互联** → FSDP/ZeRO-3 的 all-gather 会成为瓶颈；**DDP 只做梯度 all-reduce，几乎无损**

> **关键实践：能 DDP 就 DDP。** 模型 + 梯度 + 优化器状态全部塞进单卡 24GB 时，用 DDP 而非 FSDP，8 卡扩展效率 ~85%，而 FSDP over PCIe 只有 ~50–60%。

### 显存速算（决定你用不用得起 4090）

全参训练每参数占用（bf16 权重 + bf16 梯度 + fp32 Adam m/v）：

```
每参数字节 ≈ 2 (权重) + 2 (梯度) + 8 (Adam) = 12 B/param
```

| 模型 | 全参状态 | 单卡 24G 能放下？ | 结论 |
|---|---|---|---|
| 64M | 0.8 GB | ✅ 轻松 DDP | 4090 × 8 |
| 350M | 4.2 GB | ✅ DDP | 4090 × 8 |
| 1B | 12 GB | ✅ DDP（紧但可行，需梯度检查点） | 4090 × 8 |
| 3B | 36 GB | ❌ | 需 FSDP / 8-bit Adam |
| 7B | 84 GB | ❌ | A100/H100 × 8 |
| 27B | 324 GB | ❌ | H100 80G × 8（下限 5 卡） |

用 8-bit Adam 可把状态压到 `2+2+2 = 6 B/param`，3B 降到 18GB —— 勉强能进 24GB，但吞吐会掉。**1B 是 4090 的甜点上限。**

---

## 3. 分场景成本估算

**方法**：GPU-小时 = `6 × N × D / (有效FLOPS × 3600)`，再乘单价。下表已算好，供快速查表。所有数字为**估算**，±50% 属正常。

### 3.1 从头预训练：64M – 1B

**配置：8× RTX 4090 Spot，纯 DDP，$2.72/hr**

两种 token 预算口径：
- **Chinchilla 最优** = 20 × params
- **过训练** = 200 × params（小模型实用做法，质量明显更好；参考 SmolLM-135M 用了 600B tokens = 4400×）

| 模型 | D (20×) | 成本 | D (200×) | 成本 | 墙钟(8×4090, 85%) |
|---|---|---|---|---|---|
| 64M | 1.3B | **$0.5** | 12.8B | **$4** | 1.5 h |
| 125M | 2.5B | **$2** | 25B | **$15** | 3 h |
| 350M | 7B | **$14** | 70B | **$124** | 17 h |
| 1B | 20B | **$100** | 200B | **$1,110** | 17 天 |

> **解读**：64M–350M 的从头预训练基本是零成本，随便跑。真正需要预算的是 1B 的过训练版本（约 $1,100），这也是"训一个能用的 1B"的真实价格。

**性价比建议**：与其训练一个 1B / 20B tokens 的欠训练模型，不如训练 350M / 70B tokens。同样 $124，后者通常下游表现更好。

### 3.2 Midtraining：1B – 27B

**配置分档**（由显存决定，见 §2）：

| 模型 | 最低配置 | Spot 单价 |
|---|---|---|
| 1B | 8×4090 DDP | $2.72/hr |
| 3B | 8×A100 80G（或 8×4090 + 8-bit Adam） | $11.12/hr |
| 7B | 8×A100 80G | $11.12/hr |
| 14B | 8×H100 80G | $21.52/hr |
| 27B | 8×H100 80G | $21.52/hr |

**成本表**（H100 SXM Spot，4090 档按 4090 费率单独算）：

| 模型 | 10B tokens | 20B tokens | 50B tokens |
|---|---|---|---|
| 1B (4090) | $113 | $226 | $566 |
| 3B | $340 | $679 | $1,698 |
| 7B | $792 | $1,585 | $3,962 |
| 14B | $1,585 | $3,170 | $7,925 |
| 27B | $3,057 | **$6,114** | **$15,285** |

> **27B midtraining 的真实成本约 $6k–15k**，墙钟 12–30 天（8×H100）。这是本研究范围内最贵的一项。如果要压，只有三条路：
> 1. **LoRA midtraining** —— 省显存，可以跑在 2–4×4090 上。但注意 **FLOPs 不变**，只是单价降了（$0.34 vs $2.69），净省约 4–6 倍，代价是墙钟慢 3–5 倍。
> 2. **缩减 token 预算到 10B** —— 对领域适配通常够用。
> 3. **不做 27B** —— 7B 的 midtraining 只要 $800–4,000，效果差距未必值 8 倍价格。

### 3.3 微调 / LoRA：64M – 27B

**最重要的一条认知：LoRA 不减少计算量。**

LoRA 的前向+反向仍然要过全部参数，FLOPs ≈ 6ND，与全参微调**相同**。LoRA 省的是：
- **优化器状态**（不存 base model 的 Adam m/v）→ 显存大降
- **因此可以用更便宜的卡** → 单价降

所以 LoRA 的钱是这么省的：**不是算得少，而是用便宜的卡算得慢**。

| 场景 | 配置 | 单价 | 说明 |
|---|---|---|---|
| 64M–1B 全参 SFT | 8×4090 DDP | $2.72/hr | |
| 3–7B 全参 SFT | 8×A100 | $11.12/hr | |
| **27B QLoRA (4-bit)** | **1–2×4090** | **$0.34–0.68/hr** | 4-bit base ≈ 14GB，单卡可跑 |
| **27B LoRA (bf16)** | **3–4×4090** | **$1.02–1.36/hr** | base 54GB 需切分 |
| 27B 全参 SFT | 8×H100 | $21.52/hr | 与 midtraining 同价 |

**典型微调成本**（数据集 100k–1M 样本，2–3 epoch）：

| 模型 | 规模 | 配置 | 成本 |
|---|---|---|---|
| 64M–350M | 1M × 1k tok | 8×4090 | **<$5** |
| 1B | 1M × 1k tok | 8×4090 | **~$25** |
| 7B LoRA | 200k × 2k tok | 2×4090 | **~$15** |
| 27B QLoRA | 200k × 2k tok | 2×4090 | **~$120**（墙钟 ~3 周） |
| 27B LoRA | 200k × 2k tok | 4×4090 | **~$70**（墙钟 ~1 周） |
| 27B 全参 | 1M × 1k tok | 8×H100 | **~$1,900** |

> **27B LoRA 建议用 bf16 而非 4-bit**。4-bit 反量化是访存瓶颈，实际吞吐可能只有 bf16 的 1/3–1/4，省下的显存不值损失的时间。多买 2 张 4090 更划算。

### 3.4 后训练：RLHF / RLAIF / GRPO / OPSD

**结构性事实（决定一切）：**

1. **成本大头是 rollout 生成，不是训练。** PPO/GRPO 中 70–90% 的 GPU 时间花在采样上。所以成本杠杆是 **vLLM/SGLang 的吞吐**，不是训练 MFU。
2. **RLHF (PPO) 同时在显存里放 4 个模型**：policy + reference + reward + critic。27B 全参 PPO 需要 16+ 张 H100，个人研究者不现实。
3. **减负路径**：GRPO（去掉 critic）→ DPO/SimPO（去掉 RM 和 critic）→ 规则奖励 RLVR（去掉 RM）。

**显存需求对比（N = 模型参数量）：**

| 方法 | 常驻模型 | bf16 显存 | 27B 可行？ |
|---|---|---|---|
| PPO (RLHF) | policy+ref+RM+critic | ~8N | ❌ 需 16+ 卡 |
| GRPO | policy+ref+RM | ~6N | ⚠️ LoRA 可行 |
| DPO / SimPO | policy+ref | ~4N | ⚠️ 需 6×A100/H100 |
| **GRPO + LoRA + 规则奖励** | **policy+ref(LoRA)** | **~2N** | ✅ **推荐路径** |

**成本估算：**

| 场景 | 配置 | 成本 |
|---|---|---|
| 64M–1B DPO | 2×4090 | **<$5** |
| 1B GRPO（规则奖励，1000 步） | 2×4090 | **~$30** |
| 7B DPO（100k pairs） | 8×A100 | **~$60** |
| 7B GRPO（8 samples × 1k tok，1000 步） | 2×A100 + vLLM | **~$400** |
| 27B DPO（100k pairs） | 6×H100 | **~$700** |
| 27B GRPO + LoRA | 4×H100 | **~$2,500** |
| 27B 全参 PPO | 16+×H100 | **$20k+**（不建议） |

**GRPO 成本拆解示例（7B，1000 步，bs=64 prompts × 8 samples × 1k tokens）：**
- 每步生成 512k tokens，vLLM 在 1×H100 上约 4000 tok/s → 128 s/步
- 1000 步 = 128,000 s = **35.6 H100-小时 ≈ $96**（rollout）
- 策略更新：6 × 7e9 × 5.12e5 = 2.15e16 FLOPs ≈ **15 H100-小时 ≈ $40**
- 奖励模型打分：与 rollout 同量级或更多，**取决于 RM 大小**
- **结论：rollout 是主导项。优化 vLLM 配置比优化训练循环收益大得多。**

**省钱技巧：**

- **奖励模型放 RunPod Serverless（scale-to-zero）**。RM 只在打分时用，独占 GPU 会浪费大量 idle 时间。Serverless 按秒计费，闲置不计费。
- **DPO/SimPO 优先于 PPO**。除非确实需要在线探索，DPO 用 1/4 的成本拿到 80% 的效果。
- **规则奖励（RLVR）替代学习式 RM**。数学/代码/格式类任务完全可以用规则打分，省掉整个 RM 训练和推理成本。
- **缩短 rollout 长度**。成本与生成长度近似线性，把 max_tokens 从 4k 压到 1k 就是 4 倍。

> **关于 OPSD**：本文按 on-policy 自蒸馏（on-policy self-distillation）理解，其成本结构与 GRPO 类似 —— 生成主导，且需要 teacher 的 logits（显存开销 ≈ +2N）。如果指的是别的算法，请告知，我重新算。

---

## 4. 抗抢占工程（Spot 的必备配套）

Spot 比 Secure 便宜约 50%，但**警告时间不可信赖**（各来源说法从"无预警"到"30 秒"都有）。设计原则：**假设随时消失**。

### 4.1 抢占后的状态

- Pod 进入 **stopped** 状态，**容器盘被擦除**
- volume disk 保留，但**计费从 $0.10 涨到 $0.20/GB/月**
- 恢复路径：network volume 上的 checkpoint + 代码 → 重新挂载 → 拉起新 Pod → 从最新 checkpoint 续训

### 4.2 原子写 checkpoint

抢占可能发生在 `torch.save` 中途，留下损坏文件。必须先写临时文件再 rename：

```python
import os, glob, torch

def save_ckpt(state, step, ckpt_dir):
    """原子写。零填充保证 sorted() 顺序正确。"""
    path = f"{ckpt_dir}/step_{step:07d}.pt"
    tmp = path + ".tmp"
    torch.save({
        "step": step,
        "model": state.model.state_dict(),
        "opt": state.opt.state_dict(),
        "sched": state.sched.state_dict(),
        "data_pos": state.data_iter.position(),   # ← 最容易漏的一项
        "rng": torch.get_rng_state(),
    }, tmp)
    os.replace(tmp, path)          # POSIX rename 是原子的

def load_latest(ckpt_dir):
    cks = sorted(glob.glob(f"{ckpt_dir}/step_*.pt"))
    if not cks:
        return None
    return torch.load(cks[-1], map_location="cuda")
```

**`data_pos` 必须存。** 只存模型和优化器是最常见的错误 —— 恢复后数据从 shard 开头重新采样，等于某段数据分布被重复喂了几遍。

### 4.3 自动续训 supervisor

作为容器 entrypoint，训练进程崩溃或 Pod 重启后自动拉起：

```bash
#!/bin/bash
# entrypoint.sh
CKPT_DIR=${CKPT_DIR:-/workspace/ckpt}
mkdir -p "$CKPT_DIR"

while true; do
  torchrun --nproc_per_node=$NUM_GPUS \
           --rdzv_backend=c10d --rdzv_endpoint=localhost:29500 \
           train.py --ckpt-dir "$CKPT_DIR" --resume
  code=$?
  if [ $code -eq 0 ]; then
    echo "[supervisor] training finished cleanly"
    break
  fi
  echo "[supervisor] exit=$code, resuming from latest checkpoint..."
  sleep 15
done
```

配合 RunPod 的 restart policy（Pod 不退出时不会重启容器，所以 supervisor 必须是进程内循环而非依赖平台重启）。

### 4.4 checkpoint 频率怎么定

设抢占率 λ（次/小时）、单次 checkpoint 时间占比 c，最优间隔：

```
T_opt ≈ sqrt(2c / λ)
```

实际取 **每 30–60 分钟一次**较平衡。存太勤浪费时间（大模型一次 checkpoint 可能 30–60 秒），存太疏一次抢占损失几小时。

> 对 27B，一次全量 checkpoint = 54GB 写到 network volume。@ 500MB/s 约 110 秒。若每 30 分钟存一次，开销约 6% —— 可接受。

---

## 5. 存储与计费陷阱

### 5.1 三档存储

| 类型 | 运行中 | 停止后 | 持久性 | 用途 |
|---|---|---|---|---|
| Container disk | $0.10/GB/月 | 停止即擦除 | 临时 | OS、依赖 |
| **Volume disk** | $0.10/GB/月 | **$0.20/GB/月** ⚠️ | 随 Pod | **尽量别用** |
| **Network volume** | $0.07/GB/月 | $0.07/GB/月 | 独立于 Pod | **数据集 + checkpoint** |

> **最大的坑：volume disk 停止后涨价到 $0.20/GB/月，是全平台最贵的存储。**
> 很多人为了省钱"停止"Pod，却留着 500GB volume disk，每月白交 $100。

### 5.2 计费规则

- **按秒计费**，包括**容器启动时间**。每次 `pip install flash-attn` 浪费 10 分钟就是白烧钱。
- **出网流量免费**（ingress/egress 均无费用，但有 fair-use 约束，部分 Community 主机可能收网络费）。
- 账户余额归零时：有 network volume 的 Pod 停止并保留数据；**没有的会被终止且数据不可恢复**。
- 余额不足以覆盖约 10 分钟运行时间时会被提前停止。
- **默认花费上限 $80/小时** —— 对个人来说高得离谱。8×H100 忘关 12 小时就是 $258。**立刻去账户设置调低。**

### 5.3 省钱策略

- **永远 `terminate` 而不是 `stop`**，除非正在等抢占恢复
- **数据集不必常驻 network volume**。1TB = $70/月。既然出网免费，每次从 HuggingFace 重新拉取（几十分钟）在长周期上更划算 —— 前提是你的启动流程能自动化这件事
- **自建镜像**，预装 torch + flash-attn + 训练代码。启动即开跑，不为 `pip install` 付费
- **checkpoint 只保留最近 2–3 个**，旧的删掉。27B 每个 54GB，存 20 个就是 1TB = $70/月

### 5.4 常用命令

```bash
runpodctl get pod                       # 列出所有 Pod 及状态
runpodctl stop pod $POD_ID              # 停机（⚠️ volume disk 涨价）
runpodctl remove pod $POD_ID            # 彻底终止，停止所有计费
runpodctl get volume                    # 检查遗留的 volume（隐性扣费源）
```

> **每周检查一次 `runpodctl get volume`。** 遗忘的 volume disk 是个人研究者最常见的持续失血点。

---

## 6. Savings Plans

- 3 或 6 个月**预付**，约 **10–20% off** on-demand
- **只覆盖 GPU 计算**，存储仍按标准价计费
- Pod 停止后自动应用于同型号 GPU 的下次部署
- **不可退款、不可转让、到期作废**

**建议**：只有在**时间线高度确定**且连续跑 3 个月以上时才考虑。对本研究范围（多为短期实验、随时可能换模型规模），**不要碰**。省 10–20% 换来失去灵活性，不划算。

---

## 7. Spot vs Secure：先决定要不要搞编排

**这一节比"用哪种 CI"重要得多。** 上 Spot 的前提是有一套抢占恢复机制，而维护这套机制有隐性成本。省下的钱必须超过这个成本才划算。

### 7.1 决策阈值

Spot 比 Secure 便宜约 50%。按实际规模算账：

| 工作负载 | Spot | Secure | 省下 | 值不值得搞编排 |
|---|---|---|---|---|
| 64M 从头 | $0.5 | ~$1 | $0.5 | ❌ 荒谬 |
| 350M 从头 | $14 | ~$28 | $14 | ❌ 不值 |
| 微调 / LoRA（全部规模） | $5–120 | ~2× | <$120 | ❌ 不值 |
| 1B 从头 (200×) | $1,110 | ~$2,220 | **$1,110** | ✅ |
| 3B midtraining (20B) | $679 | ~$1,358 | **$679** | ✅ |
| 7B midtraining (20B) | $1,585 | ~$3,170 | **$1,585** | ✅ |
| 27B midtraining (20B) | $6,114 | ~$12,228 | **$6,114** | ✅ |
| 后训练 GRPO 7B | $400 | ~$800 | **$400** | ✅ |
| 后训练 GRPO 27B | $2,500 | ~$5,000 | **$2,500** | ✅ |

> **分界线大约在 spot 成本 $200–300。**

**这意味着大部分日常实验根本不该用 Spot。** 64M–350M 从头预训练、所有微调/LoRA、小规模 DPO —— 全部用 **Secure Cloud 按需**，随开随用，不抢占，**不需要任何外部编排**。多花的几美元到几十美元，远低于搭一套监控基础设施的时间成本。

只有 **1B+ 从头训练、3B+ midtraining、中等规模后训练** 才值得上 Spot + 恢复机制。

### 7.2 4090 的实际抢占频率

常被忽略的一点：**4090 是 Community 上供给最充足的卡**。在低需求期，可能连续几天到几周都不被抢占；高需求期则可能一天数次。

所以即使上了 Spot，实际中断频率也可能远低于预期 —— 一套简单的恢复脚本就够用，不需要复杂的编排系统。

### 7.3 Secure Cloud 也有省法

不用 Spot 不等于不看价格：

- **Community Cloud 的按需价**已经比 Secure 便宜（如 4090 $0.34 vs $0.60–0.69），且部分场景不那么容易抢占
- **Serverless（scale-to-zero）** 用于奖励模型等间歇性负载，闲置不计费
- **不为启动时间付费**：自建镜像，启动即开跑（见 §8.3）

---

## 8. 编排架构：需要什么、不需要什么

### 8.1 结论先行

**独立研究者在 Mac 上工作，最简方案就是最优方案。**

| 工作负载 | 推荐架构 | 需要 CI 吗 |
|---|---|---|
| 小模型实验（< $200） | Secure 按需 + console/CLI 手动 | ❌ |
| 中等训练（$200–2k） | Spot + network volume + Pod 内 supervisor + 本地 cron | ❌ |
| 多周长任务 | 加一个 $5 VPS 做监控 | ❌ |
| 需要多人复现 / 频繁 CI | GitHub Actions | ✅ |

**GitHub Actions 解决的是团队协作和 CI 问题。** 独立研究者这些需求基本不存在，引入它反而增加复杂度。下面对比完整方案，供需要时查。

### 8.2 唯一必须由"外部"完成的事

验证结论：**被抢占的 Pod 不会自己重启，平台不会替你重新拉起。**

数据落在 network volume 上能保住，但"发现 Pod 死了 → 重新创建 → 挂载 volume → 从 checkpoint 续训"这一步**必须有人或系统去做**。

其余都不需要外部编排：

- **checkpoint / 抗崩溃** → Pod 内的 supervisor 循环足够（见 §4.3）
- **代码传递** → `runpodctl send`（一次性码，不需要 API key）或 rsync
- **镜像构建** → 可以完全不自己做（见 §8.3）
- **启动 Pod** → console 点一下，或本地一行命令

所以问题收敛为一个：**"谁来盯着 Pod 死没死"**。

> 注意：如果你**手动** `runpodctl start` 一个被抢占停止的 Pod，容器盘的 entrypoint 会重新执行，supervisor 就会从最新 checkpoint 续训。所以"恢复"本身不难，难的是"发现"。

### 8.3 代码与镜像传递

**代码**：

```bash
# 方式一：runpodctl send（一次性码，无需 API key）
runpodctl send ./code.tar.gz       # 输出如: Code is: 8338-galileo-collect-fidel
# 在 Pod 内：
runpodctl receive 8338-galileo-collect-fidel

# 方式二：rsync（大文件更快，支持增量）
rsync -avz -e "ssh -p $POD_PORT -i ~/.ssh/id_ed25519" \
      ./src/ root@$POD_IP:/workspace/src/
```

**镜像 —— 一个 Mac 特有的坑：不要在本地 build。**

Apple Silicon 上 `docker build` 出 linux/amd64 的 torch 镜像要走 qemu 模拟，编译 flash-attn 这类包可能跑 40 分钟以上甚至失败。三个替代方案：

| 方案 | 成本 | 启动速度 | 推荐度 |
|---|---|---|---|
| **用官方 `runpod/pytorch` 模板** | 0 | 慢几分钟 | ⭐ **起步首选** |
| 在 RunPod 的便宜 Pod 里 build 一次，存成 template | ~$0.3 一次性 | 快 | ⭐ 优化阶段 |
| 远程 buildx（Docker Build Cloud 等） | 额外花钱 | 快 | 不必要 |

> **推荐路径**：先用官方模板把训练流程跑通（确认 pipeline 正确），再考虑自建镜像优化启动。
> **不要一上来就优化启动** —— 那是在优化一个还没跑通的东西。
>
> 成本核算：4090 上启动多花 15 分钟 = $0.085，可以忽略。

### 8.4 恢复方案对比（"谁来盯着"）

| 方案 | 可靠性 | 成本 | 适用 |
|---|---|---|---|
| **手动检查** | 取决于你 | $0 | spot 成本 <$500、跑几天的任务 |
| 本地 launchd / cron | 中（Mac 睡了就停） | $0 | 多数情况够用 |
| **小 VPS cron** | 高 | **$5/月** | 多周长任务 |
| GitHub Actions cron | 中 | $0 | ⚠️ 见下方警告 |

**本地 cron 的风险没想象中大**：Mac 睡着时 Pod 照常在跑，只是抢占后的恢复会被推迟到你唤醒电脑。对跑几天的任务，晚恢复几小时通常可以接受。要保险可以用 `caffeinate`。

**GitHub Actions cron 有两个坑**：
1. 高峰期可能延迟 5–15 分钟
2. **仓库 60 天无活动后定时任务自动禁用** —— 对长期训练项目很致命

> **结论：真要外部监控，用 $5 VPS 而不是 GitHub Actions cron。** 更可靠，也更省心。

### 8.5 如果确实要用 GitHub Actions

分工原则：**Action 只负责"点火"，绝不能持有训练进程。**

硬约束：
- 单 job **6 小时硬上限**（GitHub-hosted runner，`timeout-minutes` 设再大也会被拒）
- 14GB 文档可用磁盘（实际约 20–22GB 空闲），build torch 镜像会撑爆
- 没有 GPU
- **每个 job 被完整计费，即使中途取消**

职责划分：

```
push tag / 手动触发
   ↓
Actions ──┬─→ build & push 镜像到 GHCR
          └─→ 调 RunPod API 创建 Pod（注入 bootstrap 脚本）
                 ↓
              Pod 启动 → 挂 network volume → 解压代码 → torchrun
                 ↓
              supervisor 循环（自包含，抗抢占）
                 ↓
              Action 立即结束（只活 2–5 分钟）
```

成本：**公开仓库的 Actions 和 GHCR 完全免费**，私有仓库 2000 min/月免费额度、超出 $0.006/min。所以**把训练仓库设为 public**（但密钥一律走 GitHub Secrets，不要写进 repo）。

构建前需要清理磁盘以腾出空间：

```yaml
- name: Free disk space
  uses: jlumbroso/free-disk-space@<sha>
  with:
    tool-cache: true
    android: true
    dotnet: true
    haskell: true
    large-packages: false
    docker-images: false      # 保留 docker
    swap-storage: true        # 可腾出约 20–30GB
```

> **API 注意**：`runpod` Python SDK 在 CI 中不会可靠读取本地 `~/.runpod/config.toml`，必须在 workflow 里设 `RUNPOD_API_KEY` 环境变量，且**在 `import runpod` 之前生效**（构造函数在实例化时读凭据）。SDK 迭代很快，参数名（如 `interruptible` / `cloud_type`）下单前用 `runpod.create_pod.__doc__` 确认。

**另外**：如果是 serverless 部署，RunPod Hub 可以直连 GitHub 仓库，push 即自动构建部署，比自写 Action 可靠 —— 这是官方集成。

### 8.6 本地控制面

`runpodctl` 可 brew 安装，能力足够当本地控制面：

```bash
brew install runpod/runpodctl/runpodctl
runpodctl config --apiKey=YOUR_KEY
# 或 runpodctl doctor 做引导式配置

runpodctl gpu list                       # 查可用 GPU 和价格
runpodctl get pod                        # 列出所有 Pod
runpodctl create pod --gpuType "NVIDIA GeForce RTX 4090" \
  --imageName "runpod/pytorch:latest" --name train
runpodctl ssh connect <pod-id>           # 获取 SSH 连接串
runpodctl send ./code.tar.gz             # 传文件，输出一次性码
runpodctl start pod <id>                 # 启动已停止的 Pod（--bid=0.3 走 spot）
runpodctl stop pod <id>                  # 停止（⚠️ volume disk 涨价）
runpodctl remove pod <id>                # 终止
runpodctl network-volume list            # 检查遗留 volume（隐性扣费源）
```

> `runpodctl` 在 Pod 内也预装，使用 pod 作用域的 key。**但不要依赖它做自重启** —— Pod 死的时候它也死了。

---

## 9. 训练数据获取

### 9.1 核心结论：传输完全免费

- **HuggingFace 出网免费**（egress 与 CDN 均不计费）
- **RunPod 入网免费**（ingress/egress 均无费用）

> **推论：永远不要从 Mac 上传公共数据集。** 让 Pod 自己去拉。
>
> 家庭上行带宽是瓶颈：800GB 在 50 Mbps 上行下要传 **35 小时**；RunPod 机房用
> `HF_HUB_ENABLE_HF_TRANSFER=1` 可跑到 ~300 MB/s（**约 45 分钟**）。差 45 倍。

### 9.2 按数据规模选方案

| 用途 | token 数 | 原始文本量¹ | 方案 |
|---|---|---|---|
| 微调 / LoRA / 后训练 / 64M 预训练 | < 13B | **< 50GB** | 启动脚本下载，容器盘够用 |
| 350M 预训练 | 70B | ~280GB | 下载到 network volume |
| midtraining | 10–50B | 40–200GB | 同上 |
| **1B 预训练 (200×)** | 200B | **~800GB** | network volume + **预 tokenize** |

¹ 按 ~4 bytes/token 估算（英文 UTF-8；中文 ~1.5 字/token × 3 bytes，量级相近）

**大部分工作负载（微调 / LoRA / 后训练）数据量是 GB 级**，直接下载即可，无需特殊处理。
真正需要设计的只有 1B 从头预训练那一档。

### 9.3 推荐默认方案：entrypoint 幂等下载

```bash
# entrypoint.sh 的数据准备段
DATA_DIR=/workspace/data

if [ ! -f "$DATA_DIR/.ready" ]; then
    export HF_HUB_ENABLE_HF_TRANSFER=1     # Rust 并行下载，快 5–10 倍
    export HF_TOKEN=$HF_TOKEN              # 匿名访问会被限速，务必带 token

    rm -rf "$DATA_DIR.tmp"
    huggingface-cli download "$DATASET" \
        --repo-type dataset --local-dir "$DATA_DIR.tmp"
    mv "$DATA_DIR.tmp" "$DATA_DIR"         # 同分区 rename，原子
    touch "$DATA_DIR/.ready"
fi
```

三个设计点（与 §4.2 checkpoint 同一思路）：

- **临时目录 + 原子 rename** —— 下载中途被抢占不会留下半份数据
- **`.ready` 哨兵** —— 重启时跳过重复下载和 API 往返
- **下到 `/workspace`（network volume）而非容器盘** —— 抢占后数据仍在

`huggingface-cli download` 本身可续传（只补缺失文件），哨兵主要省每次启动的 API 查询。

**hf_transfer 的两个坑：**

- **快但断点续传弱**。800GB 在不稳定连接上断掉可能从头来。下到 network volume 可缓解（已完成文件保留）
- **部分网络环境下多线程会失败**（有报告 2% 即卡住）。反复失败时 `unset HF_HUB_ENABLE_HF_TRANSFER` 退回单线程——慢但稳

### 9.4 1B 级别：预 tokenize

唯一需要额外工程的地方。200B tokens 每次重新 tokenize 是纯浪费——CPU 开销会成为数据加载
瓶颈，而 GPU 在空转烧钱。

收益：

- 训练时**零 tokenization 开销**，memmap 顺序读可达 GB/s
- 提前完成 **packing**（拼接 + 切定长），不在训练时现算
- **数据顺序可复现**，checkpoint 恢复后分布一致

**存储技巧**：vocab ≤ 65536 时用 **uint16** 存，200B tokens 从 800GB 降到 **400GB**，
network volume 费用直接减半。现代 tokenizer 多为 128k vocab 需 uint32；若自训 tokenizer，
控制在 64k 以内可省一半存储。

> **在 CPU Pod 上做 tokenize，别占用 GPU Pod。** GPU 按秒计费，CPU 活儿不该烧 GPU 的钱。

### 9.5 存储选择

| 方案 | 价格 | 特性 |
|---|---|---|
| 容器盘 | $0.10/GB/月 | 停止即擦除，无需清理 |
| **network volume** | **$0.07/GB/月** | 持久，**地域锁定** |
| 流式（不落盘） | $0 | 每次重新拉，无随机访问 |

800GB 的具体数字：容器盘 $80/月（仅运行期），network volume $56/月（持续计费）。

> **network volume 每 GB-月更便宜，但必须记得删。** 训练完忘删就是每月 $56 白流走 —— 见 §5 的每周 `runpodctl get volume` 检查。

**容器盘硬限制**：默认 20GB；CPU Pod 按 vCPU 自动分配（CPU3G/3C = vCPU×10GB，
CPU5C = vCPU×15GB）。**800GB 这种量级必须用 network volume。**

### 9.6 地域锁定陷阱

**network volume 创建时须指定数据中心，之后不可迁移，且只有该机房的 GPU 能用它。**

对 Spot 用户这是真实张力：spot GPU 在各机房的可用性随时变化，数据锁在没货的机房就只能干等。

缓解办法正是 §9.1 的结论 —— **因为传输免费，你永远可以选择重新下载而非死等**：

- **长期大项目** → 建 network volume，锁在 4090 供给最充足的机房
- **短期实验** → 不建 volume，每次重新拉，保持机房选择自由

### 9.7 自有数据：S3 兼容 API

不在 HF 上的自有语料，可直接从 Mac 传到 network volume，**不用租 Pod 中转**：

```bash
# 先在 console → Settings → S3 API Keys 建独立密钥（区别于普通 API key）
aws s3 cp ./corpus.tar.gz s3://$NETWORK_VOLUME_ID/ \
    --endpoint-url https://s3api-us-ca-2.runpod.io \
    --region US-CA-2 \
    --cli-read-timeout 7200
```

- **bucket 名就是 network volume ID**，对象名直接映射文件系统路径
- **region 就是机房 ID**，endpoint 为 `https://s3api-<机房>.runpod.io/`
- **不额外计费**（只有存储费）
- 当前支持 15 个机房（含 US-CA-2、US-NC-1、US-MO-1、EU-RO-1、EUR-IS-1 等）

**限制**：不支持建桶/删桶、预签名 URL、ACL、版本控制；超 500MB 自动分片；
**`aws s3 sync` 在大目录树上不可靠**（EOF / AccessDenied / 重复 token 错误），
建议小批量 `cp`；文件数超 1 万时 `ls` 变慢；时钟偏差超 1 小时会被拒。

### 9.8 对象存储与跨境搬运

**国内对象存储的公网流出定价高度一致，没有便宜的。**

| 服务 | 公网流出 | 标准存储 | 免费额度 | 可用性 |
|---|---|---|---|---|
| 阿里云 OSS | **¥0.5/GB**（闲时 ¥0.25） | ¥0.12/GB/月 | 仅新用户试用 | ✅ |
| 腾讯云 COS | **¥0.5/GB** | ¥0.118/GB/月 | 无 | ✅ |
| 火山引擎 TOS | ~¥0.5/GB | 相近 | 仅试用 | ✅ |
| 七牛云 | 阶梯 | — | 10GB + 10GB/月 | ❌ |
| 又拍云 | 阶梯 | — | 10GB + 15GB/月 | ❌ |
| **Cloudflare R2** | **$0** | $0.015/GB/月 ≈ ¥0.11 | 10GB | ⭐ 非国内 |

**七牛 / 又拍的免费额度（10–15GB）对预训练没有意义** —— 那是图床和博客的量级。

国内两个压价技巧：

- **闲时计费**：阿里云 00:00–08:00 流量半价 **¥0.25/GB**
- **CDN 回源**：**¥0.15/GB**（需配 CDN，回源仍走同一跨境链路）

800GB 的账：忙时 **¥400** / 闲时 **¥200** / CDN 回源 **¥120**。

> **但流量费不是主要成本。** 800GB 一次性 ¥400（~$55）；若传输期间 GPU 在跑，
> 3 天闲置就是 **$500+**。**相比 GPU 闲置费，流量费便宜近 10 倍。**

#### 关键：让传输发生在没有 GPU 计费的时候

**RunPod network volume 独立于 Pod 存在**，可用 S3 兼容 API 直接读写：

```
Mac ──(RunPod S3 API)──> network volume     ← 全程无需 GPU 运行
```

传输期间只有存储费，没有 GPU 费。**§9.1 的"让 Pod 自己拉"只适用于 Pod 侧能高速访问的源**
（HF / R2）；对 ModelScope 这类跨境慢源，正确的做法是预加载 network volume 而不是让 GPU 等。

#### 国内 OSS 在中转链中的位置

价值有限 —— 只是多一跳，最终仍要跨境：

```
Mac ──境内快──> 国内 OSS ──?──> RunPod
                            ↑ 这一跳是问题
```

第二跳需要有人拉：GPU Pod 拉（烧 GPU）/ CPU Pod 拉（便宜但慢）/ RunPod S3 API 无法拉外部 URL。

> **阿里云"传输加速"** 走其全球骨干网优化跨境，但额外收费，价格未核实。

（**内网流量免费**只适用于同云内部，对跨境到 RunPod 无效。）

#### 决策顺序

1. **可公开 → HuggingFace 公开 repo**：免费、无硬上限、RunPod 侧同区域 CDN 最快。
   预训练语料大多本就公开，这是最优解。
2. **必须私有 → Cloudflare R2**：零流量费 + 全球 CDN（RunPod 侧快）+ 存储 ¥0.11/GB/月。
3. **强制国内 → 阿里云 OSS 闲时**（¥0.25/GB），任务排在 00:00–08:00。

> ⚠️ **未验证**：R2 从中国大陆上传的实际速度与稳定性无可靠数据，Cloudflare 国内访问质量波动。
> **传 1GB 实测即可确认**，成本几乎为零。不通则退回方案 3。

---

## 10. 针对你的场景：推荐配置总表

| 工作负载 | 规模 | 推荐配置 | 预估成本 |
|---|---|---|---|
| 从头预训练 | 64M | 1–8×4090 Spot | **$0.5–4** |
| 从头预训练 | 125M | 8×4090 Spot | **$2–15** |
| 从头预训练 | 350M | 8×4090 Spot | **$14–124** |
| 从头预训练 | 1B | 8×4090 Spot (DDP) | **$100–1,110** |
| Midtraining | 1B | 8×4090 Spot | **$113–566** |
| Midtraining | 3–7B | 8×A100 Spot | **$340–3,962** |
| Midtraining | 14–27B | 8×H100 Spot | **$1,585–15,285** |
| 微调 / LoRA | 64M–1B | 8×4090 Spot | **$5–25** |
| 微调 / LoRA | 7B | 2×A100 / 2×4090 LoRA | **$15–60** |
| 微调 / LoRA | 27B | 2–4×4090 (QLoRA/LoRA) | **$70–120** |
| 后训练 (DPO) | 64M–7B | 2×4090 / 8×A100 | **$5–60** |
| 后训练 (GRPO) | 1–7B | 2–4×A100 + vLLM | **$30–400** |
| 后训练 (GRPO+LoRA) | 27B | 4×H100 | **~$2,500** |

**全范围年度预算量级：$5k–20k**，取决于 27B 相关工作的比重。如果 27B 只做 LoRA 微调而不做全参 midtraining，可以压到 **$2k 以内**。

---

## 11. 落地参考：minimind + RunPod 外壳

[minimind](https://github.com/jingyaogong/minimind)（64M–1B 从头训练，Apache-2.0）的 `trainer/`
目录正好覆盖本文档的全部分阶段目标，可以直接作为落地起点。

配套外壳：[AiMeshes/minimind-runpod](https://github.com/AiMeshes/minimind-runpod)
—— **不改动 minimind 一行代码**，靠目录约定实现持久化。

### 11.1 外壳能很薄，因为 minimind 自带了这些

| 能力 | 位置 | 意义 |
|---|---|---|
| **原子写 checkpoint** | `trainer_utils.py:74-76` | `.tmp` + `os.replace`，正是 §4.2 建议的做法 |
| **`--from_resume 1`** | `train_pretrain.py:104` | 自动找最新 checkpoint 续训 |
| **GPU 数自适应** | `trainer_utils.py:110-114` | ⭐ 卡数变化时自动换算 step |
| **确定性 shuffle** | `train_pretrain.py:160` | `seed + epoch` 播种，恢复后数据顺序一致 |

**已实测验证**（CPU 上跑，成本为 0）：在训练进行到 60% 时 SIGKILL，
再用 `--from_resume 1` 重启 —— checkpoint 从正确的 step/epoch 接续，
且**学习率与余弦调度曲线吻合**（续训处 lr 比初始值低 77%，若 scheduler 被重置
会立刻暴露）。验证脚本见外壳仓库的 `tests/`。

> ⚠️ **一个需要注意的例外**：`train_pretrain.py:68` 对 `../out/*.pth` 是**非原子写**
> （直接 `torch.save` 到目标路径）。抢占若恰好落在该保存窗口，这个文件可能损坏。
> **不影响续训**（`--from_resume 1` 只读 `../checkpoints/` 下的原子写入文件），
> 且下次保存会覆盖它。只有在"抢占后永久停止训练、又想拿 `out/` 权重做下一阶段"
> 这个组合下才会踩到。

**第 3 条对 Spot 尤其关键**：8 卡被抢占后重开的机器可能只有 4 卡甚至 1 卡，
minimind 能直接接上继续训，不需要任何改造——这是多数训练框架没有的能力。

**同时修正 §4.2 的一条建议**：那里说"必须保存数据迭代器位置"。
在 minimind 中**不需要**——它的 shuffle 是种子确定性的，配合 `SkipBatchSampler`
跳过已消费 batch，恢复后数据顺序天然一致。那条建议对通用框架成立，对 minimind 是多余的。

### 11.2 持久化的关键：目录约定而非改代码

`train_pretrain.py:69,118` 中 `save_dir='../checkpoints'` 是**硬编码**的，
且 minimind 约定**从 `trainer/` 目录内运行**（所以还有 `../out`、`../model`、`../dataset`）。

于是只要整体放在 network volume 上，checkpoint 就自动持久化：

```
/workspace/                      ← network volume 挂载点
└── minimind/
    ├── trainer/                 ← 从这里运行
    ├── dataset/
    ├── checkpoints/             ← '../checkpoints' 解析到此 ← 持久化
    └── out/
```

**这是 wrap 优于 patch 的具体体现**：上游改 `train_pretrain.py` 的任何内容都不影响这层约定。

### 11.3 为什么是外壳而不是 fork 改造

minimind 上游非常活跃（60k stars，几乎每天有 commit）。直接改 `trainer/*.py` 会让每次
同步上游都在处理合并冲突；外壳模式下上游改了什么都不影响你。

| 方案 | 上游同步 | 适用 |
|---|---|---|
| submodule | 干净 | 不修改代码 |
| subtree | 冲突多 | 不修改代码、要单次 clone |
| fork 改造 | 冲突多 | 修改核心逻辑 |
| **外壳 wrap** | **干净** | **只加运行层 ← 推荐** |

### 11.4 基础镜像已核实

`runpod/pytorch:1.1.0-cu1281-torch260-ubuntu2204`：Python 3.12 / torch 2.6.0 /
CUDA 12.8.1 / 10.6GB。

> **该镜像已预设 `HF_HUB_ENABLE_HF_TRANSFER=1`**（§9.3 建议的并行下载开箱即用），
> 但**同时把 `HF_HOME` 指向 `/workspace/.cache/huggingface/`** —— 即 network volume。
> **HF 缓存会常驻并计费（$0.07/GB/月）**，不需要跨 pod 复用时记得清理或改到容器盘。

---

## 12. 行动清单

### 立刻做
1. **调低花费上限**，从默认 $80/hr 降到 $5/hr 量级
2. **检查有没有遗留 volume**：`runpodctl get volume`
3. **建 network volume**，放代码和 checkpoint（不放数据集）
4. **起步用官方 `runpod/pytorch` 模板**，不要先在 Mac 上 build 镜像（见 §8.3）

### 每个实验前
5. **先算这笔账的 Spot 成本**：< $200 就用 Secure 按需，别搞编排（见 §7.1）
6. **花 $5 校准**：跑 500 步，实测 tokens/sec，再外推。本文所有数字都是估算，实测值可能差 2 倍
7. 只有确定上 Spot 时，才确认 checkpoint 目录在 network volume 上，**不在容器盘**

### 训练中
8. 每 30–60 分钟原子写 checkpoint
9. 监控 MFU，低于 30% 就该查瓶颈（数据加载？通信？）
10. 达到目标后**立即 terminate**，不要 stop

### 每周末
11. 跑 `runpodctl get pod` + `runpodctl get volume`，清理遗漏资源

---

## 13. 参考数字速查

```
有效 FLOPS（bf16，实际可达）
  4090:  120 TFLOPS     5090:  145 TFLOPS
  3090:   45 TFLOPS     A100:  125 TFLOPS
  H100:  396 TFLOPS (SXM) / 302 (PCIe)

GPU-小时 = 6 × N × D / (有效FLOPS × 3600)

例：7B × 20B tokens on H100 SXM
  = 6 × 7e9 × 2e10 / (396e12 × 3600)
  = 8.4e20 / 1.4256e18
  = 589 GPU-小时 → $1,585 @ $2.69/hr

显存（全参训练）= N × 12 字节（bf16 权重+梯度 + fp32 Adam）
  用 8-bit Adam → N × 6 字节

显存（LoRA 微调）≈ N × 2 字节 (bf16 base) + 适配器 + 激活
  4-bit QLoRA   ≈ N × 0.5 字节 + 适配器 + 激活
```

---

## 附：来源

- [RunPod Pods Pricing（官方）](https://docs.runpod.io/pods/pricing)
- [RunPod Storage Options（官方）](https://docs.runpod.io/pods/storage/types)
- [RunPod Network Volumes（官方）](https://docs.runpod.io/storage/network-volumes)
- [RunPod Discounts 2026: Spot Pricing & Committed Rates](https://costbench.com/software/ai-gpu-cloud/runpod/discounts/)
- [RunPod Spot vs On-Demand — When the 50% Discount Is Worth the Interruption](https://ice-ice-bear.github.io/posts/2026-04-22-runpod-spot-vs-ondemand/)
- [RunPod GPU Pricing: 2026 Comprehensive Pricing Guide](https://deploybase.ai/articles/runpod-gpu-pricing)

**数据获取**
- [RunPod S3 兼容 API（官方文档）](https://docs.runpod.io/storage/s3-api)
- [Cloudflare R2 定价（官方文档）](https://developers.cloudflare.com/r2/pricing/)
- [阿里云 OSS 与腾讯云 COS 价格对比](https://heng666.cn/posts/2026-07-29-aliyun-oss-oss-vs-cos-price-compare.html)
- [七牛云免费额度说明（官方）](https://developer.qiniu.com/af/kb/1574/free-credit-information)
- [火山引擎 TOS 资源包概述（官方）](https://docs.volcengine.com/docs/6349/178341)
- [HF 流式数据集（官方博客，中文）](https://huggingface.co/blog/zh/streaming-datasets)
- [hf_transfer 集成说明](https://deepwiki.com/huggingface/hf_transfer/4.1-integration-with-huggingface_hub)
- [HuggingFace 存储与计费](https://deepwiki.com/huggingface/hub-docs/7.2-billing-and-storage-management)

**编排 / CI**
- [runpodctl 官方仓库](https://github.com/runpod/runpodctl)
- [runpodctl 概览（官方文档）](https://docs.runpod.io/runpodctl/overview)
- [RunPod 文件传输文档](https://docs.runpod.io/pods/storage/transfer-files)
- [RunPod CI/CD 集成指南（官方）](https://www.runpod.io/articles/guides/integrating-runpod-with-ci-cd-pipelines)
- [GitHub Actions timeout 限制讨论](https://github.com/orgs/community/discussions/180907)
- [GitHub Actions Pricing 2026](https://toolradar.com/tools/github-actions/pricing)
- [Mastering Disk Space on GitHub Actions Runners](https://www.geraldonit.com/mastering-disk-space-on-github-actions-runners-a-deep-dive-into-cleanup-strategies-for-x64-and-arm64-runners/)

> 本文所有成本为估算，基于 §13 的假设。实际差异可能达 2 倍。**跑 500 步实测再外推。**
