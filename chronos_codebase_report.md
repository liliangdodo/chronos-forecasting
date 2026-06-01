# Chronos 仓库代码报告

结论先说：**这个仓库里，完整暴露出来的“预训练”流水线主要对应原始 Chronos（token-based 版本）**；**Chronos-Bolt 主要给了模型实现与推理路径**，**Chronos-2 给了模型实现、推理和 `fit()` 微调路径，但没有像 `scripts/training/train.py` 那样完整的预训练脚本**。下面这份报告只基于仓库代码与文档本身。

## 1. 详细模型架构

### 1.1 原始 Chronos（`src/chronos/chronos.py`）

这是一个**把时间序列离散成 token，再交给语言模型做生成**的方案。

#### 核心组成

1. **`MeanScaleUniformBins` tokenizer**
   - 先对每条序列做 mean-scale 风格的缩放：用观测值绝对值均值当 scale。
   - 再把缩放后的实数落到均匀分箱上，变成 token id。
   - 缺失值/左 padding 用 `pad_token_id`。
   - `seq2seq` 模型可在末尾追加 EOS。
2. **`ChronosModel`**
   - 底层直接包 Hugging Face 的 `AutoModelForSeq2SeqLM` 或 `AutoModelForCausalLM`。
   - 预测时调用 `generate()`，按 `temperature/top_k/top_p/num_samples` 采样未来 token。
3. **`ChronosPipeline`**
   - 输入支持 1D tensor、list of 1D tensor、2D batch tensor。
   - 先左侧用 `torch.nan` 对齐，再 tokenization。
   - 输出是**样本路径** `(batch, num_samples, horizon)`。
   - 如果用户预测长度超过模型原生 horizon，pipeline 会**自回归滚动**：把上一段预测的中位数拼回 context，继续生成。

#### 训练视角

- 本质上是**标准 LM 训练**：
  - `seq2seq`：`past_target -> future_target tokens`
  - `causal`：把 context 和 label 拼成一段序列，依赖模型内部 shift 做 next-token loss
- 训练目标不是显式数值 loss，而是**token 级交叉熵**。

### 1.2 Chronos-Bolt（`src/chronos/chronos_bolt.py`）

这是一个**patch-based、直接输出分位数**的架构，不再走 token 生成。

#### 核心组成

1. **`Patch`**
   - 把一维时间序列切成 patch。
   - 如果长度不是 patch size 的整数倍，会在左侧补 `NaN`。
2. **`InstanceNorm`**
   - 对每条序列做实例标准化，保存 `(loc, scale)`，预测后再 inverse。
3. **输入嵌入**
   - 每个 patch 由 **`[patch_values, patch_mask]`** 拼接。
   - 送入 `ResidualBlock` 做 patch embedding。
4. **主干网络**
   - 主体是 **T5 风格 encoder-decoder**：
     - encoder 读 patch embeddings
     - decoder 只吃一个起始 token，输出一个隐藏状态
5. **输出头**
   - `output_patch_embedding` 直接把 decoder 输出映射成 `num_quantiles * prediction_length`
   - 即一次前向直接得到整段 horizon 的所有 quantile
6. **训练 loss**
   - 显式使用 **quantile / pinball loss**
   - 输出是 `(batch, num_quantiles, prediction_length)`

#### 长预测策略

- 如果预测长度超过模型原生 `prediction_length`，不是简单递归均值，而是：
  - 保留第一段所有 quantile
  - 把 quantile 作为多条“路径”回灌到 context
  - 再次预测后做经验分位数归并
- 所以 Bolt 的长 horizon 是**量化路径展开 + 分位数聚合**。

### 1.3 Chronos-2（`src/chronos/chronos2/`）

这是仓库里**结构最复杂**的一代，支持：

- 单变量
- 多变量
- past covariates
- known future covariates
- cross-learning / group learning

#### 模型输入组织

`src/chronos/chronos2/dataset.py`

- 一个 task 可由：
  - `target`
  - `past_covariates`
  - `future_covariates`
  组成
- 内部会把它们整理成：
  - `context`: `(n_targets + n_covariates, history_length)`
  - `future_covariates`: `(n_targets + n_covariates, prediction_length)`
- 其中 **known future covariates 必须是 `past_covariates` 的子集**。
- batch 里的不同行通过 `group_ids` 标记哪些变量属于同一个 forecasting task。

#### 编码结构

`src/chronos/chronos2/model.py` + `layers.py`

1. **Patch + InstanceNorm**
   - 与 Bolt 一样，先 patch 化和实例归一化。
2. **时间编码**
   - context patch 会附加归一化时间索引 `[-C, ..., -1] / scale`
   - future patch 会附加 `[0, 1, ..., H-1] / scale`
3. **输入 embedding**
   - 每个 patch 输入是 **`[time_encoding, patch_values, patch_mask]`**
   - 经 `ResidualBlock` 投影到 `d_model`
4. **Chronos2Encoder**
   - 每层包含三段：
     1. **TimeSelfAttention**：沿时间维做注意力，带 **RoPE**
     2. **GroupSelfAttention**：沿 batch/group 维做注意力，只允许同组变量互相看见
     3. **FeedForward**
5. **未来 patch 作为“待预测槽位”**
   - 模型不是 decoder-only，也不是 T5 encoder-decoder
   - 它把 **context patch + future patch embeddings** 拼起来，一起送进 encoder
   - 最后只取末尾 `num_output_patches` 的隐藏状态做输出
6. **输出头**
   - `output_patch_embedding` 输出 `num_quantiles * output_patch_size`
   - reshape 后得到 `(batch, num_quantiles, horizon)`
7. **训练 loss**
   - 也是 **quantile loss**
   - 但会额外 mask 掉：
      - 缺失 target
      - 已知 future covariates 对应的行
   - 所以模型训练时只对真正需要预测的目标变量计 loss

#### 更细的网络拓扑

为了便于直接调 `scripts/training/configs/chronos-2-small.yaml`，下面把 Chronos-2 写成**从用户输入到用户输出**的端到端框图。

先约定几个符号：

- `B`：送入模型的总序列条数（注意，不是 task 数；target 和 covariate 都算一条）
- `G`：一个 batch 里 task 的数量
- `L`：输入历史长度，实际进入模型前会被截到 `context_length`
- `C`：`context_length`
- `P_in`：`input_patch_size`
- `S_in`：`input_patch_stride`
- `P_out`：`output_patch_size`
- `N_ctx`：历史 patch 数，近似 `floor((有效历史长度 - P_in) / S_in) + 1`
- `N_out`：输出 patch 数，训练时约等于 `ceil(prediction_length / P_out)`
- `Q`：quantile 个数，即 `len(quantiles)`
- `D`：`d_model`
- `H`：`num_heads`
- `D_kv`：`d_kv`
- `D_ff`：`d_ff`

> 这份 `chronos-2-small.yaml` 中，主要影响张量形状的是：`context_length`、`prediction_length`、`input_patch_size`、`input_patch_stride`、`output_patch_size`、`max_output_patches`、`quantiles`、`d_model`、`d_kv`、`num_heads`、`d_ff`、`num_layers`。

**总框图（从输入到输出）**

```text
用户输入
  context            : (B, L)
  future_covariates  : (B, prediction_length) 或 None
  group_ids          : (B,)
  future_target      : (B, prediction_length)    # 仅训练/验证时
        │
        ▼
+-------------------------------+
| 1. 输入检查 / 长度约束        |
|   - 若 L > C，截断到最后 C 步 |
|   - 要求 N_out * P_out >= H   |
+-------------------------------+
输入:  context           (B, L)
输出:  context           (B, min(L, C))
       future_*          (B, prediction_length)
        │
        ▼
+-------------------------------------------+
| 2. InstanceNorm / arcsinh                 |
|   - 每条序列独立缩放                      |
|   - 保存 loc_scale 供输出反归一化         |
+-------------------------------------------+
输入:  context           (B, <=C)
输出:  normalized_ctx    (B, <=C)
       loc_scale         tuple of tensors
        │
        ├──────────────────────────────────────────────┐
        ▼                                              ▼
+----------------------------------+     +----------------------------------+
| 3A. 历史 patchify                |     | 3B. 未来 patchify                |
| Patch(P_in, S_in)                |     | 未来长度先补齐到 N_out * P_out   |
|                                  |     | 然后 reshape 成 future patches   |
+----------------------------------+     +----------------------------------+
输入:  normalized_ctx    (B, <=C)        输入: future_covariates   (B, H)
输出:  patched_ctx       (B, N_ctx, P_in)输出: patched_future      (B, N_out, P_out)
       patched_mask      (B, N_ctx, P_in)       future_mask        (B, N_out, P_out)
        │                                              │
        └──────────────────────────────┬───────────────┘
                                       ▼
+---------------------------------------------------------------+
| 4. patch 特征拼接                                             |
| concat([time_encoding, patch_values, patch_mask], dim=-1)     |
+---------------------------------------------------------------+
输入:  history patches      (B, N_ctx, P_in)
输出:  history patch feats  (B, N_ctx, 3 * P_in)

输入:  future patches       (B, N_out, P_out)
输出:  future patch feats   (B, N_out, 3 * P_out)
        │
        ▼
+---------------------------------------------------------------+
| 5. input_patch_embedding = ResidualBlock                      |
|   主支路: Linear(3*P -> D_ff) -> Act -> Dropout -> Linear(D_ff -> D)
|   残差支路: Linear(3*P -> D)
+---------------------------------------------------------------+
输入:  history patch feats  (B, N_ctx, 3 * P_in)
输出:  history embeds       (B, N_ctx, D)

输入:  future patch feats   (B, N_out, 3 * P_out)
输出:  future embeds        (B, N_out, D)
        │
        ▼
+---------------------------------------------------------------+
| 6. 序列拼接                                                   |
| [history_embeds] + [可选 REG token] + [future_embeds]         |
+---------------------------------------------------------------+
输入:  history embeds       (B, N_ctx, D)
       REG token            (B, 1, D) 或省略
       future embeds        (B, N_out, D)
输出:  encoder input        (B, N_ctx + reg + N_out, D)
       attention_mask       (B, N_ctx + reg + N_out)
        │
        ▼
+---------------------------------------------------------------+
| 7. Chronos2Encoder (重复 num_layers 次)                       |
| 每层 = TimeSelfAttention -> GroupSelfAttention -> FeedForward |
+---------------------------------------------------------------+
输入:  hidden_states        (B, T, D)  , T = N_ctx + reg + N_out
输出:  hidden_states        (B, T, D)
        │
        ▼
+---------------------------------------------------------------+
| 8. 取最后 N_out 个 token                                      |
| 这些位置对应“未来槽位”                                        |
+---------------------------------------------------------------+
输入:  encoder output       (B, T, D)
输出:  forecast_embeds      (B, N_out, D)
        │
        ▼
+---------------------------------------------------------------+
| 9. output_patch_embedding = ResidualBlock                     |
|   D -> D_ff -> (Q * P_out)                                    |
+---------------------------------------------------------------+
输入:  forecast_embeds      (B, N_out, D)
输出:  patch outputs        (B, N_out, Q * P_out)
        │
        ▼
+---------------------------------------------------------------+
| 10. reshape / 拼接 horizon                                    |
+---------------------------------------------------------------+
输入:  patch outputs        (B, N_out, Q * P_out)
输出:  quantile_preds       (B, Q, N_out * P_out)
        │
        ├─────────────── 训练时 ────────────────┐
        ▼                                        ▼
+----------------------------------+   +----------------------------------+
| 11A. quantile loss               |   | 11B. 反归一化 inverse()          |
| mask 掉 missing target 和        |   | 恢复到原始数值尺度               |
| 已知 future covariates 行        |   +----------------------------------+
+----------------------------------+   输出: user forecast  (B, Q, horizon)
```

**单个 Encoder block 框图**

```text
输入 hidden_states: (B, T, D)
        │
        ▼
+---------------------------------------------------+
| TimeSelfAttention                                 |
| pre-norm                                          |
| Q/K/V/O: D -> H * D_kv -> D                       |
| attention 轴: 时间维 T                            |
| RoPE 只作用在 q/k 上                              |
+---------------------------------------------------+
输入:  (B, T, D)
输出:  (B, T, D)
        │ residual
        ▼
+---------------------------------------------------+
| GroupSelfAttention                                |
| pre-norm                                          |
| 先视角变换成“同一时间步上，不同变量彼此注意”       |
| attention 轴: group/batch 维                      |
| mask: 只允许相同 group_id 互相可见                |
+---------------------------------------------------+
输入:  (B, T, D)
输出:  (B, T, D)
        │ residual
        ▼
+---------------------------------------------------+
| FeedForward                                       |
| pre-norm                                          |
| Linear(D -> D_ff) -> Act -> Dropout -> Linear(D_ff -> D) |
+---------------------------------------------------+
输入:  (B, T, D)
输出:  (B, T, D)
```

**把 YAML 参数映射到图里**

| YAML 参数 | 作用位置 | 直接影响的 shape/容量 |
|---|---|---|
| `context_length` | 输入截断、time encoding | 决定输入最多保留多少历史步，影响 `N_ctx` |
| `prediction_length` | 训练目标长度 | 决定训练时 `N_out = ceil(prediction_length / P_out)` |
| `input_patch_size` | 历史 patch 切分 | 决定 `patched_ctx` 最后一维 `P_in` |
| `input_patch_stride` | 历史 patch 步长 | 决定 `N_ctx` 的多少 |
| `output_patch_size` | 输出 patch 长度 | 决定每个未来 token 覆盖多少步 |
| `max_output_patches` | 模型原生 horizon 容量 | 静态容量约为 `max_output_patches * output_patch_size` |
| `quantiles` | 输出头 | 决定 `Q`，即输出 `(B, Q, horizon)` 的第二维 |
| `d_model` | 所有主干隐藏层 | 决定 encoder/token hidden size `D` |
| `d_kv` | 每个 head 的 q/k/v 宽度 | 决定 attention 内部单头维度 |
| `num_heads` | 多头注意力 | 决定 `H`，总 attention 内维约为 `H * D_kv` |
| `d_ff` | MLP / ResidualBlock 中间层 | 决定每层 FFN 宽度 |
| `num_layers` | Encoder depth | 决定 block 重复次数 |
| `use_reg_token` | 序列拼接 | 若开启，`T = N_ctx + 1 + N_out`；否则 `T = N_ctx + N_out` |

**结合已对齐 `amazon/chronos-2` 的 `chronos-2-small.yaml`，可以直接代入**

- `C = 8192`
- `prediction_length = 64`
- `P_in = 16`
- `S_in = 16`
- `P_out = 16`
- `Q = 21`
- `D = 768`
- `H = 12`
- `D_kv = 64`
- `D_ff = 3072`
- `num_layers = 12`
- `use_reg_token = true`

因此：

- 训练时 `N_out = ceil(64 / 16) = 4`
- 若历史长度正好取满 `C=8192` 且 stride=patch_size=16，则 `N_ctx = 8192 / 16 = 512`
- encoder 序列长度约为 `T = 512 + 1 + 4 = 517`
- 最终输出 shape 是：
  - patch 输出前：`(B, 4, 21 * 16)`
  - reshape 后：`(B, 21, 64)`

这样你调参数时可以直接看：

- 想增加历史容量：优先调 `context_length`
- 想增加 horizon：调 `prediction_length`，同时留意 `ceil(prediction_length / output_patch_size)`
- 想减少 encoder 序列长度/显存：增大 `input_patch_size` 或 `input_patch_stride`
- 想增加模型容量：调 `d_model` / `d_ff` / `num_layers` / `num_heads`

#### Chronos-2 的关键设计点

- **group attention**：允许同一 task 内多变量/协变量共享信息
- **future covariates 显式入模**
- **cross-learning**：推理时可把多个 task 设为同组或同批共享模式
- **长预测**：通过 `unrolled_quantiles` 做自回归 quantile 展开，再用 `weighted_quantile` 聚合

## 2. 预训练策略与流程（含脚本）

### 2.1 仓库里真正的预训练入口

**`scripts/training/train.py`** 是唯一完整的预训练/继续训练脚本，面向**原始 Chronos**。

### 2.2 训练流程

结合 `scripts/README.md` 和 `scripts/training/train.py`，流程是：

1. **准备数据**
   - 要求是 **GluonTS-compatible FileDataset**
   - 推荐 Arrow 格式
   - 每条样本至少有：
     - `start`
     - `target`
2. **可选：生成合成数据**
   - 用 **`scripts/kernel-synth.py`** 生成 `kernelsynth-data.arrow`
3. **配置训练**
   - 用 `scripts/training/configs/*.yaml`
   - 默认配置都混合两份数据：
     - `tsmixup-data.arrow`
     - `kernelsynth-data.arrow`
   - 默认混合权重是 **0.9 / 0.1**
4. **启动训练**
   - 单卡：`python training/train.py --config ...`
   - 多卡：`torchrun --nproc-per-node=... training/train.py --config ...`
5. **训练脚本内部处理**
   - 用 `FileDataset(path, freq="h")` 读取数据
   - 先过滤：
     - 长度至少 `min_past + prediction_length`
     - 缺失率不超过 `max_missing_prop`
   - `ChronosDataset` 再做：
     - 训练/验证/测试 window 切片
     - 训练时随机 missing-value dropout（仅 seq2seq 生效）
     - tokenizer 编码
     - 组装成 HF Trainer 需要的 `input_ids / attention_mask / labels`
6. **Trainer 训练**
   - 用 Hugging Face `Trainer`
   - 按 `max_steps/save_steps/log_steps`
   - TensorBoard 上报
   - 最终保存到 `output/run-*/checkpoint-final`
7. **保存训练信息**
   - 会额外写 `training_info.json`
   - 包含 training config 和运行环境信息

### 2.3 预训练策略细节

#### A. 数据混合策略

默认 config（`chronos-t5-*.yaml`, `chronos-gpt2.yaml`）统一是：

- `training_data_paths = [tsmixup-data.arrow, kernelsynth-data.arrow]`
- `probability = [0.9, 0.1]`

说明默认策略是：**以真实/外部 prepared 数据为主，KernelSynth 合成为辅**。

#### B. 窗口采样策略

`ChronosDataset._create_instance_splitter()`

- 训练时使用 `ExpectedNumInstanceSampler(num_instances=1.0, min_past, min_future)`
- 即从长序列里**随机采样训练窗口**
- 输入长度受 `context_length` 限制
- 预测段长度固定为 `prediction_length`

#### C. 缺失值增强

`ChronosDataset.preprocess_entry()`

- `seq2seq` 模型训练时会对 target 随机置 NaN
- `drop_prob` 不是固定比例，而是每条样本先随机采一个 `drop_p ~ U(0, drop_prob)`
- 这是显式的 missing-value augmentation

#### D. causal 与 seq2seq 两种训练方式

- **T5 配置 (`chronos-t5-*.yaml`)**
  - `model_type: seq2seq`
  - `random_init: true`
  - `tie_embeddings: true`
  - 即：用 T5 架构，但不是直接继承已有权重做微调，而是**随机初始化后从头训练 Chronos token 任务**
- **GPT2 配置 (`chronos-gpt2.yaml`)**
  - `model_type: causal`
  - `random_init: false`
  - `model_id: openai-community/gpt2`
  - 即：从 GPT-2 权重初始化再适配 Chronos token 词表
  - 且默认 `max_missing_prop` 更严格（0.1）
  - 训练前会做 `LastValueImputation`

#### E. tokenizer / 离散化策略

训练配置固定了：

- `tokenizer_class: MeanScaleUniformBins`
- `n_tokens: 4096`
- `low_limit: -15.0`
- `high_limit: 15.0`
- `use_eos_token: true`

也就是：**先缩放，再把数值压到 4096 token 的均匀离散空间**。

### 2.4 Chronos-2 的训练路径：是微调，不是完整预训练

`src/chronos/chronos2/pipeline.py::fit`

仓库给 Chronos-2 的是**微调 API**：

- `finetune_mode="full"` 或 `"lora"`
- 构建 `Chronos2Dataset`
- 用自定义 `Chronos2Trainer`
- 可选 `validation_inputs`
- 训练后自动保存 `finetuned-ckpt`

**LoRA 目标模块**

- `self_attention.q`
- `self_attention.k`
- `self_attention.v`
- `self_attention.o`
- `output_patch_embedding.output_layer`

所以代码层面看：

- **Chronos-2：有系统化微调路径**
- **没有对应 `scripts/training/train.py` 那种完整预训练入口**

### 2.5 Chronos-Bolt 的训练路径

从仓库代码看：

- 有完整模型定义和推理逻辑
- 有评测脚本
- **没有与原始 Chronos 对应的公开预训练脚本**

因此如果只按仓库代码判断，Bolt 在这里主要是**推理/评测实现**。

## 3. 预训练使用的数据集

### 3.1 代码里明确出现的“预训练数据”

| 数据 | 来源/状态 | 仓库中的角色 | 备注 |
|---|---|---|---|
| `tsmixup-data.arrow` | 外部准备好的 Arrow 文件 | 默认主训练集 | 在所有训练 config 中占 90%；仓库**没有**提供生成脚本 |
| `kernelsynth-data.arrow` | `scripts/kernel-synth.py` 生成 | 默认辅助训练集 | 在默认 config 中占 10% |
| 用户自定义 Arrow 数据 | 用户自己转换 | 通用训练输入 | `scripts/README.md` 提供 `convert_to_arrow` 示例 |

### 3.2 KernelSynth 数据集细节

`scripts/kernel-synth.py`

- 默认生成 **100 万条**时间序列
- 每条长度 **1024**
- 从 **Gaussian Process prior** 采样
- kernel bank 包含：
  - 周期核 `ExpSineSquared`
  - 线性核 `DotProduct`
  - RBF
  - RationalQuadratic
  - WhiteKernel
  - ConstantKernel
- 每条序列随机选 1 到 `max_kernels`（默认 5）个 base kernel
- 再随机用 `+` 或 `*` 组合成复合核
- 最后采样得到合成序列
- `start` 时间戳是任意固定值，不承载真实语义

这说明 KernelSynth 在这里扮演的是：**提供大规模、多模式、可控的合成时序先验**。

### 3.3 `tsmixup-data.arrow`

代码里只看到它被默认训练 config 引用，**没有生成脚本、没有数据说明、也没有字段定义之外的信息**。因此从仓库本身只能确认：

- 它是一个 **GluonTS-compatible Arrow 数据集**
- 它被当作主要预训练语料
- 默认与 KernelSynth 按 9:1 混合

### 3.4 `autogluon/chronos_datasets` 是什么

仓库很多地方还出现：

- `autogluon/chronos_datasets`
- `autogluon/chronos_datasets_extra`

但这些主要出现在：

- `README.md`
- `scripts/evaluation/configs/*.yaml`
- `scripts/evaluation/evaluate.py`
- 测试代码

所以按代码角色判断，它们是**评测/基准数据集入口**，**不是这里默认预训练脚本直接消费的训练集**。

## 一句话总结

- **原始 Chronos**：离散化成 token，用 T5/GPT2 类 LM 做生成式预训练；完整训练脚本在 `scripts/training/train.py`
- **Chronos-Bolt**：patch-based + T5 encoder-decoder + 直接分位数回归；仓库里没给完整预训练脚本
- **Chronos-2**：patch-based + 时间注意力 + 组注意力 + 协变量统一建模；仓库里重点给了推理和 `fit()` 微调，而不是完整预训练流水线
- **默认预训练数据**：`tsmixup-data.arrow`（主） + `kernelsynth-data.arrow`（辅，代码可生成）
