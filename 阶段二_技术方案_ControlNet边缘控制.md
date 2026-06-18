# 阶段二技术方案 · 给 Consistency Model 加「二值边缘图控制生成」

> 配套交接文档：`阶段二_交接文档_边缘图控制生成.md`
> 本文是**实现方案 + 因果链讲解 + 运行手册 + 面试预演**。重点把"为什么这样设计"讲透。

---

## 0. TL;DR（一句话到一段话）

把官方 `consistency_models`（ADM U-Net）从无条件/类别条件的 `f_θ(x_t, t)`，扩展成**边缘图条件**的 `f_θ(x_t, t, c)`。

采用的方案（已和你对齐选定）：
- **路线**：ControlNet 旁路分支——**冻结**预训练 CM 主干，复制其编码器为**可训练副本**，用**零卷积**把边缘图特征注入主干解码器的 skip 连接。
- **训练范式**：**条件 CT（consistency training）微调**——从预训练 CM 检查点初始化主干，无教师，在线模型与 EMA target 模型吃同一张边缘图，保持"同轨道 + 同条件 → 自洽"。
- **数据**：在线 Canny 生成 (图像, 边缘图) 配对；先用极小合成数据集打通全流程。
- **现场 Demo**：`scripts/demo.py --gradio`，老师当场拖任意网页图片 → 调阈值 → 即时看"原图 | 边缘 | 生成"三联图。

**已完成并通过 CPU 校验**：模型/损失/训练步全链路（17 项 smoke test 全 PASS）。**待你在 `cm` 环境 + GPU 上执行**：建环境、下权重、跑训练、现场 demo。

---

## 1. 最关键的洞察：条件如何"免费"流经全链路

读 `cm/karras_diffusion.py` 的 `denoise()`（约 347 行）：

```python
model_output = model(c_in * x_t, rescaled_t, **model_kwargs)
denoised = c_out * model_output + c_skip * x_t
```

**条件是靠 `model_kwargs` 字典透传的**。原本类别条件就是 `model_kwargs={"y": labels}`。而这个字典在 `training_losses`、`consistency_losses`（CD/CT）、`karras_sample` 里**全部原样透传**；上游训练循环 `cm/train_util.py` 把数据集产出的 `cond` 字典转成 `micro_cond` 再喂进去；数据集 `__getitem__` 的 `out_dict` 就是 `cond` 的源头。

**因果链**：数据集 `out_dict` 里加一个 `"edge"` 键 → 它自动流经训练损失 / CD / CT / 采样全链路 → 最终到达 `model(..., edge=...)`。

**这带来一个设计决定**：真正要写的只有两处——① U-Net 的前向接住 `edge` 并送进 ControlNet；② 数据集把图像跑 Canny 生成 `edge`。其余全部"免费"。这也天然满足交接文档 §4.4 的"条件 c 跨所有 t 固定"：边缘图不经过 `c_in` 缩放、不随 t 变。

---

## 2. 设计决策与"为什么"（因果链，面试重点）

### 2.1 为什么必须训练、不能只靠推理期 trick？

论文的零样本编辑（上色/超分/SDEdit，Appendix D）控制信号是**与输出同构的像素图**，在多步采样里"迭代替换"即可，不改权重。但**二值边缘图是异构的结构约束**（不是待生成图的残缺版，而是另一种模态）。没有任何"把边缘图替换进 x_t 的某个子空间"的自然算子。所以必须像 ControlNet 那样**训练一个分支，把控制信号编码进网络**。

### 2.2 为什么是 ControlNet 旁路分支（而非通道拼接 / 交叉注意力）？

| 路线 | 控制力 | 训练稳定性 | 算力 | 关键问题 |
|---|---|---|---|---|
| **ControlNet 旁路（选用）** | 强（空间逐像素对齐） | 零卷积→起点=原模型，最稳 | 只训分支，省 | 需复制编码器 |
| 通道拼接 | 弱~中 | 破坏预训练首层权重 | 要重训较多 | 第一层卷积要扩通道 |
| 交叉注意力 | 中（偏语义） | 中 | 中 | 对"精确轮廓"不如前两者 |

ControlNet 与这个 U-Net 的「编码器存 skip → 解码器 pop skip」结构**天作之合**：把控制特征加到 `hs.pop()` 上即可。

### 2.3 ControlNet 分支的三个部件（`cm/controlnet.py`）

1. **input_hint_block**：把边缘图（1 通道）编码成与第一层特征同形状（`model_channels` 通道、同分辨率）的特征图。**末层零初始化** → 起始注入为 0。
2. **可训练编码器副本**：deepcopy 主干的 `input_blocks + middle_block + time_embed`（加载预训练后再 `sync_controlnet_from_unet` 同步权重）。
3. **零卷积（zero-conv）**：每个 skip 一个 1×1 卷积、权重/偏置全 0，把副本特征变成"残差"加回主干。

### 2.4 为什么"零卷积初始化"是稳定性的关键？

零卷积初始为 0 → 注入残差为 0 → **训练第一步，条件模型的输出与冻结主干完全相同**（smoke test 已验证 `controlled(edge) == backbone`）。训练只能从这个"安全起点"缓慢偏离，不会一上来就破坏预训练能力。

**ControlNet 的梯度动态（容易被问）**：零卷积权重为 0，但它**仍能收到梯度**——卷积权重的梯度 = 上游梯度 ⊗ 输入特征，输入非零则梯度非零（smoke test 验证 zero-convs 收到非零梯度）。所以第一步只有 zero-conv 更新；它一旦变非零，梯度才继续往上游的编码器副本流。这是 ControlNet 标准且健康的训练动态。

### 2.5 为什么加了边缘条件，**边界条件 f(x, ε, c)=x 天然守住**？（最漂亮的一点）

一致性模型用参数化 `f = c_skip(σ)·x + c_out(σ)·F_θ`，其中 `c_skip, c_out` 是 σ 的**固定函数、不学习**（`get_scalings_for_boundary_condition`）。在 σ=σ_min（即 ε）处：`c_skip=1, c_out=0`。于是

```
f(x, ε, c) = c_out(ε)·F_θ(x, ε, c) + c_skip(ε)·x = 0·F_θ + 1·x = x
```

**`c_out(ε)=0` 把整个网络输出（主干 + ControlNet）一并清零**。所以无论边缘图 c 是什么、训练到哪一步，边界条件恒成立——**边缘条件在架构上不可能破坏它**。smoke test 直接验证了 `denoised(σ_min) == x`。

### 2.6 为什么用条件 CT，而不是 CD？

`consistency_losses` 把同一个 `model_kwargs` 传给在线模型、target 模型、**以及 CD 的教师**。
- **CD 的死结**：教师是预训练的扩散模型，**不认识边缘图**。把 `edge=...` 传给它，它的 `forward` 不接受这个参数会直接报错；而且官方放出的教师是无边缘条件的，没有现成的"边缘条件教师"。
- **CT 没有教师**（`teacher_model=None`，用无偏 score 估计 `−(x_t−x)/t²`）。在线模型与 EMA target 模型吃**同一张边缘图**，"同轨道两点 + 同一 c → 输出一致"天然成立。

所以条件 CT 是与本任务最相容、最省事的范式：从预训练 CM 检查点初始化主干（冻结），只训 ControlNet 分支，用 CT 损失自蒸馏保持一致性。

### 2.7 一个工程坑：flash-attn 与注意力回退

代码默认走 flash 注意力（`flash-attn==0.2.8`，现代环境几乎装不上，且用的是旧 API）。我让 `AttentionBlock` 在 flash 不可用时**自动回退到 `QKVAttention`**（纯 torch、无 einops、不强制 half，且与 flash 路径**数值等价**——同样的 Q/K/V head 分组）。所以官方用 flash 训练的权重，在没有 flash-attn 的 `cm` 环境/CPU 上都能正确加载并运行。

---

## 3. 逐文件改动清单

| 文件 | 改动 | 作用 |
|---|---|---|
| `cm/unet.py` | `UNetModel.forward` 增 `control=None, only_mid_control=False`；`__init__` 保存 `_input_block_chans`、`_middle_block_channels`；`AttentionBlock` flash→`QKVAttention` 自动回退 | 让主干能接住 ControlNet 残差；暴露通道数；去 flash 依赖 |
| `cm/controlnet.py`（新） | `ControlNet` 分支、`ControlledModel` 包装、`sync_controlnet_from_unet` | ControlNet 核心 |
| `cm/karras_diffusion.py` | `from piq import LPIPS` 改为惰性（仅 lpips 分支内 import） | l2 路径/CPU 测试无需 piq |
| `cm/script_util.py` | 增 `use_controlnet/hint_channels/loss_norm`；`use_controlnet` 时构造 `ControlledModel`；把 `loss_norm` 真正传入 `KarrasDenoiser`（修复原本被忽略的 bug） | 配置接入 |
| `cm/image_datasets.py` | 在线 Canny → `out_dict["edge"]`（与图像同 crop/flip 对齐）；`canny_random` 阈值抖动 | 配对数据 |
| `cm/edge_util.py`（新） | 鲁棒图像加载（含 URL）、Canny、三联图可视化 | 采样/Demo 共用 |
| `scripts/cm_train.py` | `--controlnet_init_checkpoint` 加载预训练主干 + `sync` + 冻结；`cond_edge=use_controlnet` 接入数据 | 训练入口 |
| `scripts/image_sample.py` | `--edge_dir` 边缘条件批量采样 + 三联图 + npz | 评测用采样 |
| `scripts/demo.py`（新） | CLI + Gradio 现场 Demo | 面试演示 |
| `scripts/prep_demo_data.py`（新） | 合成/真实极小数据集 | 打通流程 |
| `scripts/eval_edge_fidelity.py`（新） | 边缘服从度 F1/IoU | 评测 |
| `tests/smoke_controlnet.py`（新） | 17 项 CPU 正确性校验 | 验证 |
| `environment.yml` / `scripts/setup_cm_env.sh` / `scripts/download_checkpoints.sh` / `scripts/run_demo_pipeline.sh`（新） | 环境/权重/一键流程 | 工程化 |

> 兼容性：`control=None` 时 `UNetModel.forward` 行为与原版**完全一致**，原有无条件/类别采样不受影响。

---

## 4. 数据流图

```
ImageDataset.__getitem__
  image(uint8) --crop/flip--> arr
        |                       |
        | /127.5-1              | cv2.Canny
        v                       v
   x0 (3,H,W)            edge (1,H,W in [0,1])
        \                       /
         out_dict = {"y": cls, "edge": edge}
                    |
   train_util.forward_backward: micro_cond = {"y","edge"}
                    |
   consistency_losses(model, x0, num_scales, target_model, model_kwargs=micro_cond)
                    |  (CT: 同 edge 喂在线 & target)
   KarrasDenoiser.denoise: model(c_in*x_t, rescaled_t, y=, edge=)
                    |
   ControlledModel.forward(x, t, edge, y)
        |                                  \
   controlnet(x, edge, t, y) -> [res...]    unet(x, t, y, control=[res...])
        |  (zero-conv 残差)                   |  解码器 skip += res
        +-----------------------------------+
                    |
            f = c_out*F + c_skip*x_t
```

---

## 5. 环境搭建（`cm` 环境，你在自己的 shell 跑）

> 说明：助手所在的容器是**受限只读环境**（项目目录可写、GPU 不可达、conda 目录只读），无法替你建环境/训练。以下命令请在你**有 GPU、可写 conda** 的交互式 shell 执行。

```bash
bash scripts/setup_cm_env.sh        # 建 cm 环境（如驱动版本不同，改 CU_TAG=cu117/cu121）
conda activate cm
```
- 关键依赖：torch+cuda、piq、blobfile、einops、opencv、mpi4py(经 conda-forge 自带 openmpi)、gradio。
- **flash-attn 故意跳过**（装不上，已用纯 torch 注意力回退）。
- 验证：脚本末尾会打印 `torch.cuda.is_available()` 和 GPU 数。

---

## 6. 运行手册

```bash
# (一次) 下载冻结主干（ImageNet-64 CM，约 1.2GB）
bash scripts/download_checkpoints.sh checkpoints

# 一键跑完整 demo 流程（CPU 也能跑的微缩版）
bash scripts/run_demo_pipeline.sh
```
脚本内含 **GPU 正式训练**命令（`mpiexec -n 4 ... --use_fp16 True --loss_norm lpips`）。关键超参：
- 架构必须匹配 checkpoint：`--num_channels 192 --num_res_blocks 3 --num_head_channels 64 --attention_resolutions 32,16,8 --class_cond True --use_scale_shift_norm True --resblock_updown True`。
- `--use_controlnet True --controlnet_init_checkpoint <ckpt>` 触发 ControlNet 初始化 + 冻结。
- `--canny_random True` 训练期抖动 Canny 阈值 → 对老师现场用的任意阈值更鲁棒。

---

## 7. 面试现场 Demo 指南（重点准备）

老师会**当场从网页找图、手动测效果**。推荐用 Gradio 网页界面：

```bash
conda activate cm
python scripts/demo.py --model_path runs/edge_ct_imagenet64/model<NNN>.pt \
   --use_controlnet True --gradio --share
```
- 打开终端里的本地/公网链接 → **把任意图片拖进去** → 实时出"原图 | Canny 边缘 | 生成图"。
- 现场可调：**Canny 高/低阈值**（演示边缘多少对结构的影响）、**类别 id**（语义由类控制、结构由边缘控制，可演示二者解耦）、**1 步 vs 多步**（展示一致性模型的单步生成卖点）、随机种子（同一边缘出多样结果）。
- 鲁棒性已处理：URL/灰度图/RGBA/超大图/极端长宽比都先 `load_image_any` + 中心裁剪 resize 到模型分辨率。

**讲解话术建议**：先用 1 步生成展示"consistency model 单步即出图"，再调 Canny 阈值让老师看到"边缘越密 → 结构约束越强"，最后改类别 id 说明"边缘管结构、类别管语义，二者解耦"。

> 注意：ImageNet-64 主干生成的是 64×64 小图，**重点演示的是"结构服从边缘"这个控制能力**，而非高清画质。若要更好观感，可在 LSUN-256 主干上重做（更慢更吃显存）。

---

## 8. 评测方法

- **边缘服从度（核心指标）**：对生成图重提 Canny，与输入边缘算 **F1 / dilated-IoU**（带 2px 容差）。`scripts/eval_edge_fidelity.py`。
- **生成质量 FID**：用 `evaluations/`（需参考 batch；demo 阶段可选，需额外装 tf/clip）。
- **对照实验（消融）**：同一边缘、`edge=None` vs `edge=c`，看结构是否真的被 c 控制；不同 Canny 阈值下 F1 曲线。

---

## 9. 进度与 TODO 清单

**已完成（代码 + CPU 验证）**
- [x] U-Net 接入 control 残差 + 通道元数据 + 注意力回退
- [x] `cm/controlnet.py`：ControlNet 分支 / ControlledModel / 权重同步
- [x] piq 惰性化；script_util 接入 controlnet + 修复 loss_norm
- [x] 数据集在线 Canny；edge_util 共享工具
- [x] cm_train 检查点初始化 + 冻结主干 + 边缘数据接入
- [x] image_sample 边缘条件采样；demo.py（CLI+Gradio）；prep/eval 脚本
- [x] CPU smoke test 17 项全通过（形状/零初始化等价/边界条件/冻结/梯度/CT 步）
- [x] environment.yml / setup / download / run 脚本

**进度（需 GPU + cm 环境）**
- [x] `bash scripts/setup_cm_env.sh` 建 `cm` 环境（cu118 + 国内镜像）
- [x] `bash scripts/download_checkpoints.sh` 下 `cd_imagenet64_lpips.pt`
- [x] 微缩流程跑通：合成数据上 ControlNet 训了 11200 步，三联图**确认结构服从边缘**（机制验证通过）
- [ ] **当前下一步**：真实 ImageNet-64 数据（`scripts/prep_imagenet64.py`）+ LPIPS 重训（GPU，`mpiexec -n 4`，带 `--lr_anneal_steps`/`--save_keep_last`）
- [ ] 训练后 `python scripts/demo.py --gradio` 演练现场 demo
- [ ] （可选）LSUN-256 主干或 +ESRGAN 超分提升输出观感（见 §12.1）

---

## 10. 三种方法对比、相关工作与后续优化

- **路线对比**：见 §2.2。
- **相关工作**：ControlNet (Zhang 2023, 零卷积/可训练副本)；T2I-Adapter (Mou 2023, 更轻量的适配器)；GLIGEN (Li 2023, 门控注意力做 grounding)；通道拼接系 SR3/Palette/InstructPix2Pix。一致性模型侧：CTM (Kim 2024)、LCM / LCM-LoRA (Luo 2023, 社区已有 LCM+ControlNet 组合) 是有力佐证——"一致性模型 + ControlNet 控制"是可行且已被验证的方向。
- **后续优化**：① 训练期随机 Canny 阈值 + 随机 dropout 边缘（classifier-free 风格）→ 推理可调控制强度；② 多步采样提质（CT 用 `--ts 0,106,200`）；③ 换 LSUN-256 主干提观感；④ 边缘以外的条件（HED/深度/分割）只需换 hint 通道与数据；⑤ 控制强度缩放（推理时给 control 残差乘系数）。

---

## 11. 面试问答预演（针对你的薄弱项：正向讲清"为什么"）

**Q：加了边缘条件，一致性模型的边界条件还成立吗？**
A：成立，而且是架构层面保证的。`f = c_skip·x + c_out·F`，`c_skip/c_out` 是 σ 的固定函数；在 ε 处 `c_out=0` 把整个网络输出（含 ControlNet）清零，`c_skip=1`，所以 `f(x,ε,c)=x` 恒成立，与 c 无关。（smoke test 已数值验证。）

**Q：为什么零卷积初始化能稳定训练？它一开始输出 0，怎么学得动？**
A：输出 0 不代表梯度 0。卷积权重的梯度 = 上游梯度 ⊗ 输入特征；输入非零则权重收到非零梯度。所以零卷积第一步就开始更新，一旦非零，梯度再往上游编码器副本流。起点等价于原模型 → 不破坏预训练能力 → 稳。

**Q：为什么选 CT 不选 CD？**
A：CD 的教师是无边缘条件的扩散模型，把边缘喂给它要么报错要么没意义；也没有现成的边缘条件教师。CT 无教师，在线与 target 同吃一张边缘图，"同轨道+同条件→一致"天然成立，且能从预训练 CM 检查点微调，省算力。

**Q：边缘条件怎么进网络的？为什么不随 t 变？**
A：通过 `model_kwargs` 字典透传到 `model(c_in*x_t, rescaled_t, edge=...)`，全链路（训练/CD/CT/采样）自动带上。边缘图走独立的 hint 编码器、不经过 `c_in` 缩放、不随 t 变，正体现"c 是跨所有噪声档固定的结构约束"。

**Q：你只训了 ControlNet 分支，主干冻结，这能保证生成质量吗？**
A：主干是已训练好的强一致性模型，冻结保留其生成能力；ControlNet 只学"如何把结构约束注入"。这正是 ControlNet 的核心思想——用少量可训练参数引导一个强生成先验，既省算力又防遗忘。

---

## 12. 关键细节 FAQ（实操中必然会问到的）

### 12.1 能不能测试/生成比 64×64 更大的图？

要分清**输入**和**输出**：
- **输入图可以是任意尺寸**：demo 工具会先中心裁剪 + 高质量缩放到模型分辨率再提 Canny，所以老师拿任何尺寸的网页图都行。
- **输出分辨率被主干锁死在 64×64**。原因（面试可讲的因果链）：扩散/一致性模型是在**固定分辨率**上训练的——卷积本身与分辨率无关，但 ① 注意力层、② 噪声 schedule（σ 与像素统计的关系）、③ 学到的权重统计，全部是按 64×64 标定的。换个分辨率跑，这些都不匹配 → 输出是噪声/伪影。**不能把 ImageNet-64 模型直接当 256 用。**

想要更大输出，三条路：
1. **换高分辨率主干**：仓库有 LSUN-256 的 CM 检查点（`cd_bedroom256` / `cd_cat256`），ControlNet 改在 256 上训 → 输出 256×256。代价：① 这些主干是**单类别**（卧室/猫），不是通用域；② 256 训练更吃显存、更慢。
2. **后接超分**：先用 64×64 边缘条件生成（结构由边缘控制），再用独立的超分模型（如 Real-ESRGAN）放大到 256/512。结构来自第一步，细节由超分补。**现场演示提升观感最实用**的一招。
3. **仅显示放大**：把 64×64 输出用双三次插值放大显示（只是看得清，不是真高清）。

**结论**：通用域（任意网页图）目前能用的最高分辨率就是 **ImageNet-64 的 64×64**（CIFAR 是 32，没有更高的通用 CM 主干）。面试演示建议：64×64 + 强调"控制能力"，或 64×64 + ESRGAN 包装观感。

### 12.2 怎么下载 ImageNet-64 训练数据

数据源：`ChocolateDave/imagenet-64`（HF，标准 Chrabaszcz 下采样版，128 万张，**标签就是 ImageNet 0-999，与主干类别条件完全对齐**）。脚本 `scripts/prep_imagenet64.py` 已写好，把它转成 `{label}_{idx}.png`（数据加载器识别数字前缀直接当类别 id，子集缺类也不会错位）。

```bash
conda activate cm
pip install datasets
export HF_HOME=/data1/<你的目录>/hf_cache      # 缓存放大盘(/home 只有 ~59G)
# 取 10 万张随机子集(覆盖全部 1000 类，足够训 ControlNet)
python scripts/prep_imagenet64.py --out_dir data/imagenet64 --n 100000
```
磁盘：parquet 缓存 ~10.6GB（一次性），10 万张 PNG ~0.5GB。要更省盘加 `--streaming`（边下边转，不先下全量）。

### 12.3 想用 LPIPS 距离，需要重新微调吗？

**需要。** `loss_norm`（l2 / lpips）是**训练期**一致性损失的度量，只在训练时起作用、塑造模型学到什么；**采样/推理时根本不算 loss**，所以不能"切换"——要享受 LPIPS 的收益必须用 `--loss_norm lpips` **重训（或在已有权重上续训）**。LPIPS 是感知距离，比逐像素 l2 更贴合人眼，**生成更锐利真实**（论文实测 LPIPS 最优）。正式训练直接 `--loss_norm lpips` 从头训即可（前面的 l2 模型是玩具数据上的，不必继承）。

### 12.4 现在用 l2 微调出来的模型能拿去展示吗？

**作为"机制验证"可以，作为"面试现场 demo"不行。** 当前模型是在**合成几何形状**上用 l2 训的：
- 它**只见过几何形状**，没见过真实图像的边缘分布 → 直接拿真实网页图测，效果会差（域差）。
- 画质偏糊（玩具数据 + l2 + onestep + 强加 class 0）。
- 但它**成功证明了"边缘→结构"的控制链路是对的**（三联图里结构跟着边缘走）——这是阶段性成果，适合讲"机制跑通了"。

**面试要拿得出手 → 必须在真实 ImageNet-64 上用 LPIPS 重训**（§12.2 数据 + 脚本底部 GPU 命令，记得带 `--lr_anneal_steps`（防死循环）、`--save_keep_last`（防爆盘）、`--loss_norm lpips`）。

### 12.5 踩过的坑（已修，备忘）

| 坑 | 现象 | 修复 |
|---|---|---|
| `lr_anneal_steps=0` 时 `run_loop` 死循环 | `total_training_steps` 被忽略，一直训、每 save_interval 存一个 ~5.8GB ckpt，撑爆 1.3TB 磁盘 | 训练命令必须带 `--lr_anneal_steps N` 才会停 |
| 每次 save 存全量(主干+ControlNet)×{model,ema,target}+opt ≈5.8GB | 长训练填满盘 | 新增 `--save_keep_last N` 只留最近 N 组；后续可优化为只存 controlnet 分支 |
| fp32 下 target EMA 尺寸不匹配崩溃 | 原代码 EMA 假设 fp16 扁平 master params | 已修：fp32 用逐参数列表 |
| image_sample 默认 `training_mode=edm` | 采样用错 c_skip/c_out → 出图错 | 采样要带 `--training_mode consistency_training` |
| 数字类别前缀按字符串排序 | "10"排到"2"前、子集缺类错位 → 类别条件被打乱 | 已修：数字前缀直接当 int 类别 id |
