# 滤波基石系统 — 文档一致性审查报告

> 审查人：Alice (Product Manager)
> 审查日期：即时
> 范围：README.md、算法参考书、算法清单 vs 实际代码

---

## 1. 📊 数量声明冲突

| 来源 | 声称数量 | 分类数 | 与实际偏差 |
|------|---------|-------|-----------|
| **README.md** (L12, L20) | **89+** 种 | **14** 大类 | 少约 2 种 |
| **算法参考书** (L3, §1.5) | **71** 种 | **13** 大类 | 少 **20** 种 |
| **算法清单** (汇总表) | **70** 种 | **13** 大类 | 少 **21** 种 |
| **实际代码** | **91** 个 `class ...(BaseAlgorithm)` | **14** 个子目录 | ✅ 基准 |

三个文档互不相同，且均低于实际代码数量。参考书和算法清单严重滞后，少了约 20 个算法，主要是因为：

- **优化算法**：代码中有 7 个注册算法（joint_optimization.py），但参考书的分类表中列为理论章节（第15-16章），未计入 71 总数
- **参考书**第1.5章的分类表（13类=71个）与目录章节数量（算法章节涵盖超过84个算法ID）也不一致
- **算法清单**汇总为70个，但附录说明提到"部分算法标注为'社区实现'的未计入"——标准不统一

### 实际代码各分类算法数量

| 分类目录 | 实际数量 | README声称 | 参考书声称 |
|---------|---------|-----------|-----------|
| classical | 7 | ~5 | 7 |
| time_domain | 7 | 7 | 5 |
| freq_domain | 5 | 3 | 4 |
| adaptive | 8 | 4 | 6 |
| wavelet | 7 | 2 | 5 |
| emd_family | 10 | 8 | 9 |
| svd | 3 | 2 | 2 |
| sparse | 6 | 4 | 4 |
| bss | 6 | 4 | 4 |
| deep_learning | 8 | 8 | 8 |
| acoustic | 4 | 2 | 4 |
| rotating_specific | 6 | 4 | 6 |
| advanced | 7 | 4 | 7 |
| optimization | 7 | 4 | 0 (未计入) |
| **合计** | **91** | **~61** | **71** |

---

## 2. 📋 README 分类覆盖缺陷

README 的 14 个分类中，多数分类描述的算法仅覆盖了实际代码的 **不到一半**。

### 严重缺失（覆盖 < 50%）

| 分类 | 实际算法 | README 仅列出 | 遗漏算法 |
|------|---------|--------------|---------|
| **频域降噪** | 5 (FFT理想低通, 谱减法, 倒谱分析, 同态滤波, 多带谱减法) | 3 | **倒谱分析滤波(CepstralFilter)**、**同态滤波(HomomorphicFilter)** |
| **小波变换** | 7 (DWT, WPT, DTCWT, SWT, EWT, SureShrink, TI-Wavelet) | 2 | DTCWT、SWT、EWT、SureShrink、TI小波 (5个) |
| **自适应滤波** | 8 (LMS, NLMS, RLS, APA, Kalman, Wiener, VSS-LMS, Leaky-LMS) | 4 | APA、**维纳滤波**、变步长LMS、泄露LMS |
| **稀疏表示** | 6 (MP, OMP, BPDN, K-SVD, CoSaMP, FISTA) | 4 | MP(匹配追踪)、K-SVD、CoSaMP、FISTA — README 反而提到不存在的 **ISTA** 和 **ADMM-L1** |
| **盲源分离** | 6 (PCA, ICA, FastICA, SOBI, KernelPCA, JADE) | 4 | **PCA降噪**、**核PCA** |
| **旋转机械专用** | 6 (MED, MCKD, MOMEDA, 谱峭度, 包络解调, 循环平稳) | 4 | MOMEDA、包络解调、循环平稳 — README 提到不存在的"自相关增强" |
| **前沿方法** | 7 (压缩感知, 随机共振, 形态学, TV, NLM, 图信号, 分形) | 4 | **压缩感知**、**形态学滤波**、**全变分去噪(TV)** |

### 分类归属错误

| 算法 | README 所在分类 | 代码实际分类 | 问题 |
|-----|----------------|------------|------|
| **维纳滤波 (WienerFilter)** | 语音/声学 | **自适应滤波 (ADAPTIVE)** | ❌ 分类错误 |
| **谱减法增强** | 语音/声学 | **频域降噪 (FREQ_DOMAIN)** | ❌ 分类模糊 |
| **倒谱分析滤波 (CepstralFilter)** | 频域降噪 ❌ 未列出 | 频域降噪 (FREQ_DOMAIN) | ❌ 未收录 |
| **同态滤波 (HomomorphicFilter)** | 频域降噪 ❌ 未列出 | 频域降噪 (FREQ_DOMAIN) | ❌ 未收录 |

### README 提到的不存在的算法
- **ISTA** — 代码中无此算法（有 FISTA 但无 ISTA）
- **ADMM-L1** — 代码中无此算法
- **自相关增强** — 代码中无此独立算法

---

## 3. 🔍 参数一致性抽查结果

### 抽样1：巴特沃斯低通滤波 (low_pass_butterworth)

| 参数 | 代码 default_params | 参考书记录 | 一致？ |
|-----|-------------------|-----------|-------|
| `cutoff_freq` | 1000.0 | 1000.0 ✅ | ✅ |
| `order` | 4 | 4 ✅ | ✅ |
| **`filter_type`** | **"lowpass"** | **未记录** | **❌ 遗漏** |

**结论**：代码有 3 个参数，参考书仅记录 2 个，漏掉 `filter_type` 参数。

### 抽样2：LMS自适应滤波 (LMSAdaptiveFilter)

| 参数 | 代码 default_params | 参考书记录 | 一致？ |
|-----|-------------------|-----------|-------|
| `filter_order` | 32 | 32 ✅ | ✅ |
| `mu` | 0.01 | 0.01 ✅ | ✅ |

**结论**：完全一致 ✅

### 抽样3：VMD降噪 (VmdDenoise)

| 参数 | 代码 default_params | 参考书记录 | 一致？ |
|-----|-------------------|-----------|-------|
| `K` | 5 | 5 ✅ | ✅ |
| `alpha` | 2000 | 2000 ✅ | ✅ |
| `tau` | 0.0 | 0.0 ✅ | ✅ |
| `DC` | False | False ✅ | ✅ |
| `init` | 1 | 1 ✅ | ✅ |
| `tol` | 1e-6 | 1e-6 ✅ | ✅ |
| `remove_first_n` | 1 | 1 ✅ | ✅ |

**结论**：完全一致 ✅

### 抽样4：小波阈值降噪 (DwtThresholdDenoise)
- 代码 `default_params`: wavelet="db4", level=5, threshold_mode="soft", threshold_rule="universal"
- 参考书写录: wavelet="db4", level=5, threshold_mode="soft", threshold_rule="universal"
- **结论**: 完全一致 ✅

### 抽样5：PCA降噪 (PCADenoise)
- 代码 `default_params`: n_components=None, retained_variance=0.9
- 参考书记录: n_components=None, retained_variance=0.9
- **结论**: 完全一致 ✅

### 参数一致性总结

| 算法 | 代码参数数 | 参考书记参数数 | 差异 |
|-----|-----------|--------------|------|
| 巴特沃斯低通 | 3 | 2 | 漏 `filter_type` |
| LMS | 2 | 2 | ✅ |
| VMD | 7 | 7 | ✅ |
| DWT | 4 | 4 | ✅ |
| PCA | 2 | 2 | ✅ |
| **整体** | **18** | **17** | **1处遗漏** |

---

## 4. API 示例验证

检查 README L179-L201 中的 Python API 示例：

| 代码引用 | 实际存在？ | 文件/位置 |
|---------|-----------|----------|
| `from src.algorithms import registry` | ✅ | `src/algorithms/__init__.py` L9 导出 |
| `from src.algorithms.base import BaseAlgorithm` | ✅ | `src/algorithms/base.py` |
| `registry.ensure_discovered()` | ✅ | `src/algorithms/registry.py` L148 |
| `registry.list_algorithms()` | ✅ | `src/algorithms/registry.py` L86 |
| `registry.count()` | ✅ | `src/algorithms/registry.py` L120 |
| `registry.create_instance("low_pass_butterworth", ...)` | ✅ | 算法类存在 |
| `from src.config import ConfigLoader` | ✅ | `src/config/loader.py` L22 |
| `from src.pipeline.orchestrator import PipelineOrchestrator` | ✅ | `src/pipeline/orchestrator.py` L48 |

结论：✅ **所有 API 示例均有效**。

CLI 示例中的 `filter-bench` 命令依赖于 CLI 入口 `src/cli/main.py`，语法层面合理。

---

## 5. 📝 文档完整性缺陷清单

### 缺陷1：算法数量严重滞后（高优先级）
- 参考书声称 "71种" 比实际 **91种** 少 20 个
- 算法清单声称 "70种" 比实际少 21 个
- README 声称 "89+" 勉强接近但仍有差距（少2个）
- 用户若基于参考书做技术选型，会遗漏大量可用算法

### 缺陷2：分类归属错误（高优先级）
- 维纳滤波被 README 归入"语音/声学"，但代码中属于"自适应滤波"
- 谱减法增强被归入"语音/声学"，但代码中属于"频域降噪"
- 这会严重误导用户查找算法

### 缺陷3：覆盖算法列表不完整（中优先级）
- 13/14 个分类的 README 描述均不完整，平均仅列出实际算法的 50-60%
- 个别分类（如小波变换）仅列出 2/7 个算法

### 缺陷4：参考书参数文档遗漏（低优先级）
- 巴特沃斯低通滤波器漏记 `filter_type` 参数
- 其他 4 个抽样算法参数一致，说明总体质量尚可

### 缺陷5：README 中提及不存在算法（中优先级）
- "ISTA"、"ADMM-L1"、"自相关增强" 在代码中不存在
- 可能是早期设计或更名前的残留记录

### 缺陷6：CWRU 数据集仅被提及路径，无使用说明（低优先级）
- 项目结构显示 `cwru/` 目录但 README 无如何使用 CWRU 数据的说明

### 缺陷7：技术栈版本不一致（低优先级）
- README 徽章显示 Python ≥ 3.11
- 参考书 §1.2 写 Python 3.9+
- 二者矛盾

---

## 6. 用户体验影响评估

| 问题 | 影响程度 | 用户影响描述 |
|-----|---------|------------|
| 算法数量不准确 | 🔴 严重 | 用户以为只有71种算法可用，错失大量潜在的有效工具 |
| 分类归属错误 | 🔴 严重 | 用户在"语音/声学"下找不到维纳滤波的真实参数文档 |
| 覆盖列表不完整 | 🟡 中等 | 用户看到 README 列表以为系统能力有限 |
| 不存在算法记录 | 🟡 中等 | 用户尝试使用 ISTA/ADMM-L1 会报错 |
| 参数文档遗漏 | 🟢 轻微 | 仅影响 filter_type 参数，对功能使用影响小 |
| CWRU 缺少说明 | 🟢 轻微 | 熟悉该数据集的用户可自行使用 |

---

## 7. 整体文档健康度评分

| 评估维度 | 得分 | 说明 |
|---------|------|------|
| **数量准确性** | 60/100 | 三个文档三个数，均不准确 |
| **分类准确性** | 55/100 | README 分类覆盖严重不全，有错误归类 |
| **参数一致性** | 90/100 | 抽查 5 个算法仅 1 处遗漏，质量较好 |
| **示例可用性** | 95/100 | API 示例均可追溯至实际代码 |
| **安装文档** | 85/100 | 安装说明完整，依赖分组清晰 |
| **使用文档** | 65/100 | CLI/API 示例完整但分类信息误导性强 |

### 🏆 总分：72 / 100

**评级：C（需要重大修正）**

---

## 8. 🔧 修复优先级建议

### P0（立即修复）
1. **更新 README 算法数量** → 从 "89+ 种" 改为 "91 种"（或动态显示）
2. **修正维纳滤波分类归属** → 从"语音/声学"移至"自适应滤波"
3. **更新参考书 L3 声明** → 从 "71种" 改为 "91种"

### P1（本周修复）
1. **补全 README 各分类算法列表** → 确保每个分类列出至少 80% 的算法
2. **删除 README 中不存在的算法** → ISTA、ADMM-L1、自相关增强
3. **更新参考书 §1.5 分类表** → 增加 OPTIMIZATION 类别，更新数量

### P2（下个迭代修复）
1. **更新算法清单文档** → 同步至 91 个算法
2. **统一 Python 版本要求** → 统一为 ≥ 3.11
3. **补全参考书巴特沃斯参数** → 增加 `filter_type` 参数文档
4. **增加 CWRU 数据集使用说明**
