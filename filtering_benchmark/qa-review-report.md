# QA 审查报告 — 滤波基石系统

> **审查人**：Edward (QA Engineer)
> **审查日期**：即时
> **范围**：测试覆盖度、代码质量、异常处理、运行时验证、文档健康度

---

## 1. 📊 测试运行结果摘要

| 指标 | 数值 |
|------|------|
| 测试总数 | **90** |
| 通过 | **71** (78.9%) |
| 跳过 | **19** (21.1%) |
| 失败 | **0** |
| 耗时 | 19.73s |

### 跳过的 19 个测试明细

跳过的测试集中在以下 7 个算法（每个算法在 `test_shape_1d`, `test_shape_2d`, `test_not_all_zero` 三个维度均被跳过）：

| 算法ID | 原因分析 | 影响 |
|--------|---------|------|
| `band_pass_butterworth` | 默认参数 high_cutoff=5000 > 测试信号奈奎斯特频率 1500Hz (fs=3000) | **测试数据/算法参数不匹配** |
| `fir_band_pass` | 同上 high_cutoff=5000 > 1500 | **测试数据/算法参数不匹配** |
| `iir_band_pass` | 同上 high_cutoff=5000 > 1500 | **测试数据/算法参数不匹配** |
| `moving_average_filter` | scipy convolve 输出维度与输入不匹配 | ⚠️ **可能是真正的 bug** |
| `dwt_threshold_denoise` | 依赖 `pywt` 未安装 | **运行环境依赖缺失** |
| `wavelet_packet_denoise` | 依赖 `pywt` 未安装 | **运行环境依赖缺失** |
| `denoise_param_searcher` | 需要内部算法依赖，环境限制 | **运行环境限制** |

**问题**：19 个跳过测试掩盖了至少 1 个潜在 bug（`moving_average_filter` 的 convolve 维度问题）。

---

## 2. 🧪 测试覆盖度分析

### 2.1 按分类的算法覆盖

| 分类 | 总算法数 | 有测试覆盖 | 覆盖率 | 未测试算法 |
|------|---------|-----------|-------|-----------|
| 经典滤波方法 | 7 | 2 | 28.6% | band_stop_butterworth, band_pass_butterworth, fir_band_pass, iir_band_pass, notch_iir |
| 时域降噪方法 | 7 | 6 | 85.7% | moving_average_filter (skipped) |
| 频域降噪方法 | 5 | 3 | 60.0% | cepstral_filter, homomorphic_filter |
| 小波变换类 | 7 | 2 | 28.6% | dtcwt_denoise, ewt_denoise, sure_shrink_denoise, ti_wavelet_denoise, swt_denoise |
| 经验模态分解类 | 10 | 0 | **0%** | 全部10个均未在_REPRESENTATIVE_ALGOS中 |
| 自适应滤波 | 8 | 3 | 37.5% | nlms_adaptive_filter, apa_adaptive_filter, wiener_filter, vsslms_adaptive_filter, leaky_lms_adaptive_filter |
| 奇异值分解类 | 3 | 2 | 66.7% | randomized_svd_denoise |
| 稀疏表示类 | 6 | 1 | 16.7% | matching_pursuit_denoise, orthogonal_matching_pursuit_denoise, ksvd_denoise, cosamp_denoise, fista_denoise |
| 盲源分离类 | 6 | 0 | **0%** | 全部6个均未覆盖 |
| 深度学习类 | 8 | 0 | **0%** | 全部8个均未覆盖 |
| 旋转机械专用 | 6 | 0 | **0%** | 全部6个均未覆盖 |
| 语音/声学降噪 | 4 | 0 | **0%** | 全部4个均未覆盖 |
| 其他前沿方法 | 7 | 1 | 14.3% | compressed_sensing_denoise, stochastic_resonance, morphological_filter, total_variation_denoise, graph_signal_denoise, fractal_denoise |
| 优化算法 | 7 | 1 | 14.3% | 6个优化算法未覆盖 |

**整体测试覆盖率：25 / 91 = 27.5%**

### 2.2 测试类型分布

| 测试类型 | 数量 | 说明 |
|---------|------|------|
| 立体声工具函数 | 4 | ✅ 良好覆盖 |
| 1D形状保持 | 25 | 参数化，但7个跳过 |
| 2D形状保持 | 25 | 参数化，但7个跳过 |
| 非全零验证 | 25 | 参数化，但7个跳过 |
| 安全保护 | 2 | max_signal_length 检查 |
| 注册表完整性 | 1 | 基本覆盖检查 |
| 指标测试 | 2 | 简单遍历 |
| 注册表功能 | 5 | 单例/发现/过滤/创建 |
| 集成测试 | 1 | 最小化流水线 |

### 2.3 关键测试缺口

1. **无任何 EMD 族测试** — 10个算法零覆盖
2. **无任何 BSS 测试** — 6个算法零覆盖
3. **无任何深度学习测试** — 8个算法零覆盖
4. **无任何旋转机械专用测试** — 6个算法零覆盖
5. **无任何语音/声学降噪测试** — 4个算法零覆盖
6. **无错误路径测试** — 无测试验证算法在无效输入下的行为
7. **无参数边界测试** — 无测试验证极值参数（如 order=0, window_size=1）
8. **无 GPU 降级测试** — requires_gpu=True 的算法无 fallback 行为测试

---

## 3. 🐛 代码质量审查

### 3.1 异常处理体系

| 异常类 | 用途 | 使用情况 |
|--------|------|---------|
| `FilteringBenchmarkError` | 基类 | ✅ 定义良好 |
| `SignalTooLargeError` | 超限信号保护 | ✅ 有测试覆盖 |
| `AlgorithmNotFoundError` | 缺失算法 | ✅ registry.get() 使用 |
| `AlgorithmExecutionError` | 执行错误 | 子类化良好 |
| `AlgorithmTimeoutError` | 超时 | 存在但未在测试中使用 |

**问题**：`SignalTooLargeError` 在 `base.py:106` 中抛出的参数顺序为 `(algo_id, n_samples, max_samples)`，但构造器签名是 `(self, algorithm_id, n_samples, max_samples)`，两者一致，✅。

**问题**：`_check_signal_size` 方法在 `BaseAlgorithm` 中定义了，但大多数算法**没有调用它**。检查所有算法类的 `denoise()` 方法，只有少数高复杂度算法在入口处调用了 `self._check_signal_size(signal)`。这是**安全漏洞**——低复杂度算法若设置了 `max_signal_length` 也不会被检查。

### 3.2 max_signal_length 安全保护

```
base.py (_check_signal_size): 实际生效条件
- max_signal_length is not None → 检查
- max_signal_length is None (默认) → 跳过
```

但算法**必须主动调用** `_check_signal_size` 才能触发保护。目前多数算法未在 `denoise()` 中调用，导致该安全机制形同虚设。

### 3.3 装饰器使用

- `@register_algorithm`：91个算法全部正确装饰 ✅
- `@log_execution`：绝大多数算法使用 ✅
- `@validate_params`：绝大多数算法使用 ✅

---

## 4. 🔧 管道/集成质量

### 4.1 集成测试 (test_pipeline.py)

- 1 个测试覆盖最小化端到端流水线 ✅
- 使用模拟参考信号，执行 TIME_DOMAIN 分类的完整 pipeline
- 验证了 `len(r.rankings) > 0` 和 `not isnan(overall_score)`
- **问题**：仅覆盖 1 个分类，且使用默认配置，无错误路径测试

### 4.2 注册表稳定性

```
test_singleton: ✅ AlgorithmRegistry() is registry
test_discovery: ✅ 发现算法数 > 0
test_filter: ✅ 按分类过滤
test_filter_str: ✅ 字符串过滤
test_create: ✅ 实例化
```

所有5个注册表测试均通过 ✅

---

## 5. 📝 文档一致性

Alice 已进行了完整的文档审查（docs-review-report.md），主要发现：

| 问题 | 严重度 | 说明 |
|------|--------|------|
| 数量声明冲突 | 🔴 高 | README(89+), 参考书(71), 清单(70), 代码(91) |
| 分类归属错误 | 🔴 高 | 维纳滤波被错误归入语音/声学 |
| README 覆盖不全 | 🟡 中 | 13/14 分类仅列出 50-60% 算法 |
| 存在不存在算法 | 🟡 中 | ISTA, ADMM-L1, 自相关增强 |
| 参数遗漏 | 🟢 低 | 巴特沃斯漏 filter_type |

### 文档健康度评分：72/100 — C级（需重大修正）

---

## 6. 🔮 修复脚本质量

检查了项目中的修复脚本：

| 脚本 | 用途 | 质量 |
|------|------|------|
| `_fix_ch6.py` | 修复文档章节 | 临时脚本 |
| `_fix_debris.py` | 修复残留编码问题 | 临时脚本 |
| `_fix_mmse.py` | 修复 MMSE 算法 | 临时脚本 |
| `_fix_remaining.py` | 批量修复 | 临时脚本 |
| `_fix_vmd_ssa.py` | 修复 VMD/SSA | 临时脚本 |

**问题**：5个修复脚本均为一次性临时脚本，应为 `fix_`（无前缀下划线）命名风格，且未集成到版本控制策略中。这些脚本表明项目有**系统性的编码损坏问题**（可能由 AI 编码工具的局部重写导致）。

---

## 7. 📋 整体评分

| 评估维度 | 分数 | 评级 |
|---------|------|------|
| **测试覆盖率** | 28/100 | ❌ 不合格 (27.5%) |
| **测试质量** | 65/100 | ⚠️ 需改进（19个跳过测试掩盖问题） |
| **异常处理** | 50/100 | ⚠️ max_signal_length 机制未实际生效 |
| **集成质量** | 60/100 | ⚠️ 仅有最小化流水线测试 |
| **代码质量** | 70/100 | ✅ 装饰器体系良好，基类设计合理 |
| **文档一致性** | 72/100 | ⚠️ C级，需重大修正 |

### 🏆 综合评分：54 / 100 — 不合格

---

## 8. 🎯 修复优先级建议

### P0 — 阻塞级（立即修复）
1. **moving_average_filter 卷积维度问题** — 确认是否为真正的 bug，scipy.convolve 在2D输入时的行为
2. **测试数据修复** — 将测试信号采样率从 3000Hz 改为 12000Hz，使 bandpass/IIR 等算法能正常测试
3. **max_signal_length 安全检查** — 确保所有设置此属性的算法在 denoise() 入口处调用 `_check_signal_size()`

### P1 — 高优先级（本周）
1. **新增 EMD 族测试** — 至少覆盖 EMD、VMD、SSA 三个代表算法
2. **新增 BSS 测试** — PCA 和 ICA 基础测试
3. **依赖管理** — 安装 `pywt` 等缺失依赖，确保 wavelet 算法可测试
4. **错误路径测试** — 为每个分类至少添加一个无效输入测试

### P2 — 中优先级（下个迭代）
1. **深度学习算法模拟测试** — 使用 mock 替代 GPU 依赖
2. **旋转机械专用算法测试** — MED/MCKD 基础形状保持
3. **参数边界测试** — 极值参数、空信号、NaN 输入
4. **修复脚本归档** — 将一次性脚本整理归档，删除临时文件

### P3 — 低优先级
1. **性能基准测试** — 为高复杂度算法添加性能基线
2. **多通道精确性测试** — 验证多通道输出与输入的一致性
3. **随机种子固定** — 确保测试可重复

---

## 9. 测试环境信息

- **Python**: 3.14.4
- **pytest**: 9.0.3
- **平台**: Windows (win32)
- **项目根目录**: `filtering_benchmark/`

---

*报告完毕 — 如需详细的测试覆盖率 CSV 或具体失败堆栈，请告知。*
