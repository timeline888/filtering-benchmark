"""
流水线编排器（Pipeline Orchestrator）。

连接所有8个Stage的总调度：
1. LoadStage → 2. PreprocessStage → 3. DenoiseStage → 4. EnvelopeStage
→ 5. MetricStage → 6. ScoringStage → 7. RankingStage → 8. OutputStage
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from loguru import logger

ProgressCallback = Optional[Callable[[int, str], None]]

from ..core.types import (
    SignalData, DenoisedResult, EnvelopeResult,
    AlgorithmRanking, EvaluationReport, AlgorithmCategory,
)
from ..core.exceptions import PipelineStageError
from ..core.constants import DEFAULT_METRIC_WEIGHTS, DEFAULT_TIMEOUT
from ..core.throttler import ComputationThrottle
from ..signal_io import SignalReaderFactory
from ..signal_io.preprocessor import SignalPreprocessor
from ..algorithms import registry as algo_registry
from ..envelope.analyzer import EnvelopeAnalyzer
from ..metrics import metric_registry
from ..scoring.ranking import RankingEngine, MetricNormalizer
from ..config import AppConfig
from ..signal_simulation.simulator import SignalSimulator


class PipelineContext:
    """流水线上下文，在各Stage之间传递数据"""
    def __init__(self, config: AppConfig):
        self.config = config
        self.original_signal: Optional[SignalData] = None
        self.preprocessed_signal: Optional[SignalData] = None
        self.denoised_results: Dict[str, DenoisedResult] = {}
        self.envelope_results: Dict[str, EnvelopeResult] = {}
        self.metric_scores: Dict[str, Dict[str, float]] = {}
        self.algorithm_metas: Dict[str, dict] = {}
        self.report: Optional[EvaluationReport] = None


class PipelineOrchestrator:
    """流水线编排器"""

    def __init__(self, config: AppConfig, progress_callback: ProgressCallback = None):
        self.config = config
        self.context = PipelineContext(config)
        self.progress_callback = progress_callback
        self._abort_flag = False

        # 从配置中创建资源节流器
        exec_cfg = config.execution
        self._throttle = ComputationThrottle(
            process_priority=exec_cfg.process_priority,
            max_cpu_cores=exec_cfg.max_cpu_cores,
            max_workers=exec_cfg.max_workers,
        )

    def _report_progress(self, percent: int, message: str):
        """报告进度（如果回调存在）"""
        if self.progress_callback:
            self.progress_callback(percent, message)

    def abort(self):
        """请求中止流水线"""
        self._abort_flag = True

    def run(self) -> EvaluationReport:
        """执行完整评估流水线（自动应用资源节流）"""
        with self._throttle.apply() as workers:
            self._effective_workers = workers
            logger.info("=" * 60)
            logger.info(f"开始评估流水线: {self.config.project.name}")
            logger.info(f"资源节流: 优先级={self.config.execution.process_priority}, "
                        f"并行数={workers}, "
                        f"CPU核心={self.config.execution.max_cpu_cores or '不限'}")
            logger.info("=" * 60)

            try:
                self._report_progress(0, "开始评估流水线...")
                self._stage_load()
                if self._abort_flag:
                    raise PipelineStageError("流水线被用户中止")
                self._report_progress(15, "信号加载完成，开始预处理...")

                self._stage_preprocess()
                if self._abort_flag:
                    raise PipelineStageError("流水线被用户中止")
                self._report_progress(20, "预处理完成，开始滤波降噪...")

                self._stage_denoise()
                if self._abort_flag:
                    raise PipelineStageError("流水线被用户中止")
                self._report_progress(60, "降噪完成，开始包络谱分析...")

                self._stage_envelope()
                if self._abort_flag:
                    raise PipelineStageError("流水线被用户中止")
                self._report_progress(70, "包络谱分析完成，开始指标计算...")

                self._stage_metrics()
                if self._abort_flag:
                    raise PipelineStageError("流水线被用户中止")
                self._report_progress(85, "指标计算完成，开始评分...")

                self._stage_scoring()
                self._report_progress(90, "评分完成，生成排名报告...")

                self._stage_ranking()
                self._report_progress(100, "评估完成!")
            except Exception as e:
                logger.error(f"流水线执行失败: {e}")
                if not isinstance(e, PipelineStageError):
                    raise PipelineStageError(f"流水线执行失败: {e}") from e
                raise

        logger.info("=" * 60)
        logger.info("评估完成!")
        logger.info(f"TOP-3 算法:")
        for rank in self.context.report.top_algorithms:
            logger.info(f"  #{rank.overall_rank}: {rank.algorithm_name} (score={rank.overall_score:.4f})")
        logger.info("=" * 60)

        return self.context.report

    # ---- Stage 1: 信号读取 ----

    def _stage_load(self):
        """Stage 1: 读取信号"""
        logger.info("[Stage 1/7] 读取信号...")
        signal_config = self.config.signal

        if signal_config.source.simulate_reference:
            # 仿真生成信号
            logger.info("  使用仿真信号模式")
            self.context.original_signal = self._generate_simulated_signal()
        else:
            # 读取文件
            file_path = signal_config.source.path
            if not file_path:
                raise PipelineStageError("未指定信号文件路径")
            reader = SignalReaderFactory.create_reader(file_path)
            self.context.original_signal = reader.read(file_path, channel=signal_config.source.channel)
            logger.info(f"  读取完成: {file_path}, shape={self.context.original_signal.data.shape}, fs={self.context.original_signal.sample_rate}Hz")

    def _generate_simulated_signal(self) -> SignalData:
        """生成仿真信号用于测试"""
        sim = SignalSimulator()
        fs = self.config.signal.source.sample_rate or 12000
        duration = 1.0
        noise_level = 0.3

        # 检查配置中是否有故障频率
        fault_cfg = self.config.envelope.fault_frequencies
        fault_freq = 78.5  # 默认BPFO
        if fault_cfg.mode == "manual" and fault_cfg.manual.bpfo:
            fault_freq = fault_cfg.manual.bpfo
        elif fault_cfg.mode == "auto":
            shaft_rpm = fault_cfg.auto.bearing_params.shaft_rpm or 1500
            shaft_freq = shaft_rpm / 60.0
            fault_freq = shaft_freq  # fallback

        bearing_signal = sim.bearing_fault(
            fs, duration, fault_freq=fault_freq, noise_level=noise_level
        )

        return SignalData(
            data=bearing_signal.reshape(1, -1),
            sample_rate=float(fs),
            channel_names=["simulated_bearing"],
            source_path="simulated",
            unit="m/s²",
        )

    # ---- Stage 2: 预处理 ----

    def _stage_preprocess(self):
        """Stage 2: 信号预处理"""
        logger.info("[Stage 2/7] 信号预处理...")
        preproc_config = self.config.signal.preprocessing
        preprocessor = SignalPreprocessor(preproc_config.model_dump())
        self.context.preprocessed_signal = preprocessor.process(self.context.original_signal)
        logger.info(f"  预处理完成: shape={self.context.preprocessed_signal.data.shape}")

    # ---- Stage 3: 并行滤波降噪 ----

    def _stage_denoise(self):
        """Stage 3: 并行/顺序滤波降噪（受 throttle 控制）"""
        logger.info("[Stage 3/7] 滤波降噪...")
        signal_data = self.context.preprocessed_signal
        if signal_data is None:
            raise PipelineStageError("预处理信号为空")

        algo_config = self.config.algorithms.selection

        # 获取算法列表
        algo_ids = self._select_algorithms(algo_config)

        logger.info(f"  共 {len(algo_ids)} 种算法待执行")
        self._report_progress(20, f"开始降噪 ({len(algo_ids)} 种算法)...")

        workers = getattr(self, '_effective_workers', 1)

        if workers <= 1 or len(algo_ids) <= 1:
            # ---- 顺序执行 ----
            logger.info(f"  执行模式: 顺序 (workers={workers})")
            self._run_algorithms_sequential(algo_ids, signal_data, algo_config)
        else:
            # ---- 并行执行 ----
            logger.info(f"  执行模式: 并行 (workers={workers})")
            self._run_algorithms_parallel(algo_ids, signal_data, algo_config, workers)

        total = len(algo_ids)
        success = len([r for r in self.context.denoised_results.values() if r.success])
        logger.info(f"  成功: {success}/{total}")

    def _run_algorithms_sequential(self, algo_ids, signal_data, algo_config):
        """顺序执行算法（单线程，带单算法超时保护）"""
        total = len(algo_ids)
        timeout = max(1, int(self.config.execution.timeout_per_algorithm_sec))
        # 使用单线程池以便对每个算法施加 future.result(timeout=...) 保护
        with ThreadPoolExecutor(max_workers=1) as executor:
            for idx, algo_id in enumerate(algo_ids):
                if self._abort_flag:
                    raise PipelineStageError("流水线被用户中止")
                future = executor.submit(
                    self._execute_single_algorithm,
                    idx, algo_id, signal_data, algo_config, total,
                )
                try:
                    future.result(timeout=timeout)
                except FuturesTimeoutError:
                    logger.warning(f"    算法 [{algo_id}] 执行超时 (>{timeout}s)，已标记失败")
                    # 超时的线程无法被强制 kill，继续下一个算法
                    self.context.denoised_results[algo_id] = DenoisedResult(
                        algorithm_id=algo_id,
                        algorithm_name=algo_id,
                        category=AlgorithmCategory.CLASSICAL_FILTER,
                        denoised_signal=signal_data.data.copy(),
                        execution_time=float(timeout),
                        success=False,
                        error_message=f"超时 (>{timeout}s)",
                    )

    def _run_algorithms_parallel(self, algo_ids, signal_data, algo_config, workers):
        """并行执行算法（线程池）"""
        total = len(algo_ids)
        # 预创建实例（避免线程安全竞争）
        tasks = []
        for algo_id in algo_ids:
            try:
                params = algo_config.param_overrides.get(algo_id, {})
                instance = algo_registry.create_instance(algo_id, **params)
                tasks.append((algo_id, instance, params))
            except Exception as e:
                logger.warning(f"    实例化失败 [{algo_id}]: {e}")
                self.context.denoised_results[algo_id] = DenoisedResult(
                    algorithm_id=algo_id,
                    algorithm_name=algo_id,
                    category=AlgorithmCategory.CLASSICAL_FILTER,
                    denoised_signal=signal_data.data.copy(),
                    execution_time=0,
                    success=False,
                    error_message=str(e),
                )

        completed = 0
        timeout = max(1, int(self.config.execution.timeout_per_algorithm_sec))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_algo = {}
            future_submit_time = {}
            for algo_id, instance, params in tasks:
                future = executor.submit(
                    self._denoise_one, algo_id, instance, params,
                    signal_data.data, signal_data.sample_rate
                )
                future_to_algo[future] = algo_id
                future_submit_time[future] = time.perf_counter()

            # 总体等待时间上界：单算法超时 × 任务数 / 并行度 + 缓冲
            overall_timeout = timeout * max(1, len(tasks) // max(1, workers)) + 30
            try:
                iterator = as_completed(future_to_algo, timeout=overall_timeout)
            except Exception:
                iterator = as_completed(future_to_algo)

            for future in iterator:
                if self._abort_flag:
                    # 不能真正取消已提交的任务，但可以停止等待
                    raise PipelineStageError("流水线被用户中止")
                algo_id = future_to_algo[future]
                completed += 1
                try:
                    # 单算法超时保护
                    result = future.result(timeout=timeout)
                    self.context.denoised_results[algo_id] = result
                    meta_name = result.algorithm_name
                    self.context.algorithm_metas[algo_id] = {
                        "name": meta_name,
                        "category": result.category,
                        "execution_time": result.execution_time,
                    }
                    logger.info(f"      完成: [{algo_id}] {result.execution_time:.3f}s")
                except FuturesTimeoutError:
                    logger.warning(f"    并行执行超时 [{algo_id}] (>{timeout}s)")
                    self.context.denoised_results[algo_id] = DenoisedResult(
                        algorithm_id=algo_id,
                        algorithm_name=algo_id,
                        category=AlgorithmCategory.CLASSICAL_FILTER,
                        denoised_signal=signal_data.data.copy(),
                        execution_time=float(timeout),
                        success=False,
                        error_message=f"超时 (>{timeout}s)",
                    )
                except Exception as e:
                    logger.warning(f"    并行执行失败 [{algo_id}]: {e}")
                    self.context.denoised_results[algo_id] = DenoisedResult(
                        algorithm_id=algo_id,
                        algorithm_name=algo_id,
                        category=AlgorithmCategory.CLASSICAL_FILTER,
                        denoised_signal=signal_data.data.copy(),
                        execution_time=0,
                        success=False,
                        error_message=str(e),
                    )

                progress = 20 + int((completed / total) * 35)
                self._report_progress(progress,
                    f"降噪进度 [{completed}/{total}]...")

    def _execute_single_algorithm(self, idx, algo_id, signal_data, algo_config, total):
        """执行单个算法（供顺序和异常处理使用）"""
        try:
            meta = algo_registry.get(algo_id)
            self._report_progress(
                20 + int((idx / total) * 35),
                f"[{idx+1}/{total}] 执行 {meta.name}..."
            )
            logger.info(f"    执行: [{meta.category.value}] {meta.name}...")

            # 获取参数覆盖
            params = algo_config.param_overrides.get(algo_id, {})
            instance = algo_registry.create_instance(algo_id, **params)

            start = time.perf_counter()
            denoised = instance.denoise(signal_data.data, signal_data.sample_rate)
            elapsed = time.perf_counter() - start

            if denoised.shape != signal_data.data.shape:
                denoised = denoised.reshape(signal_data.data.shape)

            self.context.denoised_results[algo_id] = DenoisedResult(
                algorithm_id=algo_id,
                algorithm_name=meta.name,
                category=meta.category,
                denoised_signal=denoised,
                execution_time=elapsed,
                params=params,
            )
            self.context.algorithm_metas[algo_id] = {
                "name": meta.name,
                "category": meta.category,
                "execution_time": elapsed,
            }

            logger.info(f"      完成: {elapsed:.3f}s")

        except Exception as e:
            logger.warning(f"    算法执行失败 [{algo_id}]: {e}")
            # 使用原始信号作为降级
            self.context.denoised_results[algo_id] = DenoisedResult(
                algorithm_id=algo_id,
                algorithm_name=algo_id,
                category=AlgorithmCategory.CLASSICAL_FILTER,
                denoised_signal=signal_data.data.copy(),
                execution_time=0,
                success=False,
                error_message=str(e),
            )

    def _denoise_one(self, algo_id, instance, params, data, sample_rate) -> DenoisedResult:
        """执行单个算法降噪并返回结果（供并行调用）。"""
        try:
            start = time.perf_counter()
            denoised = instance.denoise(data, sample_rate)
            elapsed = time.perf_counter() - start

            if denoised.shape != data.shape:
                denoised = denoised.reshape(data.shape)

            return DenoisedResult(
                algorithm_id=algo_id,
                algorithm_name=getattr(instance, 'name', algo_id),
                category=getattr(instance, 'category', AlgorithmCategory.CLASSICAL_FILTER),
                denoised_signal=denoised,
                execution_time=elapsed,
                params=params,
            )
        except Exception as e:
            logger.warning(f"    降噪失败 [{algo_id}]: {e}")
            return DenoisedResult(
                algorithm_id=algo_id,
                algorithm_name=algo_id,
                category=AlgorithmCategory.CLASSICAL_FILTER,
                denoised_signal=data.copy(),
                execution_time=0,
                success=False,
                error_message=str(e),
            )

    def _select_algorithms(self, algo_config) -> List[str]:
        """选择要执行的算法"""
        mode = algo_config.mode

        if mode == "selected":
            return algo_config.selected_ids
        elif mode == "category":
            all_algos = algo_registry.filter(
                category=algo_config.categories if algo_config.categories else None
            )
            return [m.algorithm_id for m in all_algos]
        else:  # "all"
            return algo_registry.list_ids()

    # ---- Stage 4: 包络谱分析 ----

    def _stage_envelope(self):
        """Stage 4: 包络谱分析"""
        logger.info("[Stage 4/7] 包络谱分析...")
        envelope_config = self.config.envelope
        analyzer = EnvelopeAnalyzer(envelope_config.model_dump())

        sample_rate = self.context.preprocessed_signal.sample_rate
        for algo_id, result in self.context.denoised_results.items():
            if not result.success:
                continue
            try:
                env_result = analyzer.analyze(result.denoised_signal, sample_rate)
                env_result.algorithm_id = algo_id
                self.context.envelope_results[algo_id] = env_result
            except Exception as e:
                logger.warning(f"  包络谱分析失败 [{algo_id}]: {e}")

        logger.info(f"  包络谱分析完成: {len(self.context.envelope_results)} 个")

    # ---- Stage 5: 指标计算 ----

    def _stage_metrics(self):
        """Stage 5: 指标计算"""
        logger.info("[Stage 5/7] 指标计算...")
        enabled_metrics = self.config.metrics.enabled
        sample_rate = self.context.preprocessed_signal.sample_rate

        # 确定更高越好方向
        higher_is_better = {}
        for mname in enabled_metrics:
            try:
                metric = metric_registry.get(mname)
                higher_is_better[mname] = metric.higher_is_better
            except KeyError:
                higher_is_better[mname] = True

        for algo_id in self.context.denoised_results:
            result = self.context.denoised_results[algo_id]
            if not result.success:
                self.context.metric_scores[algo_id] = {m: 0.0 for m in enabled_metrics}
                continue

            envelope = self.context.envelope_results.get(algo_id)
            scores = metric_registry.compute_all(
                denoised=result.denoised_signal,
                sample_rate=sample_rate,
                enabled=enabled_metrics,
                envelope=envelope,
                fault_freq_config=self.config.envelope.fault_frequencies,
            )
            self.context.metric_scores[algo_id] = scores

        logger.info(f"  指标计算完成: {len(self.context.metric_scores)} 个算法 × {len(enabled_metrics)} 个指标")

    # ---- Stage 6: 评分 ----

    def _stage_scoring(self):
        """Stage 6: 综合评分"""
        logger.info("[Stage 6/7] 综合评分...")
        enabled_metrics = self.config.metrics.enabled

        # 权重配置
        weights = dict(self.config.metrics.weights or DEFAULT_METRIC_WEIGHTS)

        # 自动权重分配
        auto_weights = {}
        manual_keys = set(weights.keys())
        auto_count = 0
        for mname in enabled_metrics:
            if mname not in manual_keys:
                auto_count += 1
        if auto_count > 0:
            remaining = 1.0 - sum(weights.values())
            per_metric = max(remaining / max(auto_count, 1), 0)
            for mname in enabled_metrics:
                if mname not in manual_keys:
                    auto_weights[mname] = per_metric

        final_weights = {**auto_weights, **weights}

        # 更高越好方向
        higher_is_better = {}
        for mname in enabled_metrics:
            try:
                higher_is_better[mname] = metric_registry.get(mname).higher_is_better
            except KeyError:
                higher_is_better[mname] = True

        ranking_engine = RankingEngine({
            "method": self.config.scoring.method,
            "normalization": self.config.scoring.normalization,
        })

        self.context._higher_is_better = higher_is_better
        self.context._weights = final_weights
        self.context._ranking_engine = ranking_engine

    # ---- Stage 7: 排序输出 ----

    def _stage_ranking(self):
        """Stage 7: 排序并生成报告"""
        logger.info("[Stage 7/7] 排序...")
        higher_is_better = getattr(self.context, '_higher_is_better', {})
        weights = getattr(self.context, '_weights', DEFAULT_METRIC_WEIGHTS)
        ranking_engine = getattr(self.context, '_ranking_engine', RankingEngine({}))

        rankings = ranking_engine.rank(
            metric_scores=self.context.metric_scores,
            algorithm_metas=self.context.algorithm_metas,
            weights=weights,
            higher_is_better=higher_is_better,
            top_n=self.config.scoring.top_n,
        )

        # 构建报告
        signal_info = {
            "path": self.context.original_signal.source_path or "simulated",
            "sample_rate": self.context.original_signal.sample_rate,
            "channels": self.context.original_signal.n_channels,
            "duration": self.context.original_signal.duration,
        }

        # 输出TOP-3的详细信息
        top_results = self.context.denoised_results
        top_envelopes = self.context.envelope_results
        top_scores = self.context.metric_scores

        self.context.report = EvaluationReport(
            project_name=self.config.project.name,
            signal_info=signal_info,
            rankings=rankings,
            top_n=self.config.scoring.top_n,
            # 携带 TOP-N 降噪信号供 GUI 对比展示
            top_denoised_results={
                aid: self.context.denoised_results[aid]
                for aid in [r.algorithm_id for r in rankings[:self.config.scoring.top_n]]
                if aid in self.context.denoised_results
            },
        )
