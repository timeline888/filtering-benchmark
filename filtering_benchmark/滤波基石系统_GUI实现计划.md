# 滤波基石系统 — PyQt6 图形用户界面实现计划

## 背景

滤波基石系统（Filtering Benchmark）是一个旋转机械振动/声学信号滤波降噪评估系统，包含 71 种算法、20 项评估指标、完整的 7 阶段评估管线。当前仅提供 CLI 命令行界面，用户需要一种更直观的可视化操作方式。

## 目标

基于现有的 PipelineOrchestrator、AlgorithmRegistry 和 ConfigLoader，构建一个完整的 PyQt6 图形用户界面，提供信号输入、算法选择、参数配置、执行控制、结果展示和配置管理功能。

## 架构概览

```
src/gui/
├── __init__.py
├── app.py                    # QApplication 入口
├── main_window.py            # QMainWindow 主窗口
├── models/
│   ├── __init__.py
│   └── algorithm_tree_model.py  # QAbstractItemModel 树模型
├── widgets/
│   ├── __init__.py
│   ├── signal_input_widget.py   # 信号输入面板
│   ├── algorithm_selector.py    # 算法选择器
│   ├── execution_control.py     # 执行控制面板
│   ├── results_tab_widget.py    # 结果展示标签页
│   └── chart_canvas.py          # matplotlib 画布
├── workers/
│   ├── __init__.py
│   └── pipeline_worker.py       # QObject 异步工作线程
├── utils/
│   ├── __init__.py
│   └── config_bridge.py         # UI ↔ Config 双向转换
└── dialogs/
    ├── __init__.py
    └── config_editor_dialog.py  # 配置编辑器对话框
```

## 组件详情

### 1. app.py — 应用入口
- 创建 QApplication
- 设置应用图标、默认字体、样式表
- 创建 MainWindow 并 show()
- 捕获全局异常

### 2. main_window.py — 主窗口
- QMainWindow + QSplitter 左右分栏
- 左侧：QScrollArea 包含信号输入、算法选择、执行控制面板
- 右侧：QTabWidget 包含结果标签页
- 菜单栏：文件（打开配置、保存配置、退出）、视图、帮助
- 状态栏：当前状态信息

### 3. models/algorithm_tree_model.py — 算法树模型
- QAbstractItemModel，2 层树结构
- 第一层：算法类别（13 个 AlgorithmCategory）
- 第二层：具体算法（带复选框）
- 每个节点存储：ID、名称、类别、复杂度、GPU 需求、是否选中
- 支持搜索过滤

### 4. widgets/signal_input_widget.py — 信号输入
- QStackedWidget 切换文件模式 / 仿真模式
- 文件模式：QLineEdit + 浏览按钮，格式过滤（MAT/WAV/CSV/TXT/HDF5）
- 仿真模式：下拉选择类型（bearing_fault, gear_fault, chirp, multi_harmonic）
- 参数区：采样率（默认 25600）、通道选择
- 信号预览：嵌入 matplotlib 波形图

### 5. widgets/algorithm_selector.py — 算法选择
- QLineEdit 搜索框（按名称/标签实时过滤）
- QTreeView + AlgorithmTreeModel 显示算法树
- 全选/取消全选按钮
- 选中算法数量统计显示
- 右键菜单：查看算法详情

### 6. widgets/execution_control.py — 执行控制
- 运行/停止按钮
- QProgressBar + 阶段文本标签
- 算法选择模式：全部 / 选定 / 按类别
- TOP-N 设置（SpinBox）
- 评分方法选择（加权和 / TOPSIS / Borda Count）
- 并行后端选择（顺序 / 多进程）

### 7. workers/pipeline_worker.py — 异步工作线程
- QObject 类，moveToThread(QThread)
- 信号：progress_updated(int, str), stage_changed(str), finished(dict), error(str)
- 接收 AppConfig，执行 PipelineOrchestrator.run()
- 注入进度回调到管线各阶段
- 支持停止信号（检查中止标志）

### 8. widgets/results_tab_widget.py — 结果展示
- 4 个标签页：
  - 排名表：QTableView + 排序模型，显示 TOP-N 算法排名
  - 指标对比：QTableWidget 显示所有算法的各项指标值
  - 图表：matplotlib 画布（排名柱状图、雷达图、指标对比图）
  - 报告：QTextBrowser 显示评估报告文本摘要
- 导出按钮：CSV / HTML

### 9. utils/config_bridge.py — 配置桥接
- ui_to_config(): 从 UI 控件值构建 AppConfig 字典
- config_to_ui(): 从 AppConfig 填充 UI 控件
- 支持加载 YAML 文件到 UI
- 支持从 UI 保存到 YAML 文件

### 10. dialogs/config_editor_dialog.py — 配置编辑器
- QDialog 全配置编辑
- 按类别分组的表单布局
- 参数验证

## 数据流

```
用户操作 → UI 控件变化
       ↓
config_bridge.ui_to_config() → AppConfig dict
       ↓
PipelineWorker (QThread)
  ├─ stage_changed("加载信号")
  ├─ progress_updated(10, "加载完成")
  ├─ stage_changed("执行降噪")
  ├─ progress_updated(50, "30/71 算法完成")
  ├─ stage_changed("计算指标")
  ├─ progress_updated(90, "评分完成")
  └─ finished(EvaluationReport)
       ↓
results_tab_widget.load_report(report)
  ├─ 排名表更新
  ├─ 指标表更新
  ├─ 图表绘制
  └─ 报告生成
```

## 关键集成点

1. **PipelineOrchestrator** — 需要添加可选 `progress_callback` 参数
2. **AlgorithmRegistry** — 直接使用 `list_algorithms()` 和 `filter()` 填充算法树
3. **ConfigLoader** — 通过 config_bridge 双向转换
4. **SignalReaderFactory** — 文件格式自动检测
5. **Plotter** — 嵌入 matplotlib 画布到 GUI

## 实现顺序

1. 修改 PipelineOrchestrator 添加 progress_callback
2. 创建 gui/ 目录结构和 __init__.py
3. algorithm_tree_model.py
4. signal_input_widget.py
5. algorithm_selector.py
6. execution_control.py
7. pipeline_worker.py
8. config_bridge.py
9. config_editor_dialog.py
10. chart_canvas.py
11. results_tab_widget.py
12. main_window.py（组装所有组件）
13. app.py（入口）
14. 测试和调试

## 验证方式

1. `python -m src.gui.app` 启动 GUI
2. 验证算法树正确加载 71 个算法
3. 验证仿真信号生成并显示波形
4. 验证运行管线并显示结果
5. 验证 YAML 配置的加载和保存
