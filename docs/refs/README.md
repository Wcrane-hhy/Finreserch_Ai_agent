# docs/refs —— 参考项目分析依据

这个目录是**分析依据**，不是产品代码。里面的东西是"我怎么得出那些结论"的证据，
可以随时删掉，删了不影响任何功能。

## 两个参考仓库

| 目录 | 仓库 | 文件数 | 说明 |
|---|---|---|---|
| `vt_tree.json` | [HKUDS/Vibe-Trading](https://github.com/HKUDS/Vibe-Trading) | 2612 | 官方文件树（Git Trees API） |
| `vr_tree.json` | [simonlin1212/Vibe-Research](https://github.com/simonlin1212/Vibe-Research) | 514 | 同上 |
| `vt/` | 从 Vibe-Trading 拉的 14 个关键源文件 | — | 见下 |
| `vr/` | 从 Vibe-Research 拉的 12 个关键源文件 | — | 见下 |

两个仓库都是 **MIT 许可**。Vibe-Research 的 `backtest/` 移植自 Vibe-Trading，
有 `NOTICE.md` 写明移植基线与改动清单 —— 那份清单就是本项目的迁移蓝图。

## 怎么拉下来的（不用 clone 全库）

Vibe-Trading 有 2612 个文件，clone 一遍又大又慢。用两个接口就够了：

```bash
# 1. 完整文件树（一条请求拿到所有路径，几十 KB）
curl -s "https://api.github.com/repos/<owner>/<repo>/git/trees/main?recursive=1"

# 2. 只看源码原文（挑文件，不整库下）
curl -s "https://raw.githubusercontent.com/<owner>/<repo>/main/<path>"
```

`fetch.py` 就是第 2 步的脚本，改 `JOBS` 列表即可重拉：

```bash
"C:/Users/17667/AppData/Local/Programs/Python/Python312/python.exe" docs/refs/fetch.py
```

## 精读清单（决定架构的那几个文件）

**Vibe-Trading**

| 文件 | 为什么读它 |
|---|---|
| `bt_engine_base.py` | 逐 bar 撮合内核（2374 行）—— 回测的心脏 |
| `bt_engine_china_a.py` | A股规则在**上游**的样子 |
| `bt_models.py` | 持仓/成交/交易/权益的不可变 dataclass |
| `bt_metrics.py` | 全套指标 |
| `tool_backtest.py` | **Agent 怎么调回测**：代码生成 + 子进程沙箱 |
| `strategy_models.py` / `strategy_store.py` | 策略库持久化 + 衰减监控 |
| `bt_loader_registry.py` | 25 个数据源的注册方式 |
| `requirements.txt` | 依赖全貌（能看出产品重量） |

**Vibe-Research**

| 文件 | 为什么读它 |
|---|---|
| `bt_README.md` | 回测模块自述 —— 闸口/市场规则/守卫讲得最清楚 |
| `bt_engine_china_a.py` | **A股规则最终版**（T+1/涨跌停/整手/费用） |
| `bt_run.py` | 接线 + `_Guarded` 运行期守卫 + `Result.limits` |
| `bt_loader.py` | **引擎与数据之间那条缝**：`fetch()` 契约 |
| `bt_strategies.py` | 策略契约 + "策略长什么样"的模板 |
| `AGENTS.md` | 它的"金融研究宪法"—— 数据纪律与不可越线 |
| `calc_indicators.py` | 确定性指标计算库 |
| `README.md` | 13 个一级栏目 —— 个人投研工作台的功能全貌 |

## 结论在哪

`docs/项目二-架构分析与迁移计划.md` —— 分析结论、A股改造清单、目标架构、7 期迁移计划。
