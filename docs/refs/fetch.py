"""把两个参考仓库里最关键的源码文件拉下来，方便离线精读。

只拉"决定架构"的文件：回测引擎、数据加载、指标、策略定义、Agent 侧的工具封装。
不拉测试、不拉前端、不拉 wiki。
"""
import urllib.request

VT = 'https://raw.githubusercontent.com/HKUDS/Vibe-Trading/main/'
VR = 'https://raw.githubusercontent.com/simonlin1212/Vibe-Research/main/'

JOBS = [
    # ---------- Vibe-Trading ----------
    (VT, 'README_zh.md',                                  'vt/README_zh.md'),
    (VT, 'agent/backtest/models.py',                      'vt/bt_models.py'),
    (VT, 'agent/backtest/engines/base.py',                'vt/bt_engine_base.py'),
    (VT, 'agent/backtest/engines/china_a.py',             'vt/bt_engine_china_a.py'),
    (VT, 'agent/backtest/metrics.py',                     'vt/bt_metrics.py'),
    (VT, 'agent/backtest/constraints.py',                 'vt/bt_constraints.py'),
    (VT, 'agent/backtest/runner.py',                      'vt/bt_runner.py'),
    (VT, 'agent/backtest/loaders/registry.py',            'vt/bt_loader_registry.py'),
    (VT, 'agent/backtest/loaders/base.py',                'vt/bt_loader_base.py'),
    (VT, 'agent/src/tools/backtest_tool.py',              'vt/tool_backtest.py'),
    (VT, 'agent/src/strategy_store/models.py',            'vt/strategy_models.py'),
    (VT, 'agent/src/strategy_store/store.py',             'vt/strategy_store.py'),
    (VT, 'agent/src/factors/base.py',                     'vt/factors_base.py'),
    (VT, 'agent/requirements.txt',                        'vt/requirements.txt'),
    # ---------- Vibe-Research ----------
    (VR, 'README.md',                                     'vr/README.md'),
    (VR, 'AGENTS.md',                                     'vr/AGENTS.md'),
    (VR, 'backtest/README.md',                            'vr/bt_README.md'),
    (VR, 'backtest/strategies.py',                        'vr/bt_strategies.py'),
    (VR, 'backtest/models.py',                            'vr/bt_models.py'),
    (VR, 'backtest/engines/base.py',                      'vr/bt_engine_base.py'),
    (VR, 'backtest/engines/china_a.py',                   'vr/bt_engine_china_a.py'),
    (VR, 'backtest/metrics.py',                           'vr/bt_metrics.py'),
    (VR, 'backtest/run.py',                               'vr/bt_run.py'),
    (VR, 'backtest/constraints.py',                       'vr/bt_constraints.py'),
    (VR, 'backtest/loader.py',                            'vr/bt_loader.py'),
    (VR, 'calc/indicators.py',                            'vr/calc_indicators.py'),
]

for base, path, out in JOBS:
    try:
        with urllib.request.urlopen(base + path, timeout=45) as r:
            data = r.read()
        with open(out, 'wb') as f:
            f.write(data)
        print('%-30s %8d B  %5d 行' % (out, len(data), data.count(b'\n')))
    except Exception as exc:                                  # noqa: BLE001
        print('%-30s 失败: %s' % (out, exc))
