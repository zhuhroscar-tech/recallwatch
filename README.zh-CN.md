[![English](https://img.shields.io/badge/English-555555?style=flat)](README.md) [![简体中文](https://img.shields.io/badge/简体中文-555555?style=flat)](README.zh-CN.md)

# recallwatch

按查询密度分组评估 ANN（近似最近邻）召回率，再比较两次快照，找出召回下降的分组。除了整体均值，`recallwatch` 还分别报告 **head / torso / tail**，避免表现较好的查询掩盖薄弱区域。

CLI 内置 signed random-projection LSH 索引，以精确余弦检索作为参考。它是本地评估工具，不是托管监控服务，也不自带生产向量数据库的连接器。

## 安装并试用

需要 Python 3.9+ 和 NumPy 1.23+；pip 会安装 NumPy。

```bash
git clone https://github.com/zhuhroscar-tech/recallwatch.git
cd recallwatch
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
recallwatch snapshot --n-items 3000 --n-queries 200 --dim 24 --seed 0 --out baseline.json
recallwatch snapshot --n-items 3000 --n-queries 200 --dim 24 --seed 0 --n-bits 20 --out current.json
recallwatch diff baseline.json current.json --threshold 0.05
```

示例使用可复现的合成向量，并改变 LSH 的分桶粒度。hash bits 越多，桶越细；实际召回率取决于数据和索引参数。`snapshot` 会写入指定的 JSON 文件，路径已有文件时会覆盖。

## 使用自己的数据

```bash
recallwatch snapshot --corpus corpus.npy --queries queries.npy --out report.json
recallwatch diff baseline.json report.json --json
```

输入应为二维数值 `.npy` 数组，形状分别为 `(n_items, dim)` 和 `(n_queries, dim)`，两者维度必须一致。`--k`、`--density-k`、`--n-tables` 等选项可通过 `recallwatch snapshot --help` 查看。

某一分组的召回率下降**超过**绝对阈值时，`diff` 返回 `1`，否则返回 `0`。召回提升不算回归。分组缺失或值为 NaN 时会提示数据不足，但不会触发回归退出码；因此不能仅凭 `0` 就认定检查完整通过。

## 限制与集成

分组依据当前查询批次内的距离分位数，不是跨批次固定的类别。比较漂移时，应保持数据集和参数具有可比性。精确检索的开销较大，大规模生产性能尚未验证。

如需评估其他 ANN 索引，可向 [core.py](src/recallwatch/core.py) 中的 `measure_segmented_recall()` 传入实现了 `search(query, k) -> ids` 的对象。返回 ID 必须对应 corpus 的行号。内置 LSH 用于展示和验证方法，不以替代 FAISS、HNSWlib 的性能为目标。

## 开发

```bash
python -m pip install -e ".[dev]"
python -m pytest -v
```

[测试](tests/test_core.py) · [CI](.github/workflows/ci.yml) · [发布历史](CHANGELOG.md) · [MIT 许可证](LICENSE)
