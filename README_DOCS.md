# 文档构建指南

## 快速开始
```bash
pip install -r requirements-dev.txt
make docs  # 生成到 docs/_build
```

## 结构
- `docs/conf.py`: Sphinx 配置
- `docs/index.rst`: 入口 + 目录
- `docs/api.rst`: API 自动文档（deep_learning 包）

## 查看输出
`open docs/_build/index.html` 或用任意静态服务器浏览。
