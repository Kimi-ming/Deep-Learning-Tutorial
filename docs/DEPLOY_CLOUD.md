# Cloud Deployment Guide (AWS/GCP/Azure)

目标：以最小依赖运行示例/测试。无需 GPU。

## 通用准备
- Python 3.8+，且可安装系统构建工具（gcc/clang）。
- `git clone https://github.com/Kimi-ming/Deep-Learning-Tutorial.git`
- 安装依赖：`pip install -r requirements.txt`（或 `pip install .`）
- 可选：`pip install -r requirements-dev.txt`（运行测试）

## AWS（EC2）
- 推荐 AMI：Ubuntu 22.04。
- 安装 Python3 与 pip：`sudo apt update && sudo apt install -y python3-pip python3-venv`
- (可选) 虚拟环境：`python3 -m venv .venv && source .venv/bin/activate`
- 运行：
  - `pip install -r requirements.txt`
  - `python main.py --list` / `dl-tutorial`
- 若需容器：安装 Docker，`docker build -t dl-tutorial . && docker run --rm dl-tutorial python main.py --help`

## GCP（Compute Engine）
- 选择 Debian/Ubuntu 镜像，SSH 登录后步骤同 AWS。
- 若用 Container-Optimized OS，直接 `docker run`：
  - `docker build -t dl-tutorial .`
  - `docker run --rm dl-tutorial python main.py --list`

## Azure（VM）
- 选择 Ubuntu LTS，安装 Python3/pip 或 Docker，命令同上。

## 测试与示例
- 全部测试：`pytest -q`（如安装 dev 依赖）
- 单测：`pytest tests/utils/test_numpy_backend.py`
- 示例：`python -m examples.01_perceptron_and_gate`

## 常见问题
- **无 sudo / 端口限制**：使用 `pip install --user`，或在虚拟环境内安装。
- **网络拉取慢**：预先构建镜像并推送到私有仓库（ECR/GCR/ACR），在云端直接 `docker pull`。
