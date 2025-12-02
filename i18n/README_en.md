# Deep Learning Tutorial

Pure-Python teaching project covering perceptron to Transformer. No external DL frameworks; focuses on core math and readable code.

## Highlights
- Pure Python implementations for transparent math
- Modular package `deep_learning/`
- Small examples and loaders with matching tests
- Visualization and training utilities (EarlyStopping, gradient accumulation)

## Environment & Installation
- Python >= 3.7
- NumPy >= 1.19.0 (core)
- Matplotlib >= 3.3.0 (optional, visualization)

```bash
pip install -r requirements.txt            # runtime
pip install -r requirements-dev.txt        # dev/test
```

## Usage
### 1) CLI menu
```bash
python main.py          # interactive menu
python main.py --list   # list modules
python main.py --help   # help
```

### 2) Import (recommended)
```python
from deep_learning.fundamentals import Perceptron

perceptron = Perceptron(input_size=2)
X = [[0,0],[0,1],[1,0],[1,1]]
y = [0,0,0,1]
for _ in range(100):
    perceptron.train(X, y)
print([perceptron.predict(xi) for xi in X])
```

### 3) Run examples
```bash
python -m examples.01_perceptron_and_gate
python -m examples.03_cnn_edge_detection
```

### 4) Run tests
```bash
pytest -q
make test
```

## Project Layout (simplified)
```
deep_learning/              # main package
├─ fundamentals/            # Perceptron, DeepNetwork, MLP
├─ architectures/           # CNN, RNN, Transformer demos
├─ optimizers/              # SGD/Adam/schedulers, advanced optim
├─ advanced/                # GAN/Transformer demos, projects
└─ utils/                   # activations/losses/inits/math/perf/vis
examples/                   # 01-05 scripts
datasets/                   # mnist_sample.npz, text_sequences.txt, data_loader.py
exercises/                  # exercises + reference answers
tests/                      # coverage for modules/examples/data/perf
docs/ARCHITECTURE.md        # package structure
README.md, TODO.md, CHANGELOG.md
```

Note: legacy `deep_learning_*.py` files are compatibility shims; prefer `deep_learning/` imports.

## Learning Path (suggested)
1. Basics: `deep_learning.fundamentals` (Perceptron/MLP)
2. Architectures: `deep_learning.architectures` (CNN/RNN/Transformer demos)
3. Optimization: `deep_learning.optimizers` (SGD/Adam/schedulers)
4. Advanced: `deep_learning.advanced` (GAN / project demos)
5. Practice: `examples/`, `exercises/`, `datasets/`

See also `DEEP_LEARNING_GUIDE.md` and `docs/ARCHITECTURE.md`.

## Contributing
- Run `make format && make lint && make test` before PR/Issue
- Keep docs aligned with code; use new package imports first

## License
MIT (see LICENSE).
