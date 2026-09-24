# Contributing to Laya-as-a-Judge

Thank you for your interest in contributing to **Laya-as-a-Judge**!

We welcome contributions of all kinds: bug fixes, new judge rubrics, hardware backend improvements (MLX, PyTorch, ONNX, TensorRT), benchmark fixtures, and documentation enhancements.

---

## Development Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/rbrus/laya-as-judge.git
   cd laya-as-judge
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies in editable mode:**
   ```bash
   # Core development with test suite
   pip install -e '.[dev]'

   # If developing on Apple Silicon with MLX:
   pip install -e '.[dev,mlx]'

   # If working on the (currently incomplete) PyTorch backend:
   pip install -e '.[dev,torch]'
   ```

---

## Running Tests

Run the test suite with `pytest`:

```bash
pytest -v
```

The test suite runs in about a second on the heuristic `EmulatorBackend`, so it needs no GPU and no weight downloads. Note that this means the tests check the API, schemas and plumbing, not the quality of real Laya model verdicts.

---

## Adding New Preset Judges

To add a new preset evaluation rubric:

1. Create a new module in `laya_as_judge/judges/`.
2. Inherit from `BaseJudge`.
3. Define the `@property def questions(self)` returning a dictionary of typed question specifications (`noul`, `score`, or `choice`).
4. Implement task-specific helper methods (e.g. `evaluate_*`).
5. Export your judge in `laya_as_judge/judges/__init__.py` and `laya_as_judge/__init__.py`.
6. Add corresponding unit tests in `tests/test_judges.py`.

---

## Code Quality Standards

- Maintain strict typing with Python type annotations.
- Adhere to PEP 8 standards.
- Keep output token count at **0**—Laya decision models do not generate text tokens!
- Ensure all tests pass before opening a Pull Request.

---

## License

By contributing to Laya-as-a-Judge, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
