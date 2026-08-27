# Build a wheel file
```bash
cd triton
pip install build
python -m build --wheel
```

# From setup.py
pip install --upgrade setuptools wheel
python setup.py bdist_wheel