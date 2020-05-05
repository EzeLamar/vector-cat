## Installation

Install Python 3.6 64-bit version (or higher).

```
python -m pip install -U pip
python -m pip install tensorflow-gpu
python -m pip install keras
python -m pip install opencv-python
```

## Execution

In the solutions directory:

1. Place _'training.csv'_ in the _'data'_ directory.

2. Execute ```python/train.py data/training.csv data/model.h5```

3. Execute ```python/test.py data/model.h5 cascades/haarcascade_frontalface_default.xml```

4. Execute ```python/keras2tflite.py data/model.h5 data/model.tflite```
