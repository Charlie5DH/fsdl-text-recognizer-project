# Notes

## Importing modules in python

`%load_ext autoreload` is an IPython extension to reload modules before executing user `code. autoreload` reloads modules automatically before entering the execution of code typed at the IPython prompt. This enables make changes in modules and see the changes in the notebook.

In this case we are executing code from a folder named Notebooks and we want to import modules from a folder that contains many python modules named text_recognizer.

```python
%load_ext autoreload
%autoreload 2

## Managing Modules
from importlib.util import find_spec
if find_spec("text_recognizer") is None:
    import sys
    sys.path.append('..')

from text_recognizer.datasets.emnist_dataset import EmnistDataset
```

The lower-level API in `importlib.util` provides access to the loader objects, as described in Modules and Imports from the section on the sys module. To get the information needed to load the module for a module, use `find_spec()` to find the “import spec”. Then to retrieve the module, use the loader’s `load_module()` method. If a spec cannot be found, None is returned.

A `spec` is a namespace containing the import-related information used to load a module. An instance of `importlib.machinery.ModuleSpec.`. We first look if there is a spec loaded

If a spec is founded the returns the following because it founded an `__init__.py` and the submodules. This is because the folder is added to the path using `sys.path.append`.

```python
find_spec("text_recognizer")

>>> ModuleSpec(name='text_recognizer', 
loader=<_frozen_importlib_external.SourceFileLoader object at 0x0000020CAF2AFA48>, 
origin='..\\\\text_recognizer\\\\__init__.py', 
submodule_search_locations=['..\\\\text_recognizer'])
```

## Networks models 

The files inside models folder contain only the architecture of the models, one is an MLP (Multilayer Perceptron) and the other one is a LENET architecture. The python files are functions to create the models.

```python
"""Define mlp network function."""
from typing import Tuple

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten


def mlp(
    input_shape: Tuple[int, ...],
    output_shape: Tuple[int, ...],
    layer_size: int = 128,
    dropout_amount: float = 0.2,
    num_layers: int = 3,
) -> Model:
    """
    Create a simple multi-layer perceptron: fully-connected layers with dropout between them, with softmax predictions.
    Creates num_layers layers.
    Creates a sequential model. Flatten the 28x28 input first
    and then we add layers and dropouts per number of layers specified
    """
    num_classes = output_shape[0]

    model = Sequential()
    # Don't forget to pass input_shape to the first layer of the model
    # Your code below (Lab 1)
    model.add(Flatten(input_shape=input_shape))
    for _ in range(num_layers):
        model.add(Dense(layer_size, activation="relu"))
        model.add(Dropout(dropout_amount))
    model.add(Dense(num_classes, activation="softmax"))
    # Your code above (Lab 1)

    return model

```

```python
"""LeNet network."""
from typing import Tuple

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda, MaxPooling2D
from tensorflow.keras.models import Sequential, Model


def lenet(input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> Model:
    """Return LeNet Keras model."""
    num_classes = output_shape[0]

    # Your code below (Lab 2)
    model = Sequential()
    if len(input_shape) < 3:
        model.add(Lambda(lambda x: tf.expand_dims(x, -1), input_shape=input_shape, name='expand_dims'))
        input_shape = (input_shape[0], input_shape[1], 1)
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape, padding="valid"))
    model.add(Conv2D(64, (3, 3), activation="relu", padding="valid"))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="valid"))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation="softmax"))
    # Your code above (Lab 2)

    return model
```

## Class Model

In the model files (MLP and LENET) there is no definition of how the model interact with data or the loss. This will be specified in the Model Class.

```python
DIRNAME = Path(__file__).parents[1].resolve() / "weights"

class Model:
    """Base class, to be subclassed by predictors for specific type of data."""

    def __init__(
        self,
        dataset_cls: type,
        network_fn: Callable[..., KerasModel],
        dataset_args: Dict = None,
        network_args: Dict = None,
    ):
        self.name = f"{self.__class__.__name__}_{dataset_cls.__name__}_{network_fn.__name__}"

        if dataset_args is None:
            dataset_args = {}
        self.data = dataset_cls(**dataset_args)

        if network_args is None:
            network_args = {}
        self.network = network_fn(self.data.input_shape, self.data.output_shape, **network_args)
        self.network.summary()

        self.batch_augment_fn: Optional[Callable] = None
        self.batch_format_fn: Optional[Callable] = None

    @property
    def image_shape(self):
        return self.data.input_shape

    @property
    def weights_filename(self) -> str:
        DIRNAME.mkdir(parents=True, exist_ok=True)
        return str(DIRNAME / f"{self.name}_weights.h5")

    def fit(
        self, dataset, batch_size: int = 32, epochs: int = 10, augment_val: bool = True, callbacks: list = None,
    ):
        if callbacks is None:
            callbacks = []

        self.network.compile(loss=self.loss(), optimizer=self.optimizer(), metrics=self.metrics())

        train_sequence = DatasetSequence(
            dataset.x_train,
            dataset.y_train,
            batch_size,
            augment_fn=self.batch_augment_fn,
            format_fn=self.batch_format_fn,
        )
        test_sequence = DatasetSequence(
            dataset.x_test,
            dataset.y_test,
            batch_size,
            augment_fn=self.batch_augment_fn if augment_val else None,
            format_fn=self.batch_format_fn,
        )

        self.network.fit(
            train_sequence,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=test_sequence,
            use_multiprocessing=False,
            workers=1,
            shuffle=True,
        )

    def evaluate(self, x: np.ndarray, y: np.ndarray, batch_size: int = 16, _verbose: bool = False):
        # pylint: disable=unused-argument
        sequence = DatasetSequence(x, y, batch_size=batch_size)  # Use a small batch size to use less memory
        preds = self.network.predict(sequence)
        return np.mean(np.argmax(preds, -1) == np.argmax(y, -1))

    def loss(self):  # pylint: disable=no-self-use
        return "categorical_crossentropy"

    def optimizer(self):  # pylint: disable=no-self-use
        return RMSprop()

    def metrics(self):  # pylint: disable=no-self-use
        return ["accuracy"]

    def load_weights(self):
        self.network.load_weights(self.weights_filename)

    def save_weights(self):
        self.network.save_weights(self.weights_filename)
```

