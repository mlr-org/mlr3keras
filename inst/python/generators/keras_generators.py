import os
import warnings
import numpy as np

from keras.preprocessing.image import Iterator, ImageDataGenerator
from keras.utils import Sequence


class Numpy2DArrayIterator(Iterator):
    """Iterator yielding data from a Numpy array.
    # Arguments
        x: Numpy array of input data or tuple.
            If tuple, the second elements is either
            another numpy array or a list of numpy arrays,
            each of which gets passed
            through as an output without any modifications.
        y: Numpy array of targets data.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        sample_weight: Numpy array of sample weights.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
        subset: Subset of data (`"training"` or `"validation"`) if
            validation_split is set in ImageDataGenerator.
        ignore_class_split: Boolean (default: False), ignore difference
                in number of classes in labels across train and validation
                split (useful for non-classification tasks)
        dtype: Dtype to use for the generated arrays.
    """

    def __new__(cls, *args, **kwargs):
        try:
            from tensorflow.keras.utils import Sequence as TFSequence
            if TFSequence not in cls.__bases__:
                cls.__bases__ = cls.__bases__ + (TFSequence,)
        except ImportError:
            pass
        return super(Numpy2DArrayIterator, cls).__new__(cls)

    def __init__(self,
                 x,
                 y,
                 image_data_generator,
                 batch_size=32,
                 shuffle=False,
                 sample_weight=None,
                 seed=None,
                 subset=None,
                 ignore_class_split=False,
                 dtype='float32'):
        self.dtype = dtype

        if (type(x) is tuple) or (type(x) is list):
            if type(x[1]) is not list:
                x_misc = [np.asarray(x[1])]
            else:
                x_misc = [np.asarray(xx) for xx in x[1]]
            x = x[0]
            for xx in x_misc:
                if len(x) != len(xx):
                    raise ValueError(
                        'All of the arrays in `x` '
                        'should have the same length. '
                        'Found a pair with: len(x[0]) = %s, len(x[?]) = %s' %
                        (len(x), len(xx)))
        else:
            x_misc = []

        if y is not None and len(x) != len(y):
            raise ValueError('`x` (feature matrix) and `y` (labels) '
                             'should have the same length. '
                             'Found: x.shape = %s, y.shape = %s' %
                             (np.asarray(x).shape, np.asarray(y).shape))
        if sample_weight is not None and len(x) != len(sample_weight):
            raise ValueError('`x` (feature matrix) and `sample_weight` '
                             'should have the same length. '
                             'Found: x.shape = %s, sample_weight.shape = %s' %
                             (np.asarray(x).shape, np.asarray(sample_weight).shape))
        if subset is not None:
            if subset not in {'training', 'validation'}:
                raise ValueError('Invalid subset name:', subset,
                                 '; expected "training" or "validation".')
            split_idx = int(len(x) * image_data_generator._validation_split)

            if (y is not None and not ignore_class_split and not
                np.array_equal(np.unique(y[:split_idx]),
                               np.unique(y[split_idx:]))):
                raise ValueError('Training and validation subsets '
                                 'have different number of classes after '
                                 'the split. If your numpy arrays are '
                                 'sorted by the label, you might want '
                                 'to shuffle them.')

            if subset == 'validation':
                x = x[:split_idx]
                x_misc = [np.asarray(xx[:split_idx]) for xx in x_misc]
                if y is not None:
                    y = y[:split_idx]
            else:
                x = x[split_idx:]
                x_misc = [np.asarray(xx[split_idx:]) for xx in x_misc]
                if y is not None:
                    y = y[split_idx:]

        self.x = np.asarray(x, dtype=self.dtype)
        self.x_misc = x_misc
        if self.x.ndim != 2:
            raise ValueError('Input data in `NumpyArrayIterator` '
                             'should have rank 2. You passed an array '
                             'with shape', self.x.shape)
        if y is not None:
            self.y = np.asarray(y)
        else:
            self.y = None
        if sample_weight is not None:
            self.sample_weight = np.asarray(sample_weight)
        else:
            self.sample_weight = None
        self.image_data_generator = image_data_generator
        super(Numpy2DArrayIterator, self).__init__(x.shape[0],
                                                 batch_size,
                                                 shuffle,
                                                 seed)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = self.x[index_array]

        batch_x_miscs = [xx[index_array] for xx in self.x_misc]
        output = (batch_x if batch_x_miscs == []
                  else [batch_x] + batch_x_miscs,)
        if self.y is None:
            return output[0]
        output += (self.y[index_array],)
        if self.sample_weight is not None:
            output += (self.sample_weight[index_array],)
        return output




class CombinedGenerator(Sequence):
    """Wraps 2 DataGenerators"""

    seed=None,
    batch_size=None,

    def __init__(self, gen1, gen2):

        # Real time multiple input data augmentation
        assert gen1.batch_size == gen2.batch_size
        self.batch_size = gen1.batch_size

        if gen1.seed != gen2.seed:
            Warning("Generator seeds do not match!")
        self.seed = gen1.seed

        self.gen1 = gen1
        self.gen2 = gen2

    def __len__(self):
        """It is mandatory to implement it on Keras Sequence"""
        return self.gen1.__len__()

    def __getitem__(self, index):
        """Getting items from the 2 generators and packing them, dropping first target"""
        X1_batch, Y_batch = self.gen1.__getitem__(index)
        X2_batch, Y2_batch = self.gen2.__getitem__(index)
        X_batch = [X1_batch, X2_batch]
        return X_batch, Y_batch
