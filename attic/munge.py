import numpy as np

def findNN(elementIndex, array):
    """
    elementIndex: the element that we are searching the nearest neighbor of
    array: array

    Returns
    -------
    nearestNeighborIndex: the index of the nearestNeighbor
    """
    element = array[elementIndex]
    array = np.delete( array, elementIndex, axis=0)
    nearestNeighborIndex = np.argmin(np.linalg.norm(element - array, axis=1))

    if nearestNeighborIndex >= elementIndex: # adjust
        nearestNeighborIndex += 1
    return nearestNeighborIndex



def munge(dataset, sizeMultiplier, swapProb, varParam):
    """
    This algorithm was presented in the following paper:

    C. Bucilua, R. Caruana, and A. Niculescu-Mizil. Model compression. In Proceedings of the
    12th ACMSIGKDD International Conference on Knowledge Discovery and Data Mining, KDD
    ’06, pages 535–541, New York, NY, USA, 2006. ACM.

    Creates a synthetic dataset.
    Continuous attributes should be linearly scaled to [0, 1].

    dataset: 2D numpy array (numExamples, numAttributes)
    sizeMultiplier: dataset size multiplier
    swapProb: probability of swapping attributes (draw from normal with mean)
    varParam: local variance parameter

    Returns
    -------
    synthetic: (sizeMultiplier*numExamples, numAttributes)
    """

    numExamples, numAttributes = dataset.shape
    synthetic = np.empty((sizeMultiplier*numExamples, numAttributes))

    is_continuous = np.empty(numAttributes)
    for i in range(numAttributes):
        is_continuous[i] = len(set(dataset[, i])) > 2

    for i in range(sizeMultiplier):
        tempDataset = np.copy(dataset)

        for exampleIndex in range(numExamples):
            nearestNeighborIndex = findNN(exampleIndex, tempDataset)

            for j in range(numAttributes):
                if np.random.uniform() < swapProb:
                    example_attr = tempDataset[ exampleIndex, j]
                    closestNeighbor_attr = tempDataset[ nearestNeighborIndex, j]
                    if is_continuous[j]:
                        # Do munging
                        tempDataset[ exampleIndex, j] = np.random.normal( closestNeighbor_attr, abs( example_attr - closestNeighbor_attr) / varParam)
                        tempDataset[ nearestNeighborIndex, j] = np.random.normal( example_attr, abs( example_attr - closestNeighbor_attr) / varParam)
                    else:
                        # Simply switch
                        tempDataset[ exampleIndex, j] = closestNeighbor_attr
                        tempDataset[ nearestNeighborIndex, j] = example_attr

        synthetic[i*numExamples:(i+1)*numExamples, :] = tempDataset
    return synthetic


class MungeIterator(Numpy2DArrayIterator):

    def __new__(cls, *args, **kwargs):
        return super(MungeIterator, cls).__new__(cls)

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
        dtype='float32',
        swapProb=0.5,
        varParam=2.,
        sizeMultiplier=2):

        x = munge(x, sizeMultiplier, swapProb, varParam)
        super(Numpy2DArrayIterator, self).__init__(
            x=x,
            y=y,
            image_data_generator=image_data_generator,
            batch_size=batch_size,
            shuffle=shuffle,
            sample_weight=sample_weight,
            seed=seed,
            subset=subset,
            ignore_class_split=ignore_class_split,
            dtype=dtype)


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

