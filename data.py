import numpy as np

def triplet_generator():
    ''' Dummy triplet generator for API usage demo only.

    Will be replaced by a version that uses real image data later.

    :return: a batch of (anchor, positive, negative) triplets
    '''
    while True:
        a_batch = np.random.rand(4, 3, 96, 96)
        p_batch = np.random.rand(4, 3, 96, 96)
        n_batch = np.random.rand(4, 3, 96, 96)
        yield [a_batch , p_batch, n_batch], None
