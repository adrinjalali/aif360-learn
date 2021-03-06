from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from aiflearn.algorithms.preprocessing.reweighing import Reweighing
from aiflearn.datasets import AdultDataset

def test_instance_weights():
    ad = AdultDataset(instance_weights_name='fnlwgt', features_to_drop=[])
    privileged_groups = [{'sex': 1}]
    unprivileged_groups = [{'sex': 0}]
    rw = Reweighing(unprivileged_groups=unprivileged_groups,
                    privileged_groups=privileged_groups)
    transf = rw.fit_transform(ad)
    print(transf.instance_weights.sum())
    assert np.isclose(ad.instance_weights.sum(), transf.instance_weights.sum())
