from logging import warning

import numpy as np
from sklearn.utils import check_array

from aiflearn.datasets import StructuredDataset
from aiflearn.metrics import Metric, utils


class DatasetMetric(Metric):
    """Class for computing metrics based on one StructuredDataset."""

    def __init__(self, X, col_names, sample_weights,
                 up_groups=None, p_groups=None):
        """
        Parameters
        ----------
        X : numpy.ndarray
            The input array
        col_names : list, 1d array
            Column names
        sample_weights: list, 1d array
            Sample weights
        p_groups : list(dict)
            Privileged groups. Format is a list
            of `dicts` where the keys are `protected_attribute_names` and
            the values are values in `protected_attributes`. Each `dict`
            element describes a single group. See examples for more details.
        p_groups : list(dict)
            Unprivileged groups in the same
            format as `privileged_groups`.

        Raises
        ------
        TypeError: `dataset` must be a
            :obj:`~aiflearn.datasets.StructuredDataset` type.
        ValueError: `privileged_groups` and `unprivileged_groups` must be
            disjoint.

        Examples
        --------
        >>> from aiflearn.datasets import GermanDataset
        >>> german = GermanDataset()
        >>> u = [{'sex': 1, 'age': 1}, {'sex': 0}]
        >>> p = [{'sex': 1, 'age': 0}]
        >>> dm = DatasetMetric(german.features, german.feature_names,
                               german.instance_weights, up_groups=u,
                               p_groups=p)
        """
        self.X = check_array(X)
        self.col_names = check_array(col_names, ensure_2d=False)
        self.sample_weights = check_array(sample_weights, ensure_2d=False)

        # TODO: should this deepcopy?
        self.p_groups = p_groups
        self.up_groups = up_groups

        # don't check if nothing was provided
        if not self.p_groups or not self.up_groups:
            return

        priv_mask = utils.compute_boolean_conditioning_vector(
            self.X, self.col_names, self.p_groups)
        unpriv_mask = utils.compute_boolean_conditioning_vector(
            self.X, self.col_names, self.up_groups)
        if np.any(np.logical_and(priv_mask, unpriv_mask)):
            raise ValueError("'privileged_groups' and 'unprivileged_groups'"
                             " must be disjoint.")
        if not np.all(np.logical_or(priv_mask, unpriv_mask)):
            warning("There are some instances in the dataset which are not "
                    "designated as either privileged or unprivileged. Are you "
                    "sure this is right?")

    def _to_condition(self, privileged):
        """Converts a boolean condition to a group-specifying format that can be
        used to create a conditioning vector.
        """
        if privileged is True and self.p_groups is None:
            raise AttributeError("'privileged_groups' was not provided when "
                                 "this object was initialized.")
        if privileged is False and self.p_groups is None:
            raise AttributeError("'unprivileged_groups' was not provided when "
                                 "this object was initialized.")

        if privileged is None:
            return None
        return self.p_groups if privileged else self.up_groups

    def difference(self, metric_fun):
        """Compute difference of the metric for unprivileged and privileged
        groups.
        """
        return metric_fun(privileged=False) - metric_fun(privileged=True)

    def ratio(self, metric_fun):
        """Compute ratio of the metric for unprivileged and privileged groups.
        """
        return metric_fun(privileged=False) / metric_fun(privileged=True)

    def num_instances(self, privileged=None):
        """Compute the number of instances, :math:`n`, in the dataset conditioned
        on protected attributes if necessary.

        Parameters
        ----------
        privileged : bool, optional
            Boolean prescribing whether to
            condition this metric on the `privileged_groups`, if `True`, or
            the `unprivileged_groups`, if `False`. Defaults to `None`
            meaning this metric is computed over the entire dataset.

        Raises
        ------
        AttributeError: `privileged_groups` or `unprivileged_groups` must be
            must be provided at initialization to condition on them.
        """
        condition = self._to_condition(privileged)
        return utils.compute_num_instances(
            self.X, self.sample_weights, self.col_names, condition=condition)
