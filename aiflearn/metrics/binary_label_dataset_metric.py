import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_array

from aiflearn.datasets import BinaryLabelDataset
from aiflearn.metrics import DatasetMetric, utils


class BinaryLabelDatasetMetric(DatasetMetric):
    """Class for computing metrics based on a single
    :obj:`~aiflearn.datasets.BinaryLabelDataset`.
    """

    def __init__(self, X, y, col_names, sample_weights, pos_label, neg_label,
                 up_groups=None, p_groups=None):
        """
        Parameters
        ----------
        X : array((n, m))
            The input array
        y : array(shape=(n,))
            Output values
        col_names : list, 1d array
            Column names
        sample_weights: list, 1d array
            Sample weights
        pos_label : float
            Positive/Favorable value of y
        neg_label : float
            Negative/unfavorable value of y
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
        """
        super().__init__(X, col_names, sample_weights,
                         up_groups=up_groups, p_groups=p_groups)
        self.y = check_array(y, ensure_2d=False)
        self.pos_label = pos_label
        self.neg_label = neg_label

    def num_positives(self, privileged=None):
        r"""Compute the number of positives,
        :math:`P = \sum_{i=1}^n \mathbb{1}[y_i = 1]`,
        optionally conditioned on protected attributes.

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
        return utils.compute_num_pos_neg(
            self.X,
            self.y, self.sample_weights,
            self.col_names,
            self.pos_label, condition=condition)

    def num_negatives(self, privileged=None):
        r"""Compute the number of negatives,
        :math:`N = \sum_{i=1}^n \mathbb{1}[y_i = 0]`, optionally conditioned on
        protected attributes.

        Args:
            privileged (bool, optional): Boolean prescribing whether to
                condition this metric on the `privileged_groups`, if `True`, or
                the `unprivileged_groups`, if `False`. Defaults to `None`
                meaning this metric is computed over the entire dataset.

        Raises:
            AttributeError: `privileged_groups` or `unprivileged_groups` must be
                must be provided at initialization to condition on them.
        """
        condition = self._to_condition(privileged)
        return utils.compute_num_pos_neg(
            self.X,
            self.y, self.sample_weights,
            self.col_names,
            self.neg_label, condition=condition)

    def base_rate(self, privileged=None):
        """Compute the base rate, :math:`Pr(Y = 1) = P/(P+N)`, optionally
        conditioned on protected attributes.

        Args:
            privileged (bool, optional): Boolean prescribing whether to
                condition this metric on the `privileged_groups`, if `True`, or
                the `unprivileged_groups`, if `False`. Defaults to `None`
                meaning this metric is computed over the entire dataset.
        Returns:
            float: Base rate (optionally conditioned).
        """
        return (self.num_positives(privileged=privileged)
              / self.num_instances(privileged=privileged))

    def disparate_impact(self):
        r"""
        .. math::
           \frac{Pr(Y = 1 | D = \text{unprivileged})}
           {Pr(Y = 1 | D = \text{privileged})}
        """
        return self.ratio(self.base_rate)

    def statistical_parity_difference(self):
        r"""
        .. math::
           Pr(Y = 1 | D = \text{unprivileged})
           - Pr(Y = 1 | D = \text{privileged})
        """
        return self.difference(self.base_rate)

    def consistency(self, n_neighbors=5):
        r"""Individual fairness metric from [1]_ that measures how similar the
        labels are for similar instances.

        .. math::
           1 - \frac{1}{n\cdot\text{n_neighbors}}\sum_{i=1}^n |\hat{y}_i -
           \sum_{j\in\mathcal{N}_{\text{n_neighbors}}(x_i)} \hat{y}_j|

        Args:
            n_neighbors (int, optional): Number of neighbors for the knn
                computation.

        References:
            .. [1] R. Zemel, Y. Wu, K. Swersky, T. Pitassi, and C. Dwork,
               "Learning Fair Representations,"
               International Conference on Machine Learning, 2013.
        """

        X = self.X
        num_samples = X.shape[0]
        y = self.y

        # learn a KNN on the features
        nbrs = NearestNeighbors(n_neighbors, algorithm='ball_tree').fit(X)
        _, indices = nbrs.kneighbors(X)

        # compute consistency score
        consistency = 0.0
        for i in range(num_samples):
            consistency += np.abs(y[i] - np.mean(y[indices[i]]))
        consistency = 1.0 - consistency/num_samples

        return consistency

    # ============================== ALIASES ==================================
    def mean_difference(self):
        """Alias of :meth:`statistical_parity_difference`."""
        return self.statistical_parity_difference()
