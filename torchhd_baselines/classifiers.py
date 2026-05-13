from typing import Optional, Callable, Iterable, Tuple
from typing_extensions import Literal
import math
import scipy.linalg
from tqdm import trange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, LongTensor

import torchhd.functional as functional
from torchhd.embeddings import Random, Level, Projection, Sinusoid, Density
from torchhd.models import Centroid, IntRVFL as IntRVFLModel

DataLoader = Iterable[Tuple[Tensor, LongTensor]]

__all__ = [
    "Classifier",
    "Vanilla",
    "QuantHD",
    "DistHD",
]

class Classifier(nn.Module):
    r"""Base class for all classifiers

    Args:
        n_features (int): Size of each input sample.
        n_dimensions (int): The number of hidden dimensions to use.
        n_classes (int): The number of classes.
        device (``torch.device``, optional):  the desired device of the weights. Default: if ``None``, uses the current device for the default tensor type (see ``torch.set_default_tensor_type()``). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        dtype (``torch.dtype``, optional): the desired data type of the weights. Default: if ``None``, uses ``torch.get_default_dtype()``.

    """

    encoder: Callable[[Tensor], Tensor]
    model: Callable[[Tensor], Tensor]

    def __init__(
        self,
        n_features: int,
        n_dimensions: int,
        n_classes: int,
        *,
        device: torch.device = None,
        dtype: torch.dtype = None
    ) -> None:
        super().__init__()

        self.n_features = n_features
        self.n_dimensions = n_dimensions
        self.n_classes = n_classes

    @property
    def device(self) -> torch.device:
        return self.model.weight.device

    def forward(self, samples: Tensor) -> Tensor:
        return self.model(self.encoder(samples))

    def __call__(self, samples: Tensor) -> Tensor:
        """Evaluate the logits of the classifier for the given samples.

        Args:
            samples (Tensor): Batch of samples to be classified.

        Returns:
            Tensor: Logits of each sample for each class.

        """
        return super().__call__(samples)

    def fit(self, data_loader: DataLoader):
        """Fits the classifier to the provided data.

        Args:
            data_loader (DataLoader): Iterable of tuples containing a batch of samples and labels.

        Returns:
            self

        """
        raise NotImplementedError()

    def predict(self, samples: Tensor) -> LongTensor:
        """Predict the class of each given sample.

        Args:
            samples (Tensor): Batch of samples to be classified.

        Returns:
            LongTensor: Index of the predicted class for each sample.

        """
        return torch.argmax(self(samples), dim=-1)

    def accuracy(self, data_loader: DataLoader) -> float:
        """Accuracy in predicting the labels of the samples.

        Args:
            data_loader (DataLoader): Iterable of tuples containing a batch of samples and labels.

        Returns:
            float: The accuracy of predicting the true labels.

        """
        n_correct = 0
        n_total = 0

        for samples, labels in data_loader:
            samples = samples.to(self.device)
            labels = labels.to(self.device)

            predictions = self.predict(samples)
            n_correct += torch.sum(predictions == labels, dtype=torch.long).item()
            n_total += predictions.numel()

        return n_correct / n_total

class Vanilla(Classifier):
    r"""Baseline centroid classifier.

    This classifier uses level-hypervectors to encode the feature values which are then combined using a hash table with random keys.

    Args:
        n_features (int): Size of each input sample.
        n_dimensions (int): The number of hidden dimensions to use.
        n_classes (int): The number of classes.
        n_levels (int, optional): The number of discretized levels for the level-hypervectors.
        min_level (int, optional): The lower-bound of the range represented by the level-hypervectors.
        max_level (int, optional): The upper-bound of the range represented by the level-hypervectors.
        device (``torch.device``, optional):  the desired device of the weights. Default: if ``None``, uses the current device for the default tensor type (see ``torch.set_default_tensor_type()``). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        dtype (``torch.dtype``, optional): the desired data type of the weights. Default: if ``None``, uses ``torch.get_default_dtype()``.

    """

    model: Centroid

    def __init__(
        self,
        n_features: int,
        n_dimensions: int,
        n_classes: int,
        *,
        n_levels: int = 256,
        min_level: int = 0,
        max_level: int = 1,
        device: torch.device = None,
        dtype: torch.dtype = None
    ) -> None:
        super().__init__(
            n_features, n_dimensions, n_classes, device=device, dtype=dtype
        )

        self.keys = Random(n_features, n_dimensions, device=device, dtype=dtype)
        self.levels = Level(
            n_levels,
            n_dimensions,
            low=min_level,
            high=max_level,
            device=device,
            dtype=dtype,
        )
        self.model = Centroid(n_dimensions, n_classes, device=device, dtype=dtype)

    def encoder(self, samples: Tensor) -> Tensor:
        seq_HVs = functional.hash_table(self.keys.weight, self.levels(samples)).sign()
        return functional.bind_sequence(seq_HVs)
        # return functional.hash_table(self.keys.weight, self.levels(samples)).sign()

    def fit(self, data_loader: DataLoader):
        for samples, labels in data_loader:
            samples = samples.to(self.device)
            labels = labels[:, 0].to(self.device)

            encoded = self.encoder(samples)
            self.model.add(encoded, labels)

        return self

class QuantHD(Classifier):
    r"""Implements `QuantHD: A Quantization Framework for Hyperdimensional Computing <https://ieeexplore.ieee.org/document/8906150>`_.

    Args:
        n_features (int): Size of each input sample.
        n_dimensions (int): The number of hidden dimensions to use.
        n_classes (int): The number of classes.
        n_levels (int, optional): The number of discretized levels for the level-hypervectors.
        min_level (int, optional): The lower-bound of the range represented by the level-hypervectors.
        max_level (int, optional): The upper-bound of the range represented by the level-hypervectors.
        epochs (int, optional): The number of iteration over the training data.
        lr (float, optional): The learning rate.
        device (``torch.device``, optional):  the desired device of the weights. Default: if ``None``, uses the current device for the default tensor type (see ``torch.set_default_tensor_type()``). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        dtype (``torch.dtype``, optional): the desired data type of the weights. Default: if ``None``, uses ``torch.get_default_dtype()``.

    """

    model: Centroid

    def __init__(
        self,
        n_features: int,
        n_dimensions: int,
        n_classes: int,
        *,
        n_levels: int = 256,
        min_level: int = 0,
        max_level: int = 1,
        epochs: int = 16,
        lr: float = 0.035,
        device: torch.device = None,
        dtype: torch.dtype = None
    ) -> None:
        super().__init__(
            n_features, n_dimensions, n_classes, device=device, dtype=dtype
        )

        self.epochs = epochs
        self.lr = lr

        self.feat_keys = Random(n_features, n_dimensions, device=device, dtype=dtype)
        self.levels = Level(
            n_levels,
            n_dimensions,
            low=min_level,
            high=max_level,
            device=device,
            dtype=dtype,
        )

        self.model_count = Centroid(n_dimensions, n_classes, device=device, dtype=dtype)
        self.model = Centroid(n_dimensions, n_classes, device=device, dtype=dtype)

    def encoder(self, samples: Tensor) -> Tensor:
        seq_HVs = functional.hash_table(self.feat_keys.weight, self.levels(samples)).sign()
        return functional.bind_sequence(seq_HVs)

    def binarize(self):
        self.model.weight.data = torch.sign(self.model_count.weight.data)

    def forward(self, samples: Tensor) -> Tensor:
        return self.model(self.encoder(samples), dot=True)

    def add_quantize(self, input: Tensor, target: Tensor) -> None:
        logit = self.model(input, dot=True)
        pred = logit.argmax(1)
        is_wrong = target != pred

        if is_wrong.sum().item() == 0:
            return

        input = input[is_wrong]
        target = target[is_wrong]
        pred = pred[is_wrong]

        self.model_count.weight.index_add_(0, target, input, alpha=self.lr)
        self.model_count.weight.index_add_(0, pred, input, alpha=-self.lr)

    def fit(self, data_loader: DataLoader):

        for samples, labels in data_loader:
            samples = samples.to(self.device)
            labels = labels[:, 0].to(self.device)

            samples_hv = self.encoder(samples)
            self.model_count.add(samples_hv, labels)

        self.binarize()

        for _ in trange(1, self.epochs, desc="fit"):
            for samples, labels in data_loader:
                samples = samples.to(self.device)
                labels = labels[:, 0].to(self.device)

                samples_hv = self.encoder(samples)
                self.add_quantize(samples_hv, labels)

            self.binarize()

        return self

class DistHD(Classifier):
    r"""Implements `DistHD: A Learner-Aware Dynamic Encoding Method for Hyperdimensional Classification <https://ieeexplore.ieee.org/document/10247876>`_.

    Args:
        n_features (int): Size of each input sample.
        n_dimensions (int): The number of hidden dimensions to use.
        n_classes (int): The number of classes.
        regen_freq (int): The frequency in epochs at which to regenerate hidden dimensions.
        regen_rate (int): The fraction of hidden dimensions to regenerate.
        alpha (float): Parameter effecting the dimensions to regenerate, see paper for details.
        beta (float): Parameter effecting the dimensions to regenerate, see paper for details.
        theta (float): Parameter effecting the dimensions to regenerate, see paper for details.
        epochs (int): The number of iteration over the training data.
        lr (float): The learning rate.
        device (``torch.device``, optional):  the desired device of the weights. Default: if ``None``, uses the current device for the default tensor type (see ``torch.set_default_tensor_type()``). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        dtype (``torch.dtype``, optional): the desired data type of the weights. Default: if ``None``, uses ``torch.get_default_dtype()``.

    """

    encoder: Projection
    model: Centroid

    def __init__(
        self,
        n_features: int,
        n_dimensions: int,
        n_classes: int,
        *,
        regen_freq: int = 20,
        regen_rate: float = 0.04,
        alpha: float = 0.5,
        beta: float = 1,
        theta: float = 0.25,
        epochs: int = 10,
        lr: float = 0.05,
        device: torch.device = None,
        dtype: torch.dtype = None
    ) -> None:
        super().__init__(
            n_features, n_dimensions, n_classes, device=device, dtype=dtype
        )

        self.regen_freq = regen_freq
        self.regen_rate = regen_rate
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.epochs = epochs
        self.lr = lr

        self.encoder = Projection(n_features, n_dimensions, device=device, dtype=dtype)
        self.model = Centroid(n_dimensions, n_classes, device=device, dtype=dtype)
        self._binary_weight = None

    def fit(self, data_loader: DataLoader):

        n_regen_dims = math.ceil(self.regen_rate * self.n_dimensions)

        for epoch_idx in trange(self.epochs, desc="fit"):
            for samples, labels in data_loader:
                samples = samples.to(self.device)
                samples = samples.view(1, -1)
                labels = labels[:, 0].to(self.device)

                encoded = self.encoder(samples)
                # encoded = functional.bind_sequence(encoded)
                self.model.add_online(encoded, labels.long(), lr=self.lr)

            # Regenerate feature dimensions
            if (epoch_idx % self.regen_freq) == (self.regen_freq - 1):
                scores = torch.zeros(encoded.size(1), device=encoded.device)
                for samples, labels in data_loader:
                    samples = samples.to(self.device)
                    samples = samples.view(1, -1)
                    labels = labels[:, 0].to(self.device)

                    scores += self.regen_score(samples, labels.long())

                regen_dims = torch.topk(scores, n_regen_dims, largest=False).indices
                self.model.weight.data[:, regen_dims].zero_()
                self.encoder.weight.data[regen_dims, :].normal_()

                self._binary_weight = None

        return self

    def regen_score(self, samples, labels):
        encoded = self.encoder(samples)
        # encoded = functional.bind_sequence(encoded)
        scores = self.model(encoded)
        top2_preds = torch.topk(scores, k=2).indices
        pred1, pred2 = torch.unbind(top2_preds, dim=-1)
        is_wrong = pred1 != labels

        # cancel update if all predictions were correct
        if is_wrong.sum().item() == 0:
            return 0

        encoded = encoded[is_wrong]
        pred2 = pred2[is_wrong]
        labels = labels[is_wrong]
        pred1 = pred1[is_wrong]

        weight = F.normalize(self.model.weight, dim=1)

        # Partial correct
        partial = pred2 == labels

        dist2corr = torch.abs(weight[labels[partial]] - encoded[partial])
        dist2incorr = torch.abs(weight[pred1[partial]] - encoded[partial])
        partial_dist = torch.sum(
            (self.beta * dist2incorr - self.alpha * dist2corr), dim=0
        )

        # Completely incorrect
        complete = pred2 != labels
        dist2corr = torch.abs(weight[labels[complete]] - encoded[complete])
        dist2incorr1 = torch.abs(weight[pred1[complete]] - encoded[complete])
        dist2incorr2 = torch.abs(weight[pred2[complete]] - encoded[complete])
        complete_dist = torch.sum(
            (
                self.beta * dist2incorr1
                + self.theta * dist2incorr2
                - self.alpha * dist2corr
            ),
            dim=0,
        )

        return 0.5 * partial_dist + complete_dist

    def predict(self, samples: Tensor) -> LongTensor:
        # 首次调用时缓存二值化权重
        if self._binary_weight is None:
            self._binary_weight = torch.sign(self.model.weight.data)

        # 临时替换为二值化权重
        original_weight = self.model.weight.data
        self.model.weight.data = self._binary_weight

        try:
            samples = samples.view(1, -1)
            logits = self(samples)
            predictions = torch.argmax(logits, dim=-1)
        finally:
            # 恢复原始权重
            self.model.weight.data = original_weight

        return predictions
