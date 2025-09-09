import os
import glob
import re
import datetime
import numpy as np
import pandas as pd
import torch
import torchhd as hd
from sktime.utils import load_data
from sklearn import model_selection
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchhd import embeddings
from multiprocessing import cpu_count


class Normalizer(object):
    """
    Normalizes dataframe across ALL contained rows (time steps). Different from per-sample normalization.
    """

    def __init__(self, norm_type, mean=None, std=None, min_val=None, max_val=None):
        """
        Args:
            norm_type: choose from:
                "standardization", "minmax": normalizes dataframe across ALL contained rows (time steps)
                "per_sample_std", "per_sample_minmax": normalizes each sample separately (i.e. across only its own rows)
            mean, std, min_val, max_val: optional (num_feat,) Series of pre-computed values
        """

        self.norm_type = norm_type
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, df):
        """
        Args:
            df: input dataframe
        Returns:
            df: normalized dataframe
        """
        if self.norm_type == "standardization":
            if self.mean is None:
                self.mean = df.mean()
                self.std = df.std()
            return (df - self.mean) / (self.std + np.finfo(float).eps)

        elif self.norm_type == "minmax":
            if self.max_val is None:
                self.max_val = df.max()
                self.min_val = df.min()
            return (df - self.min_val) / (self.max_val - self.min_val + np.finfo(float).eps)

        elif self.norm_type == "per_sample_std":
            grouped = df.groupby(by=df.index)
            return (df - grouped.transform('mean')) / grouped.transform('std')

        elif self.norm_type == "per_sample_minmax":
            grouped = df.groupby(by=df.index)
            min_vals = grouped.transform('min')
            return (df - min_vals) / (grouped.transform('max') - min_vals + np.finfo(float).eps)

        else:
            raise (NameError(f'Normalize method "{self.norm_type}" not implemented'))


def interpolate_missing(y):
    """
    Replaces NaN values in pd.Series `y` using linear interpolation
    """
    if y.isna().any():
        y = y.interpolate(method='linear', limit_direction='both')
    return y


class BaseData(object):

    def __init__(self):
        self.n_proc = None

    def set_num_processes(self, n_proc):

        if (n_proc is None) or (n_proc <= 0):
            self.n_proc = cpu_count()  # max(1, cpu_count() - 1)
        else:
            self.n_proc = min(n_proc, cpu_count())


class ClassiregressionDataset(Dataset):

    def __init__(self, data, indices):
        super(ClassiregressionDataset, self).__init__()

        self.data = data  # this is a subclass of the BaseData class in data.py
        self.IDs = indices  # list of data IDs, but also mapping between integer index and ID
        self.feature_df = self.data.feature_df.loc[self.IDs]

        self.labels_df = self.data.labels_df.loc[self.IDs]

    def __getitem__(self, ind):
        """
        For a given integer index, returns the corresponding (seq_length, feat_dim) array and a noise mask of same shape
        Args:
            ind: integer index of sample in dataset
        Returns:
            X: (seq_length, feat_dim) tensor of the multivariate time series corresponding to a sample
            y: (num_labels,) tensor of labels (num_labels > 1 for multi-task models) for each sample
            ID: ID of sample
        """

        X = self.feature_df.loc[self.IDs[ind]].values  # (seq_length, feat_dim) array
        y = self.labels_df.loc[self.IDs[ind]].values  # (num_labels,) array

        return torch.from_numpy(X), torch.from_numpy(y), self.IDs[ind]

    def __len__(self):
        return len(self.IDs)


def padding_mask(lengths, max_len=None):
    """
    Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
    where 1 means keep element at this position (time step)
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max_val()  # trick works because of overloading of 'or' operator for non-boolean types
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


def collate_superv(data, max_len=None):
    """Build mini-batch tensors from a list of (X, mask) tuples. Mask input. Create
    Args:
        data: len(batch_size) list of tuples (X, y).
            - X: torch tensor of shape (seq_length, feat_dim); variable seq_length.
            - y: torch tensor of shape (num_labels,) : class indices or numerical targets
                (for classification or regression, respectively). num_labels > 1 for multi-task models
        max_len: global fixed sequence length. Used for architectures requiring fixed length input,
            where the batch length cannot vary dynamically. Longer sequences are clipped, shorter are padded with 0s
    Returns:
        X: (batch_size, padded_length, feat_dim) torch tensor of masked features (input)
        targets: (batch_size, padded_length, feat_dim) torch tensor of unmasked features (output)
        target_masks: (batch_size, padded_length, feat_dim) boolean torch tensor
            0 indicates masked values to be predicted, 1 indicates unaffected/"active" feature values
        padding_masks: (batch_size, padded_length) boolean tensor, 1 means keep vector at this position, 0 means padding
    """

    batch_size = len(data)
    features, labels, IDs = zip(*data)

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in features]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)
    X = torch.zeros(batch_size, max_len, features[0].shape[-1])  # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]

    targets = torch.stack(labels, dim=0)  # (batch_size, num_labels)

    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16),
                                 max_len=max_len)  # (batch_size, padded_length) boolean tensor, "1" means keep

    return X, targets, padding_masks, IDs


class DatasetArchive(BaseData):
    """
    dataset available at: www.timeseriesclassification.com
    you only need to download and unzip the dataset, no external preprocessing is required
    """

    def __init__(self, root_dir, file_list=None, pattern=None, n_proc=1, limit_size=None, config=None):
        super().__init__()
        self.max_seq_len = None
        self.class_names = None
        self.config = config

        self.all_df, self.labels_df = self.load_all(root_dir, file_list=file_list, pattern=pattern)
        self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

    def load_all(self, root_dir, file_list, pattern=None):
        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_dir, '*'))  # list of all paths
        else:
            data_paths = [os.path.join(root_dir, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_dir, '*')))

        if pattern is None:
            # by default evaluate on
            selected_paths = data_paths
        else:
            selected_paths = list(filter(lambda x: re.search(pattern, x), data_paths))

        input_paths = [p for p in selected_paths if os.path.isfile(p) and p.endswith('.ts')]
        if len(input_paths) == 0:
            raise Exception("No .ts files found using pattern: '{}'".format(pattern))

        all_df, labels_df = self.load_single(input_paths[0])  # a single file contains dataset

        return all_df, labels_df

    def load_single(self, filepath):
        print(datetime.datetime.now())
        df, labels = load_data.load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                             replace_missing_vals_with='NaN')
        labels = pd.Series(labels, dtype="category")
        self.class_names = labels.cat.categories
        labels_df = pd.DataFrame(labels.cat.codes,
                                 dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss
        print(datetime.datetime.now())

        lengths = df.applymap(
            lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
        else:
            self.max_seq_len = lengths[0, 0]

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)
        df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
            pd.Series(lengths[row, 0] * [row])) for row in range(df.shape[0])), axis=0)

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df


def arr_normalization(train_arr, val_arr, test_arr, lower_percentile=2.5, upper_percentile=97.5):
    """normalization to [0, 1]"""
    all_arr = np.concatenate((train_arr, test_arr), axis=0)
    lower_bound = np.percentile(all_arr, lower_percentile)
    upper_bound = np.percentile(all_arr, upper_percentile)

    norm_train_arr = (train_arr - lower_bound) / (upper_bound - lower_bound + 1e-8)
    norm_val_arr = (val_arr - lower_bound) / (upper_bound - lower_bound + 1e-8)
    norm_test_arr = (test_arr - lower_bound) / (upper_bound - lower_bound + 1e-8)

    return norm_train_arr, norm_val_arr, norm_test_arr


def HD_encoder(train_features, val_features, test_features, args):
    if args.multigpu is None:
        device = torch.device("cpu")
    elif len(args.multigpu) == 1:
        device = [torch.device(f'cuda:{args.multigpu[0]}')]
    else:
        device = [torch.device(f'cuda:{gpu_id}') for gpu_id in args.multigpu]

    train_features = train_features.to(device[0])
    val_features = val_features.to(device[0])
    test_features = test_features.to(device[0])

    feature_size = train_features.size(-1)
    emb_rand = embeddings.Random(feature_size, args.HV_dim, device=device[0])
    emb_lvl = embeddings.Level(args.quant_lvl, args.HV_dim, device=device[0])
    IM_HVs = emb_rand(torch.arange(0, feature_size, dtype=torch.int32).to(device[0]))

    train_HVs = torch.zeros(train_features.size(0), train_features.size(1), args.HV_dim)
    for sample_idx in range(train_features.size(0)):
        fea_HVs = emb_lvl(train_features[sample_idx, :, :])
        for time_idx in range(train_features.size(1)):
            spatial_HV = hd.hash_table(IM_HVs, fea_HVs[time_idx, :, :])
            spatial_HV = hd.hard_quantize(spatial_HV)
            spatial_HV = hd.permute(spatial_HV, shifts=train_features.size(1)-time_idx)

            train_HVs[sample_idx, time_idx, :] = spatial_HV

    val_HVs = torch.zeros(val_features.size(0), val_features.size(1), args.HV_dim)
    for sample_idx in range(val_features.size(0)):
        fea_HVs = emb_lvl(val_features[sample_idx, :, :])
        for time_idx in range(val_features.size(1)):
            spatial_HV = hd.hash_table(IM_HVs, fea_HVs[time_idx, :, :])
            spatial_HV = hd.hard_quantize(spatial_HV)
            spatial_HV = hd.permute(spatial_HV, shifts=val_features.size(1)-time_idx)

            val_HVs[sample_idx, time_idx, :] = spatial_HV

    test_HVs = torch.zeros(test_features.size(0), test_features.size(1), args.HV_dim)
    for sample_idx in range(test_features.size(0)):
        fea_HVs = emb_lvl(test_features[sample_idx, :, :])
        for time_idx in range(test_features.size(1)):
            spatial_HV = hd.hash_table(IM_HVs, fea_HVs[time_idx, :, :])
            spatial_HV = hd.hard_quantize(spatial_HV)
            spatial_HV = hd.permute(spatial_HV, shifts=test_features.size(1)-time_idx)

            test_HVs[sample_idx, time_idx, :] = spatial_HV

    return train_HVs.to('cpu'), val_HVs.to('cpu'), test_HVs.to('cpu')


def get_dataloader_HV(train_dataset, val_dataset, test_dataset, args):
    dim = train_dataset.feature_df.shape[-1]
    train_size = train_dataset.labels_df.shape[0]
    len_ = int(train_dataset.feature_df.shape[0] / train_size)

    train_data_array = train_dataset.feature_df.to_numpy()
    train_data_array = train_data_array.reshape(train_size, len_, dim)
    train_label_array = train_dataset.labels_df.to_numpy()

    val_size = val_dataset.labels_df.shape[0]
    val_data_array = val_dataset.feature_df.to_numpy()
    val_data_array = val_data_array.reshape(val_size, len_, dim)
    val_label_array = val_dataset.labels_df.to_numpy()

    test_size = test_dataset.labels_df.shape[0]
    test_data_array = test_dataset.feature_df.to_numpy()
    test_data_array = test_data_array.reshape(test_size, len_, dim)
    test_label_array = test_dataset.labels_df.to_numpy()

    train_data_array, val_data_array, test_data_array = arr_normalization(train_data_array,
                                                                          val_data_array,
                                                                          test_data_array)

    train_data_tensor = torch.tensor(train_data_array, dtype=torch.float32)
    val_data_tensor = torch.tensor(val_data_array, dtype=torch.float32)
    test_data_tensor = torch.tensor(test_data_array, dtype=torch.float32)

    train_label_tensor = torch.tensor(train_label_array, dtype=torch.int32)
    val_label_tensor = torch.tensor(val_label_array, dtype=torch.int32)
    test_label_tensor = torch.tensor(test_label_array, dtype=torch.int32)

    print("HD encoding in process...")
    train_HV_tensor, val_HV_tensor, test_HV_tensor = HD_encoder(train_data_tensor,
                                                                val_data_tensor,
                                                                test_data_tensor, args)
    print(f"HD encoding done! D = {args.HV_dim}")

    # # position 0 random embedding
    # pos_0_HV = hd.random(1, args.HV_dim)
    # pos_0_embed = pos_0_HV.repeat(train_HV_tensor.size(0), 1, 1)
    # train_HV_tensor = torch.cat([pos_0_embed, train_HV_tensor], dim=1)
    # pos_0_embed = pos_0_HV.repeat(val_HV_tensor.size(0), 1, 1)
    # val_HV_tensor = torch.cat([pos_0_embed, val_HV_tensor], dim=1)
    # pos_0_embed = pos_0_HV.repeat(test_HV_tensor.size(0), 1, 1)
    # test_HV_tensor = torch.cat([pos_0_embed, test_HV_tensor], dim=1)

    train_dataset_tensor = TensorDataset(train_HV_tensor, train_label_tensor)
    val_dataset_tensor = TensorDataset(val_HV_tensor, val_label_tensor)
    test_dataset_tensor = TensorDataset(test_HV_tensor, test_label_tensor)

    train_loader = DataLoader(dataset=train_dataset_tensor, batch_size=args.batch_size, shuffle=True, pin_memory=True, )
    val_loader = DataLoader(dataset=val_dataset_tensor, batch_size=args.batch_size, shuffle=False, pin_memory=True, )
    test_loader = DataLoader(dataset=test_dataset_tensor, batch_size=args.batch_size, shuffle=False, pin_memory=True, )
    for batch in train_loader:
        data, label = batch
        args.window_size = data.size(1)
        print(data.size())
        break

    return train_loader, val_loader, test_loader


def get_dataset(args):
    global all_data, test_data, val_data, train_indices, val_indices, test_indices, unique_labels
    all_data = DatasetArchive(f'./datasets/{args.dataset}', pattern='TRAIN')
    test_data = DatasetArchive(f'./datasets/{args.dataset}', pattern='TEST')
    print(args.dataset)
    print(all_data.feature_df.shape)

    if args.dataset == "Handwriting":
        val_ratio = 0.2
    else:
        val_ratio = 0.05
    val_data = all_data
    labels = all_data.labels_df.values.flatten()
    unique_labels = len(set(labels))

    splitter = model_selection.StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=1)
    train_indices, val_indices = zip(*splitter.split(X=np.zeros(len(all_data.all_IDs)), y=labels))
    test_indices = test_data.all_IDs

    # train_indices = train_indices[0]  # `split_dataset` returns a list of indices *per fold/split*
    train_indices = all_data.all_IDs
    val_indices = val_indices[0]  # `split_dataset` returns a list of indices *per fold/split*

    print(("{} samples may be used for training".format(len(train_indices))))
    print("{} samples will be used for validation".format(len(val_indices)))
    print("{} samples will be used for testing".format(len(test_indices)))

    normalizer = Normalizer('standardization')
    all_data.feature_df.loc[train_indices] = normalizer.normalize(all_data.feature_df.loc[train_indices])
    all_data.feature_df.loc[val_indices] = normalizer.normalize(val_data.feature_df.loc[val_indices])
    test_data.feature_df.loc[test_indices] = normalizer.normalize(test_data.feature_df.loc[test_indices])

    train_dataset = ClassiregressionDataset(all_data, train_indices)
    val_dataset = ClassiregressionDataset(val_data, val_indices)
    test_dataset = ClassiregressionDataset(test_data, test_indices)

    input_dim = all_data.feature_df.shape[1]

    return train_dataset, val_dataset, test_dataset, unique_labels, input_dim