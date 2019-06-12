from torch.utils.data import Dataset, DataLoader
from torch import randperm
import torch

class PreprocessedDataset(Dataset):
    """A dataset class that preprocesses image data"""
    def __init__(self, 
                 dataset, 
                 convert_data_attributes=False, 
                 image_files_to_data_tensor=False, 
                 max_size=None, 
                 to_tensor=False, 
                 split=None,
                 num_labels=None,
                 center_and_normalize=False,
                 seed=1):
        """Applies modifications to a dataset to fit a standard format"""
        
        self.labeled = None
        self.unlabeled = None
        self.seed = torch.manual_seed(seed)
        
        if image_files_to_data_tensor:
            if max_size is not None:
                dataset = self.image_files_to_data_tensor(dataset, max_size)
            else:
                dataset = self.image_files_to_data_tensor(dataset)
        
        if convert_data_attributes:
            dataset = self.convert_data_attributes(dataset)
        
        if to_tensor:
            dataset = self.convert_numpy_to_tensor(dataset)
        
        if max_size is not None:
            dataset = self.reduce_size(dataset, max_size)
            
        if split is not None or num_labels is not None:
            dataset = self.split_unlabeled(dataset, split, num_labels)
            
        if center_and_normalize:
            dataset = self.center_normalize(dataset)
            
        self.data = dataset.data
        self.labels = dataset.labels
        
    def image_files_to_data_tensor(self, dataset, length=None):
        """Extracts images from a dataset object into a data tensor and 
        sets the data as an attribute for the dataset object (only works
        for small datasets).
        Args:
            dataset (Dataset): a dataset object
            length (int): the maximum number of images to extract
        Returns:
            dataset (DataWrapper): a dataset with 'data' and 'labels' as attributes
        """
        if length is None:
            length = len(dataset)
        # Shuffle the dataset so that we take a random subset
        loader = DataLoader(dataset, 
                            batch_size=length, 
                            num_workers=12,
                            shuffle=True)
        for data, labels in loader:
            dataset = DataWrapper(data, labels)
            # Do this only once - shuffle the dataset and take the first x items
            break
        return dataset
        
    def convert_data_attributes(self, dataset):
        """For datasets from torchvision. Set the attributes simply as 'data' and 'labels'."""
        if dataset.train:
            dataset.data = dataset.train_data
            dataset.labels = dataset.train_labels
        else:
            dataset.data = dataset.test_data
            dataset.labels = dataset.test_labels
        return dataset
    
    def convert_numpy_to_tensor(self, dataset):
        """For use on the Cifar10 dataset, where some items are in numpy array format. Converts
        data and labels to torch.Tensor objects.
        Args:
            dataset (Dataset): a dataset object with the data in numpy array format
        Returns:
            dataset (Dataset): a dataset object with the data in torch.Tensor format
        """
        dataset.data = torch.from_numpy(dataset.data)
        dataset.labels = torch.LongTensor(dataset.labels)
        return dataset
        
    def reduce_size(self, dataset, max_size):
        """Takes a random permutation of images and only keeps the first max_size of the them.
        Args:
            dataset (Dataset): a dataset object
            max_size (int): the maximum number of items to keep
        Returns:
            dataset (Dataset): a dataset object of shortened length
        """
        indices = randperm(len(dataset), generator=self.seed).tolist()
        dataset.data = dataset.data[indices[:max_size]]
        dataset.labels = dataset.labels[indices[:max_size]]
        return dataset
    
    def split_unlabeled(self, dataset, split=None, num_labels=None):
        """Removes labels for split % of the dataset, replaces those labels with -1
        Args:
            dataset (Dataset): a dataset object with all items labeled
            split (float [0, 1]): the proportion of training data for which labels are provided 
                (specify this parameter XOR the next parameter)
            num_labels (int): the number of items for which labels are provided
        Returns:
            dataset (Dataset): a dataset object with some of the labels replaced with -1
        """
        x = dataset.data
        y = dataset.labels
        
        # Two options for splitting
        if split is not None:
            # Split train into labeled/unlabeled, simulating semi-supervised learning
            self.labeled = int(split * len(x))
            self.unlabeled = len(x) - self.labeled
        elif num_labels is not None:
            self.labeled = num_labels
            self.unlabeled = len(x) - num_labels
        else:
            pass
            
        lengths = [self.labeled, self.unlabeled]
        # Randomize the indices
        indices = randperm(sum(lengths), generator=self.seed).tolist()
        x_labeled = x[indices[:self.labeled]]
        y_labeled = y[indices[:self.labeled]]
        x_unlabeled = x[indices[-self.unlabeled:]]
        y_unlabeled = torch.ones(self.unlabeled).long() * -1 # -1 indicates a missing label - See https://umap-learn.readthedocs.io/en/latest/supervised.html#using-partial-labelling-semi-supervised-umap
        
        dataset.data = torch.cat((x_labeled, x_unlabeled))
        dataset.labels = torch.cat((y_labeled.long(), y_unlabeled))
        
        return dataset
    
    def center_normalize(self, dataset):
        """Gives the images mean 0 and std_dev 1 over each channel. Based on:
        https://machinelearningmastery.com/how-to-manually-scale-image-pixel-data-for-deep-learning/
        Args:
            dataset (Dataset): a dataset object containing images with 'data' (N x C x H x W) as an attribute
        Returns:
            dataset (Dataset): a dataset object with 'data' having mean 0 and std 1 for each channel for each image
        """
        
        # Take the mean/std over the spatial dimensions, preserving batch and channel - "local centering"
        mean = torch.mean(dataset.data, dim=(2, 3), keepdim=True)
        std = torch.std(dataset.data, dim=[2, 3], keepdim=True)
        dataset.data = (dataset.data - mean) / std
        
        return dataset
        
    def append_test(self, 
                    dataset, 
                    convert_data_attributes=False, 
                    image_files_to_data_tensor=False, 
                    max_size=None, 
                    to_tensor=False,
                    center_and_normalize=False):
        """Appends a test set to the train set and removes labels. Used for transductive classifiers
        Args:
            dataset (Dataset): an unmodified dataset object
            convert_data_attributes (bool): whether to set 'data' and 'label' attributes
            image_files_to_data_tensor (bool): whether to extract files from a dataset object an place
                them in a data tensor, setting that tensor as an attribute
            max_size (int): the maximum length of the dataset
            to_tensor (bool): whether to convert 'data' and 'label' attributes to torch.Tensor objects
            center_and_normalize (bool): whether to center/normalize image data
        """
        
        if convert_data_attributes:
            dataset = self.convert_data_attributes(dataset)
            
        if image_files_to_data_tensor:
            if max_size is not None:
                dataset = self.image_files_to_data_tensor(dataset, max_size)
            else:
                dataset = self.image_files_to_data_tensor(dataset)
        
        if to_tensor:
            dataset = self.convert_numpy_to_tensor(dataset)
        
        if max_size is not None:
            dataset = self.reduce_size(dataset, max_size)
            
        if center_and_normalize:
            dataset = self.center_normalize(dataset)
            
        # Preserve test labels separately from dataset
        self.test_data = dataset.data
        self.test_labels = dataset.labels
        # Remove all test labels from dataset
        split = 0.
        dataset = self.split_unlabeled(dataset, split)
            
        self.data = torch.cat((self.data, dataset.data), dim=0)
        self.labels = torch.cat((self.labels, dataset.labels), dim=0)
    
    def __getitem__(self, i):
        return self.data[i], self.labels[i]
    
    def __len__(self):
        return len(self.labels)
    
class DataWrapper(Dataset):
    """A dataset object that wraps the 'data' and 'labels' attributes extracted
    from another source"""
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __getitem__(self, i):
        return self.data[i], self.labels[i]
    
    def __len__(self):
        return len(self.data)