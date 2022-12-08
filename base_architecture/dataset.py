import cv2
from torchvision.datasets import ImageFolder

class Dataset(ImageFolder):
    def __init__(self, config=None,transform=None):
        # samples: list of image paths
        # labels: list of label
        self.samples = []
        self.labels = []
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # return labels
        '''
        :param index: index of data
        :return: img(tensor), targets
        '''
        image_path = self.samples[index]
        img = cv2.imread(image_path)
        target = self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        return img,target