from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import Compose,ToTensor,Resize,RandomCrop,RandomHorizontalFlip,Normalize
from PIL import Image
import glob

class LSUN_Dataset(Dataset):
    def __init__(self,root="/home/guyuchao/ssd/dataset/lsun-master/divided/",resolution_level=2, splits="train",transform=None):
        self.images={}
        self.images[splits]=[]
        root+="res%d/*.jpg"%resolution_level
        self.images[splits]=glob.glob(root,recursive=True)
        self.splits=splits
        self.transform=transform

    def __len__(self):
        return len(self.images[self.splits])

    def __iter__(self):
        return self

    def __getitem__(self, idx):
        img = Image.open(self.images[self.splits][idx])
        if self.transform is not None:
            img = self.transform(img)
        return img

class LSUN_Loader(object):
    def __init__(self):
        pass

    def update(self,resolution_level,batchsize):
        '''

        :param resolution_level:
            resolution_level:
                2-2,3-8,4-16,5-32,6-64,7-128,8-256
        :return:
        '''
        assert resolution_level>=2 and resolution_level<=10,"res error"
        transform = Compose([
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.dataset=LSUN_Dataset(splits="train",resolution_level=resolution_level,transform=transform)
        self.dataloader=DataLoader(dataset=self.dataset,batch_size=batchsize,shuffle=True)

    def __len__(self):
        return len(self.dataset)

    def get_batch(self):
        dataIter=iter(self.dataloader)
        return next(dataIter)

lsun_loader=LSUN_Loader()

if __name__=="__main__":
    from timeit import default_timer as timer
    pass