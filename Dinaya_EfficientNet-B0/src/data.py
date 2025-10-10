from torchvision import datasets, transforms
from torch.utils.data import DataLoader
IMAGENET_MEAN=[0.485,0.456,0.406]; IMAGENET_STD=[0.229,0.224,0.225]

def _tf(img_size, augment):
    to3 = transforms.Lambda(lambda img: img.convert("RGB"))
    if augment:
        train = transforms.Compose([
            to3, transforms.Resize((img_size,img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(7),
            transforms.ColorJitter(0.08,0.08,0.08,0.03),
            transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD)
        ])
    else:
        train = transforms.Compose([to3, transforms.Resize((img_size,img_size)),
            transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD)])
    test = transforms.Compose([to3, transforms.Resize((img_size,img_size)),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD)])
    return train, test

def build_loaders(root, img_size, bs, augment, workers=4):
    tr_tf, te_tf = _tf(img_size, augment)
    tr = datasets.ImageFolder(f"{root}/train", tr_tf)
    va = datasets.ImageFolder(f"{root}/val",   te_tf)
    te = datasets.ImageFolder(f"{root}/test",  te_tf)
    return (DataLoader(tr, bs, True,  workers, pin_memory=True),
            DataLoader(va, bs, False, workers, pin_memory=True),
            DataLoader(te, bs, False, workers, pin_memory=True))
