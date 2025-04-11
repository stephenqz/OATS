from torchvision import datasets, transforms

# From: https://github.com/facebookresearch/deit/blob/ab5715372db8c6cad5740714b2216d55aeae052e/datasets.py

def build_dataset(dataset, image_processor):
    if dataset == 'imnet_cal':
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
        ])

        prepared_dataset = datasets.ImageFolder("./imagenet/train", transform=transform) # fill in path to imagenet/train

    elif dataset == 'imnet_val':

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
        ])

        prepared_dataset = datasets.ImageFolder("./imagenet/val", transform=transform) # fill in path to imagenet/val

    return prepared_dataset
