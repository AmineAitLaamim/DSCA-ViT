from torchvision import transforms
from torchvision.transforms import InterpolationMode

def get_train_transform(image_size: int = 224) -> transforms.Compose:
    """
    Returns the training transforms for DSCA-ViT.
    
    The DSCA-ViT architecture applies color deconvolution as the first step,
    which requires raw RGB values in [0, 1] range. ImageNet normalization
    would corrupt the optical density computation and must NOT be applied.
    
    Args:
        image_size: Target size for the images.
        
    Returns:
        Compose object containing the transforms.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=10, interpolation=InterpolationMode.BILINEAR, fill=0),
        transforms.ToTensor()
    ])

def get_test_transform(image_size: int = 224) -> transforms.Compose:
    """
    Returns the testing/validation transforms for DSCA-ViT.
    
    The DSCA-ViT architecture applies color deconvolution as the first step,
    which requires raw RGB values in [0, 1] range. ImageNet normalization
    would corrupt the optical density computation and must NOT be applied.
    
    Args:
        image_size: Target size for the images.
        
    Returns:
        Compose object containing the transforms.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
