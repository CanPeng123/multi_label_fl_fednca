import random
from PIL import Image
from torchvision import transforms


pre_crop_size = 256
p_train = 0.5
p_hflip = 0.5
shift_limit = 0.0625
scale_limit = ((-0.2, 0.1))
rotate_limit = 10
scale = (0.1)
brightness_limit = (-0.2, 0.2)
contrast_limit = (-0.2, 0.2)
pad_mode = Image.BILINEAR  # PIL doesn't have BORDER_CONSTANT, so we use BILINEAR for resizing
pad_val = (105/256, 105/256, 105/256)

NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)

def transforms_train_augmented(img):
    """Transform and apply augmentation to the training set images using PIL and torchvision."""
    
    # Store the original size of the image
    original_size = img.size

    # Randomly apply transformations based on probabilities
    if random.random() < p_train:
        # Shift, Scale, Rotate
        shift_x = random.uniform(-shift_limit, shift_limit) * img.width
        shift_y = random.uniform(-shift_limit, shift_limit) * img.height
        scale_factor = random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        angle = random.uniform(-rotate_limit, rotate_limit)
        
        img = img.transform(
            img.size,
            Image.AFFINE,
            (scale_factor, 0, shift_x, 0, scale_factor, shift_y),
            resample=pad_mode
        )
        img = img.rotate(angle, resample=pad_mode)

        # Perspective (simulated with random affine transformation)
        perspective_transform = transforms.RandomPerspective(distortion_scale=scale, p=1)
        img = perspective_transform(img)

    # Random Crop (crop a random region and resize back to the original size)
    if random.random() < p_train:
        crop_size = (
            int(img.width * random.uniform(0.8, 1.0)),  # Random crop size between 80% and 100%
            int(img.height * random.uniform(0.8, 1.0))
        )
        i = random.randint(0, img.width - crop_size[0])
        j = random.randint(0, img.height - crop_size[1])
        img = img.crop((i, j, i + crop_size[0], j + crop_size[1]))
        img = img.resize(original_size, resample=pad_mode)  # Resize back to the original size

    # Horizontal Flip
    if random.random() < p_hflip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # Random Brightness and Contrast
    if random.random() < p_train:
        brightness_factor = random.uniform(1 + brightness_limit[0], 1 + brightness_limit[1])
        contrast_factor = random.uniform(1 + contrast_limit[0], 1 + contrast_limit[1])
        img = transforms.functional.adjust_brightness(img, brightness_factor)
        img = transforms.functional.adjust_contrast(img, contrast_factor)

    # Normalize and convert to tensor
    # img = transforms.ToTensor()(img)
    # img = transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)(img)

    return img
