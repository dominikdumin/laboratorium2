from PIL import Image
import numpy as np

def read_image(path):
    img = np.asarray(Image.open(path).convert('LA'))
    shape = np.shape(img)
    
    image = np.zeros((shape[0], shape[1]), dtype=np.uint8)

    for (y, x) in np.ndindex(image.shape):
        image[y][x] = img[y][x][0]

    return image
