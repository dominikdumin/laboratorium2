from solution import *
from obpng import read_png, write_png
from read_image import read_image


print("- Ocena dostateczna")
renew_pictures()



print("- Ocena dobra")
image = read_image("figures/crushed.png")
erosion = own_simple_erosion(image)
write_png(erosion, "results/own_simple_erosion.png")



print("- Ocena bardzo dobra")
image = read_image("figures/crushed.png")
kernel = np.array([[0, 1, 1, 1, 0],
                   [0, 1, 1, 1, 0],
                   [1, 1, 1, 1, 1],
                   [0, 1, 1, 1, 0],
                   [0, 1, 1, 1, 0]])
erosion = own_erosion(image, kernel)
write_png(erosion, "results/own_erosion.png")
