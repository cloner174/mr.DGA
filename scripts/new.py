import numpy as np
import pandas as pd
from PIL import Image

# Load Data
data = pd.read_csv('data/dataset.csv')

#print(type(data),
#      data.head(),
#      data.columns)

# extract pixels, for an example image
example_pic = data[' pixels'].iloc[0]

# split pixel values from the string and count the number of values
pixel_values = np.array(example_pic.split(), dtype=int)
pixel_count = len(pixel_values)

# assuming the images are square, so find the closest square root
image_dim = int(pixel_count ** 0.5)

# grayscale images, only one channel
channels = 1

# guess the dimensions
print("Inferred image dimensions:", image_dim, "x", image_dim, "with", channels, "channel")

height =  width = image_dim

image_array = pixel_values.reshape(height, width, channels)

# array to a PIL Image
image = Image.fromarray(image_array.astype('uint8').squeeze(), mode='L')  # L for grayscale

image.show()


def manual_scale(Manual_X) :
        
        for i in range(Manual_X.shape[0]) :
            for j in range(Manual_X.shape[1]) :
                
                
                temp = Manual_X[i,j]
                if temp > 0.49 :
                    Manual_X[i,j] = 0
                else:
                    Manual_X[i,j] = 1
        return Manual_X