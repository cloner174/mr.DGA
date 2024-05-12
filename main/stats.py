import numpy as np
from scipy import stats


class NameHelper:
    def __init__(self):
        pass
    def __str__(self):
        import uuid
        return f"{uuid.uuid4().hex}"


class Stats:
    
    
    def sublist(List, n):
        
        li = List
        len_ = len(li)
        max_index = ( len_ - 1 )
        splitCoef = int( round(  ( len_ / n ), 0 ) )
        split_index = list( ( i for i in range(0, max_index, splitCoef) ) )
        
        ll = []
        for i in range( len(split_index)):
            te = split_index[i]
            if i == 0:
                sub1 = li[:te+1]
                if len(sub1) >= 2:
                    ll.append(li[:te+1])
            else:
                if i == len(split_index)-1:
                    ll.append( li[te:] )
                elif te < max_index:
                    ll.append( ( li[  split_index[i-1] : split_index[i]  ]  ) )
                else:
                    pass
        
        return ll
    
    
    def stat(subset, quantiles_ = None):
        #Q = input( " Please Inter the prob Number for quantiles ...... Leave it blank for defult .. \n   You may inter here .....--->>>>.....")
        #if Q == '':
        #    quantiles = [0.2, 0.25, 0.27, 0.3, 0.33, 0.35, 0.36, 0.60, 0.61, 0.64, 0.66, 0.69, 0.72, 0.75, 0.95]
        #elif Q ==  ' ':
        #    quantiles = [0.2, 0.25, 0.27, 0.3, 0.33, 0.35, 0.36, 0.60, 0.61, 0.64, 0.66, 0.69, 0.72, 0.75, 0.95]
        #else:
        #    try:
        #        quantiles = list(json.loads(Q))
        #    except:
        #        raise ValueError( " Sorry , Something is not right ! ")
        if quantiles_:
            quantiles = quantiles_
        else:
            #quantiles = [0.5, 0.1, 0.2, 0.3, 0.35, 0.40, 0.42, 0.44, 0.48, 0.52, 0.57, 0.65, 0.70, 0.75, 0.85, 0.95]
            quantiles = [0.25, 0.75]

        
        return [
            np.mean(subset),
            np.median(subset),
            stats.mode(subset)[0],
            np.std(subset),
            np.var(subset),
            *[np.quantile(subset, q) for q in quantiles]
        ]

import matplotlib.pyplot as plt
from PIL import Image

class Images:
    
    def save(data_row, img_dir, img_size=(48, 48)):
        """Converts pixel strings to image and saves to the respective directory."""
        
        # Split string and convert to uint8 numpy array
        pixels = np.array(data_row[' pixels'].split(), dtype='uint8')
        try:
            # Reshape into image
            image = pixels.reshape(img_size)
        except ValueError:
            print(f"Error reshaping image: {img_size} may not be the right dimensions.")
            return False
        # Create a PIL image
        img = Image.fromarray(image, 'L')  # 'L' for grayscale
        # Define file path
        file_path = os.path.join(img_dir, f"{data_row.name}.png")
        # Save image
        img.save(file_path)
        return True
    
    def plot(data, num_images):
        fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
        for i, ax in enumerate(axes):
            # Process each image
            # Split the string by space and convert to numpy array of integers
            pixel_values = np.array(data[' pixels'].iloc[i].split(), dtype='uint8')
            # Assume the images are 48x48
            image_array = pixel_values.reshape(48, 48)
            ax.imshow(image_array, cmap='gray')
            ax.set_title(f'Emotion: {data["emotion"].iloc[i]}')
            ax.axis('off')
        
        plt.show()


import os
class Utilities:
    
    def strings_with_number_inside( data,
                                    column_range_starts : int = 0,
                                    column_range_ends : int = None,
                                    seperator = None):
        start = column_range_starts
        end = int( data.shape[1] ) if column_range_ends is None else column_range_ends
        new_data = {}
        try :
            for i in range( data.shape[0] ) :
                temp_new_data = []
                for j in range( start, end ) :
                    cells_real_values = np.array(data.iloc[i, j].split(sep = seperator), dtype='uint8')
                    temp_new_data.append(cells_real_values)
                new_data[i] = temp_new_data

            return new_data
        except:
            if isinstance( data , np.array ) :
                pass
            else:
                raise TypeError( " data just could be  PandasDataFrame  or  NumpyArray  ! ")    
    
    
    def create_directories(base_path, labels):
        """Creates directories for each label if they don't already exist."""
        directories = {}
        for label in labels:
            dir_path = os.path.join(base_path, str(label))
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            directories[label] = dir_path
        
        return directories

#end#