import os
import numpy as np
data_config = "config/custom.data"
data_config = parse_data_config(data_config)
valid_path = data_config["valid"]

list_path = path
imgFiles = []
with open(list_path, "r") as file:
    basePath = "data/custom/images/"  # file.readlines()[0]
    for animal in os.listdir(basePath):
        for img in os.listdir(basePath + "/" + animal):
            imgFiles = np.append(imgFiles, basePath + "/" + animal + "/" + img)
            with open("valid.txt", "a+") as file:
                file.write(basePath + animal + "/" + img + "\n")