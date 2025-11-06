import loadDataHugging as ldh
import numpy as np
import random

import whiteSpaceDetector

length_data = 5000
key = "key"

# load the data 

ds = ldh.load_datasetw()["train"]["human_answers"]

dtype = np.dtype([
    ('id', np.int32),
    ('text', object),   
    ('label', np.bool_)
])
id = 0
data = []
for single_data in ds:
    temp_data = (id, single_data, False)
    if random.random() < 0.2:
        # 20% of the data is gonna get watermarked
        print(temp_data)
        watermarked_text = whiteSpaceDetector.encode_whitespace_watermark(single_data[0],
                                                                           key, 
                                                                           gamma=0.5, 
                                                                           contextFunktion=True)
        temp_data = (id, watermarked_text, True)
    data.append(temp_data)

data_np = np.array(data, dtype=dtype)
print(len(ds))

#list_data = np.array()


