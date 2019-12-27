import keras
from keras.models import model_from_json
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import sys
import numpy as np
file = sys.argv[1]

#json_file = open('model_num.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#loaded_model = model_from_json(loaded_model_json)

#loaded_model.load_weights('model_num.h5')
#loaded_model.save('model_num.hdf5')
loaded_model=load_model('model_num.hdf5')


img = img_to_array(load_img(file))
img = np.array(img)
img.resize(1, 50, 50, 3)

years = {1:"1050 - 1099", 2:"1100 - 1149", 3:"1150 - 1199", 4:"1200 - 1249",
         5:"1250 - 1299", 6:"1300 - 1349", 7:"1350 - 1399", 8:"1400 - 1449",
         9:"1450 - 1499", 10:"1500 - 1549", 11:"1550 - 1599", 12:"1600 - 1649",
         13:"1650 - 1699", 14:"1700 - 1749", 15:"1750 - 1799",
         16:"1800 - 1849", 17:"1850 - 1899", 18:"1900 - 1949", 19:"1950 - 1999"}

print(years[loaded_model.predict_classes(img)[0]])
