from PIL import Image, ImageFile
import os
ImageFile.LOAD_TRUNCATED_IMAGES = True
count = 0
for f in os.listdir('train_1/'):
    imagePath = os.path.join('train_1/', f)
    img = Image.open(imagePath)
    img = img.convert('RGB')
    img.resize((200, 200), Image.ANTIALIAS).save(os.path.join("resized/",f))
    print(count)
    count+=1