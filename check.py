from glob import glob
from pprint import pprint

img_list = glob('./data/*.jpg')

cat_cnt = 0
dog_cnt = 0
label = []

for i in img_list:
    label.append(0 if 'cat' in i else 1)

    if 'cat' in i:
        cat_cnt += 1
    else:
        dog_cnt += 1

print(cat_cnt, dog_cnt)

print(label.count(0), label.count(1))