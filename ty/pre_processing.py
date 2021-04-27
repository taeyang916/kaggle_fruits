import os
import pandas as pd
import glob as glob

root_path = "/fruits-360"
Training_path = "/Training"
Test_path = "/Test"
print(os.getcwd())

def preprocessing(dataset_type):
    data = {'image_path':[], 'image_idx':[]}
    path = os.listdir(os.getcwd() + '/fruits-360/' + dataset_type)

    for i in path:
        img_path = glob.glob(f'./fruits-360/{dataset_type}/{i}/*.jpg')
        img_idx = path.index(i)
        data['image_path'].extend(img_path)
        data['image_idx'].extend(img_idx for _ in range(len(img_path)))

    dataframe = pd.DataFrame(data)
    dataframe.to_csv(f'{dataset_type}.csv', index=False)



"""
for i in os.listdir(os.getcwd() + "/fruits-360/Training"):
    img_path = glob.glob(f'./fruits-360/Training/{i}/*.jpg')
    img_idx = i
    data['image_path'].extend(img_path)
    data['image_idx'].extend(img_idx for _ in range(len(img_path)))
print(len(data['image_path']))
print(len(data['image_idx']))

with open('datatest.csv', 'w') as f:
    writer = csv.DictWriter(f, fieldnames=data)
    writer.writeheader()
    for elem in data:
        writer.writerow(elem)

dataframe = pd.DataFrame(data)
dataframe.to_csv("datatest.csv", index=False)
"""

preprocessing('Training')
preprocessing('Test')
print("==== PreProcessing is end ====")