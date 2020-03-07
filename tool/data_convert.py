import os
train_file=open('data/train_data_VOC/ImageSets/Main/train.txt','w')
test_file=open('data/test_VOC/ImageSets/Main/test.txt','w')
for _,_,train_files in os.walk('data/train_data_VOC/JPEGImages'):
    continue
for _,_,test_files in os.walk('data/test_VOC/JPEGImages'):
    continue
for file in train_files:
    train_file.write(file.split('.')[0]+'\n')

for file in test_files:
    test_file.write(file.split('.')[0]+'\n')