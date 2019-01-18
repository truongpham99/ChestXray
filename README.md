# Chest Xray
### Description 
This is an image classification project using resnet50 model with weights on Imagenet. The network is finetuned to detect lungs with signs of pneumonia, based on their x-ray images. There are 3 classes: Normal, Bacterial Pneumonia, and Viral Pneumonia. The pipeline for the project is simple: First using **data_extraction** to get a pickle file containing the embedding vectors of the images; Then we use **classfier** to create the pretrain model and also run it on test data. 

### Data
The data can be found on [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
