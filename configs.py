num_labels = 3
epochs = 100
batch_size = 128
learning_rate = 0.01
patience = 32

def get_feature_shape(model):
    if model == "resnet50":
        return [None, 7, 7, 2048]