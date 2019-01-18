import argparse
import sys
import pickle
import configs
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import utils 
from keras.utils import to_categorical
import numpy as np
import os
from datetime import datetime

def main(args):
    with open(args.data_path, "rb") as infile:
        (last_output, labels) = pickle.load(infile)
    
    enc = LabelEncoder()
    enc.fit(list(set(labels)))
    labels = enc.transform(labels)
    labels = to_categorical(labels)
        
    if not args.test:
        train_feature, val_feature, train_labels, val_labels = train_test_split(last_output, 
                                                                            labels, test_size = 0.1, 
                                                                            random_state = 1773)
    else:
        feature = last_output
        if args.ckpt_path == "not set": 
            print("You have to set the path for the model")
            exit(0)
    
    def next_batch(feature, labels, shuffle=False):
        if shuffle == True:
            feature, labels = utils.shuffle(feature, labels)
        num_samples = len(labels)
        num_batch = int(np.ceil(num_samples/configs.batch_size))
        x_batch, y_batch =  [], []
        for i in range(num_batch):
            start_idx = i*configs.batch_size
            end_idx = min(start_idx + configs.batch_size, num_samples) 
            x_batch = feature[start_idx:end_idx,:,:,:]
            y_batch = labels[start_idx:end_idx,:]
            yield x_batch, y_batch
    
    model_type = args.data_path[:args.data_path.find("_")] # The data's name contain the model's type
    feature_shape = configs.get_feature_shape(model_type)
    X = tf.placeholder(dtype = tf.float32, shape = feature_shape)
    Y = tf.placeholder(dtype = tf.float32, shape = [None, configs.num_labels])
    global_step = tf.Variable(0, trainable=False, name="global_step")

    o = tf.layers.AveragePooling2D(pool_size=(7,7), strides=(7,7))(X)
    o = tf.layers.Flatten()(o)
    pred = tf.layers.Dense(configs.num_labels)(o)

    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=pred))
    train_op = tf.train.AdamOptimizer(learning_rate=configs.learning_rate).minimize(loss_op, global_step=global_step)

    saver = tf.train.Saver()
    with tf.Session() as sess: 
        sess.run(tf.global_variables_initializer())
        
        if not args.test:
            best_acc = 0
            epochs_with_no_improvement = 0

            time = datetime.strftime(datetime.now(), "%Y%m%d-%H%M%S")
            os.makedirs("./model/{}/{}".format(model_type, time))    
            for epoch in range(configs.epochs):
                for feed_x, feed_y in next_batch(train_feature, train_labels):
                    loss, _ = sess.run([loss_op, train_op], feed_dict={X: feed_x, Y: feed_y})

                print("Epoch {}, loss: {}, global step: {}\n".format(epoch, loss, tf.train.global_step(sess,global_step)))
                acc = []
                for val_x, val_y in next_batch(val_feature, val_labels, shuffle=True):
                    prediction = sess.run(pred, feed_dict = {X: val_x})
                    prediction = np.argmax(prediction, axis = 1)
                    val_y = np.argmax(val_y, axis = 1)
                    accuracy = np.mean(np.equal(prediction, val_y))
                    acc.append(accuracy)
                mean_acc = np.mean(acc)
                print("Cross validation accuracy: {0:.3f}".format(mean_acc))
                if mean_acc > best_acc:
                    epochs_with_no_improvement = 0
                    best_acc = mean_acc
                    saver.save(sess, "./model/{}/{}/{}".format(model_type, time, model_type), global_step=global_step)
                    print("Accuracy increased, model saved.\n")
                else:
                    epochs_with_no_improvement += 1
                    print("No improvement after {} epochs\n".format(epochs_with_no_improvement))
                    if epochs_with_no_improvement == configs.patience:
                        print("Training terminated, top 1 error: {0:.3f}".format(1-best_acc))
                        break;
            if epochs_with_no_improvement < configs.patience:
                print("Top 1 error: {0:.3f}".format(1-best_acc))
                print("Might need for training. Last count without improvement: {}".format(epochs_with_no_improvement))
        else:
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(args.ckpt_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
            prediction = sess.run(pred, feed_dict = {X: feature})
            prediction = np.argmax(prediction, axis = 1)
            y_val = np.argmax(labels, axis = 1)
            accuracy = np.mean(np.equal(prediction, y_val))
            print("%.03f" % accuracy)
            with open("{}/_accuracy.txt".format(args.ckpt_path), "w") as outfile:
                outfile.write(str(accuracy))
    
def argument_parser(argv):
    parser = argparse.ArgumentParser(description="For finetuning or testing.")
    
    parser.add_argument("data_path", type=str,
                        help="Path for the training data")
    parser.add_argument("-t", "--test", action="store_true",
                        help="Using this flag will switch to testing finetuned model")
    parser.add_argument("-p","--ckpt_path", type=str,
                        help="Path to checkpoint folder for finetuned models", default="not set")
    
    return parser.parse_args(argv)
    
if __name__ == "__main__":
    main(argument_parser(sys.argv[1:]))
