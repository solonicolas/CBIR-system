import pickle
import numpy as np
import ipyplot
import matplotlib.pyplot as plt

def unpickle(file):
    
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_dataset(train=None, test=None):
    
    meta_data_dict = unpickle('cifar-10-batches-py/batches.meta')
    label_names = meta_data_dict[b'label_names']
    label_names = np.array(label_names)

    # train_set
    train_data = None
    train_filenames = []
    train_labels = []
    
    for i in range(1, 6): # number of batches
        train_data_dict = unpickle('cifar-10-batches-py/data_batch_{}'.format(i))
        if i == 1:
            train_data = train_data_dict[b'data']
        else:
            train_data = np.vstack((train_data, train_data_dict[b'data']))
        train_filenames += train_data_dict[b'filenames']
        train_labels += train_data_dict[b'labels']

    train_data = train_data.reshape((len(train_data), 3, 32, 32))
    train_data = np.rollaxis(train_data, 1, 4)
    train_filenames = np.array(train_filenames)
    train_labels = np.array(train_labels)

    # test set
    test_data_dict = unpickle('cifar-10-batches-py/test_batch')
    test_data = test_data_dict[b'data']
    test_filenames = test_data_dict[b'filenames']
    test_labels = test_data_dict[b'labels']

    test_data = test_data.reshape((len(test_data), 3, 32, 32))
    test_data = np.rollaxis(test_data, 1, 4)
    test_filenames = np.array(test_filenames)
    test_labels = np.array(test_labels)
    
    train_filenames = [x.decode('utf-8') for x in train_filenames]
    test_filenames = [x.decode('utf-8') for x in test_filenames]
    label_names = [x.decode('utf-8') for x in label_names]

    if train==None and test==None:
        return train_data, train_filenames, train_labels, test_data, test_filenames, test_labels, label_names
    else:
        return  train_data[:train], \
                train_filenames[:train], \
                train_labels[:train], \
                test_data[:test], \
                test_filenames[:test], \
                test_labels[:test], \
                label_names
    
def plot_results(queryImage, results, k, labels, dataset):
    
    images, _, images_labels, _, _, _, label_names = dataset
    
    results = [r[1][:k] for r in results.values()] # show only the first k results
    
    results = [img for l in results for img in l]  
    labels = np.repeat(labels, k)
   
    classes = [label_names[images_labels[int(idx)]] for idx in results]
    results = [images[int(idx)] for idx in results]
    
    # display query image
    print("Query image", queryImage[1])
    print("Class:", label_names[queryImage[2]])
    plt.figure(figsize=(10,2))
    plt.imshow(queryImage[0])
    plt.show()
    
    #display results
    ipyplot.plot_class_tabs(results, labels, max_imgs_per_tab=k, img_width=80, custom_texts=classes)