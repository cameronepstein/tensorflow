import os
import skimage


def load_data(data_directory):
    # if you find something in the data_directory, check whether it's a directory
    # and if it is one then add to the directories list.
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []

    for d in directories:
        # gather the paths of the subdirectories and file names of images within directory
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f)
                      for f in os.listdir(label_directory)
                      if f.endswith(".ppm")]

        # for each image save skimage numeric data and directory number
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))

    return images, labels


# path to data
ROOT_PATH = "traffic_signs/data"
# path to train and test data (to be added to ROOT_PATH)
train_data_directory = os.path.join(ROOT_PATH, "Training")
test_data_directory = os.path.join(ROOT_PATH, "Testing")

images, labels = load_data(train_data_directory)
