from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from lime import lime_image
import numpy as np
from tensorflow.keras.utils import to_categorical

def compute_genoSig(imageset,label, sig_class, epochs=100):
    # imageset contains the genomaps in shape (cellNum, rowNum, colNum, 1)
    # label: dataframe of shape (cellNum, 1) containing the cell labels
    # sig_class: numpy array containing the cell classes for which gene signatures will be computed

    n_clusters = len(np.unique(label))
    # Create the deep learning model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(imageset.shape[1], imageset.shape[2], imageset.shape[3])))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(n_clusters + 1, activation='softmax'))

    # Compile and train the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(imageset, label, batch_size=32, epochs=epochs)

    label = to_categorical(label)
    # Use LIME for interpretability
    explainer = lime_image.LimeImageExplainer(verbose = False)
    imgs=[]
    lbls=[]
    #sig_classARR = np.asarray([np.argmax(row) for row in sig_class])
    sig_classARR=sig_class
    labelARR = np.asarray([np.argmax(row) for row in label])
    for i in range(sig_classARR.shape[0]):  # Assuming you have 10 classes
        # Get images of class i
        #idx = [label == sig_class[i]]
        matching_indices_full = np.where(np.isin(labelARR, sig_classARR[i]))[0]
        img=imageset[matching_indices_full]
        imgs.append(img)
        lbl=label[matching_indices_full]
        lbls.append(lbl)

    # Convert the list to a numpy array

    imgs = np.vstack(imgs)
    lbls = np.vstack(lbls)

    catLime_images = []
    # Select a random image from the test set
    for i in range(imgs.shape[0]):

        image = imgs[i]
        ground_truth = np.argmax(lbls[i])
        # Generate an explanation using LIME
        explanation = explainer.explain_instance(image, model.predict, top_labels=n_clusters, hide_color=0, num_samples=100)
        
        # Get the explanation for the ground truth class
        lime_img, mask = explanation.get_image_and_mask(ground_truth, positive_only=False, num_features=imgs.shape[1]*imgs.shape[2], hide_rest=False)
        gray_image = rgb2gray(lime_img)
        catLime_images.append(gray_image)
    catLime_images = np.array(catLime_images)

    lblsARR = np.asarray([np.argmax(row) for row in lbls])
    mean_images = []
    for i in range(sig_classARR.shape[0]):
        matching_indices_full = np.where(np.isin(lblsARR, sig_classARR[i]))[0]
        class_i_images = catLime_images[matching_indices_full]
        mean_image = np.mean(class_i_images, axis=0)
        mean_images.append(mean_image)

    return mean_images


def rgb2gray(image):
    return np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])

import numpy as np
import pandas as pd

def arrays_to_dataframe(arrays, strings):
    # converts a numpy array to a panda dataframe
    # Check if the number of arrays is even
    if len(arrays) % 2 != 0:
        raise ValueError("The number of arrays should be even.")

    # Check if the number of strings is half the number of arrays
    if len(strings) != len(arrays) // 2:
        raise ValueError("The number of strings should be half the number of arrays.")

    # Initialize an empty DataFrame
    df = pd.DataFrame()

    # For each pair of arrays
    for i in range(0, len(arrays), 2):
        # Get the corresponding string
        string = np.squeeze(strings[i // 2])

        # Add the two arrays to the DataFrame as columns with the same string
        df[str(string) + '_geneName'] = arrays[i]
        df[str(string) + '_importance'] = arrays[i + 1]

    return df

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def genoSig(genoMaps,T,label,userPD,gene_names, epochs=100):

    """
    Returns the gene names and their importance score in the range of 0 to 255 in a specific data class

    Parameters
    ----------
    genoMaps : ndarray, shape (cellNum, rowNum, colNum, 1)
    T: numpy array, shape (geneNum, geneNum)
        transfer function that converts the transformation of 1D to 2D.
    label : numpy array,
         cell labels of the data
    userPD : numpy array,
         the classes for which gene signature should be computed

    Returns
    -------
    result : panda dataframe containing the gene names and their importance scores in different classes
    """

    genoMaps_3d = np.repeat(genoMaps, 3, axis=-1)

    # first, convert the strings to integer labels
    label_encoder = LabelEncoder()
    label_encoded = label_encoder.fit_transform(label)

    lc = np.array([])
    for i in range(0, userPD.shape[0]):
        first_occurrence = (label == userPD[i]).idxmax()
        lc = np.append(lc, label_encoded[first_occurrence[0]])

    lc = np.array(lc)
    n_clusters = len(np.unique(label))
    y_train = to_categorical(label_encoded)
    meanI = compute_genoSig(genoMaps_3d, label_encoded, lc, epochs=epochs)

    result = pd.DataFrame()

    for ii in range(0, len(meanI)):
        meanIX = meanI[ii]
        Ivec = np.reshape(meanIX, (1, meanIX.shape[0] * meanIX.shape[1]), order='F').copy()
        gene_namesN = np.array(gene_names)
        row_indices = np.argmax(T, axis=0)
        gene_namesT = gene_namesN[row_indices]
        Ivec = np.squeeze(Ivec)
        indices = np.argsort(Ivec)[::-1]
        Ivec_sortedX = Ivec[indices]
        gene_namesTX = gene_namesT[indices]
        labelD = np.array(label)
        strings = [userPD[ii]]
        arrays = np.vstack((gene_namesTX, Ivec_sortedX))
        df = arrays_to_dataframe(arrays, strings)
        result = horizontal_concat = pd.concat([result, df], axis=1)

    return result