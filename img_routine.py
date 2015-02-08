from IOhelper import *
from skimage.io import imread
from skimage.transform import resize
from MinorMajorRatio import getMinorMajorRatio
from scipy import ndimage
from skimage.feature import peak_local_max
from pylab import cm
import pickle

def read_img(maxPixel=25, addi_features=1):
    files = []
    i = 0    
    label = 0
    num_rows = get_num_img(get_dir_names())
    X, y = init_x_and_y(maxPixel, addi_features)
    imageSize = maxPixel * maxPixel
    print "Reading images"
    # Navigate through the list of directories
    # resize images (25x25) and add minorMajor Ratio to every file
    for folder in get_dir_names():
        for fileNameDir in os.walk(folder):   
            for fileName in fileNameDir[2]:
                # Only read in the images
                if fileName[-4:] != ".jpg":
                    continue
                # Read in the images and create the features
                nameFileImage = "{0}{1}{2}".format(fileNameDir[0], os.sep, fileName)
                image = imread(nameFileImage, as_grey=True)
                files.append(nameFileImage)
                
                #Action before resizing
                axisratio = getMinorMajorRatio(image)

                # Store the rescaled image pixels and the axis ratio
                image = resize(image, (maxPixel, maxPixel))
                X[i, 0:imageSize] = np.reshape(image, (1, imageSize))
                #Action after resizing
                #...
                #Add features to the data (* addi_feature
                X[i, imageSize] = axisratio

            
                # Store the classlabel
                y[i] = label
                i += 1

                # report progress for each 5% done
                report = [int((j+1)*num_rows/20.) for j in range(20)]
                if i in report: print np.ceil(i *100.0 / num_rows), "% done"
        label += 1
                #X contains all images data one row with pixel and ratio
                #y contains all ids for the classes, names are stored in namesClasses
    return X, y


def read_test_img(maxPixel=25, addi_features=1):
    print "Reading test images"
    files = []
    imageSize = maxPixel * maxPixel
    file_list = list(set(glob.glob(os.path.join("test", "*"))))
    X = np.zeros((len(file_list), imageSize+addi_features), dtype=float)
    i = 0
    
    for fileName in file_list:
        
        image = imread(fileName, as_grey=True)
        files.append(fileName[5:])                
        #Action before resizing
        axisratio = getMinorMajorRatio(image)
        # Store the rescaled image pixels and the axis ratio
        image = resize(image, (maxPixel, maxPixel))
        X[i, 0:imageSize] = np.reshape(image, (1, imageSize))
        #Action after resizing
        #...
        #Add features to the data (* addi_feature
        X[i, imageSize] = axisratio
        
        i += 1
        # report progress for each 5% done
        report = [int((j+1)*len(file_list)/20.) for j in range(20)]
        if i in report:print np.ceil(i *100.0 / len(file_list)), "% done"
      
    return X, files


def load_img(RUN_IMAGE_ROUTINE=False, RUN_ON_TEST= True):
    # load the data
    if RUN_IMAGE_ROUTINE:
        print 'Generating new train data...',
        x, y = read_img()
        pickle.dump(x, open(os.join.path("dump", "x.pkl"), "wb"))
        pickle.dump(y, open(os.join.path("dump", "y.pkl"), "wb"))
        print 'Pickled!'
    else:
        print 'Loading train data...',
        x = pickle.load(open("dump/x.pkl", "rb"))
        y = pickle.load(open("dump/y.pkl", "rb"))
        print 'Done'

    if RUN_ON_TEST:
        if RUN_IMAGE_ROUTINE:
            print 'Generating new test data...',
            X_test_img, img_labels = read_test_img()
            pickle.dump(X_test_img, open("dump/X_test_img.pkl", "wb"))
            pickle.dump(img_labels, open("dump/img_labels.pkl", "wb"))
            print 'Pickled!'
        else:
            print 'Loading test data...',
            X_test_img = pickle.load(open("dump/X_test_img.pkl", "rb"))
            img_labels = pickle.load(open("dump/img_labels.pkl", "rb"))
            print 'Done'
    if RUN_ON_TEST: return x, y, X_test_img, img_labels
    return x, y

def get_one_to_all_vec(y, num_classes):
    #create solution vector
    y_vec = np.zeros((len(y), num_classes))
    k = 0
    for cl_lable in y:
        y_vec[k][int(cl_lable)] = 1
        k += 1
    return y_vec