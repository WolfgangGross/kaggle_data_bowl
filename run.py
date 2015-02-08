from img_routine import *
from validation import *
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.metrics import classification_report
import pandas as pd
import datetime
import warnings
warnings.filterwarnings("ignore")

WRITE_FILE = True
RUN_ON_TEST = True
X, y, X_test_img, img_labels= load_img(RUN_IMAGE_ROUTINE=False, RUN_ON_TEST=RUN_ON_TEST)
y_vec = get_one_to_all_vec(y, num_classes=121)


#####s##Random Forrest Calassifier###########
print 'Training kFolds for getting the predictions for computing the log-loss function...',
kf = KFold(y, n_folds=5)
y_pred = np.zeros((len(y),len(set(y))))     # prediction probabilities number of samples, by number of classes
y_pred_single_value = y * 0
for train, test in kf:
    X_train, X_test, y_train, y_test = X[train,:], X[test,:], y[train], y[test]
    clf = RF(n_estimators=100, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred[test] = clf.predict_proba(X_test)
    y_pred_single_value[test] = clf.predict(X_test)
print 'Done'
print classification_report(y, y_pred_single_value, target_names=get_class_names())
print 'LogLoss: ' + str(multiclass_log_loss(y,y_pred))      # Get the probability predictions for computing the log-loss function

######Make Predictions on the test set#####
print 'Predicting...',
if RUN_ON_TEST: y_test_pred = clf.predict_proba(X_test_img)
print 'Done'

if WRITE_FILE:
    #define header here ...
    out = pd.DataFrame(data=y_test_pred, index=img_labels)
    out.to_csv(os.path.join('pred', 'out')+datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+'.csv')
    print 'Save File'