import preprocess as pre
import normalize as nor
import classification as clf


# Load training data and test data
print("Load training data ...")
X, y = pre.load_data('kddcup.data_10_percent')
print("Training data shape (%d, %d)" % (X.shape[0], X.shape[1]))
X_test, y_test = pre.load_data('corrected')
print("Training data shape (%d, %d)" % (X_test.shape[0], X_test.shape[1]))
print("Load data done \n")


# Feature normaize
print("Feature normaize ...")
X = nor.normalize(X)
X_test = nor.normalize(X_test)
print("Feature normaize done \n")

# Classification
print("Training")
model = clf.train(X, y)
print("Training done \n")


# Score
recall = model.score(X, y)

num_of_valid = 0
y_test_predict = model.predict(X_test)
for index in range(len(y_test)):
    if abs(y_test_predict[index] - y_test[index]) < 1:
        num_of_valid += 1

accuracy = num_of_valid / len(y_test_predict)
print("Recall %f, Accuracy %f \n" % (recall, accuracy))
