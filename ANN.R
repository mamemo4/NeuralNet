# Get training data
all_data <- read.csv('alldata.csv')

# Pick out three features and gift amount
dataset <- all_data[, c(14,15,21,3)]

# Split data for 10 cross validation
library(caTools)
set.seed(123)
split <- sample.split(dataset, SplitRatio = 0.9)
training_set <- subset(dataset, split == TRUE)
test_set <- subset(dataset, split == FALSE)

# Feature Scaling
training_set[-4] <- scale(training_set[-4])
test_set[-4] <- scale(test_set[-4])

# Neural Net
#install.packages('h2o')
library(h2o)
h2o.init(nthreads = -1)

NN <- h2o.deeplearning(y = 'GiftAmount', 
                       training_frame = as.h2o(training_set),
                       activation = 'Rectifier',
                       hidden = c(2,2),
                       epochs = 100,
                       train_samples_per_iteration = -2)

# Predicting the test set results
pred <- h2o.predict(NN, newdata = as.h2o(test_set[-4]))

pred <- as.vector(pred)


# Table showing the differences in test data outcomes
cm <- matrix(c(test_set[,4], pred), nrow = 6574, ncol = 2)
