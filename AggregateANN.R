# Get training data
agg_data <- read.csv('agg_data.csv')

# Split data for 10 cross validation
library(caTools)
set.seed(27)
split <- sample.split(agg_data, SplitRatio = 0.98)
training_set <- subset(agg_data, split == TRUE)
test_set <- subset(agg_data, split == FALSE)

# Feature Scaling
training_set[-5] <- scale(training_set[-5])
test_set[-5] <- scale(test_set[-5])


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
pred <- h2o.predict(NN, newdata = as.h2o(test_set[-5]))

# Table showing the differences in test data outcomes
cm <- matrix(c(test_set[,5], pred), nrow = 1, ncol = 2)

# The estimated gift amount is $170,497 and the actual gift amount is $340,760
cm