# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('Salary_Data.csv')

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$YearsExperience, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting Simple Linear Regression to the Training set
regressor = lm(Salary ~ YearsExperience, data = training_set)

# Predicting the Test set results
y_pred = predict(regressor, newdata = test_set)

# Visualizing the Training set results
library(ggplot2)
ggplot() + 
  geom_point(aes(x = training_set$YearsExperience, y= training_set$Salary),
             color = "red") + 
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set))) +
  ggtitle("Salary vs Experience (Training set)") +
  xlab('Years Experiece') +
  ylab('Salary')
  

ggplot() + 
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
             color = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set))) +
  ggtitle("Salary vs Experience (Test set)") + 
  xlab("Years Experience") +
  ylab('Salary')
