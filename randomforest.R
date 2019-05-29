# load data and packages
load("data.rda")
library(dplyr)
library(ggplot2)
library(caret)
library(randomForest)

# define the RMSE as this function
RMSE <- function(predicted_prices, true_prices){
    sqrt(mean((predicted_prices - true_prices)^2))
}

# distinct raw data to wrangled data
d <- data.rda



# EDA ---------------------------------------------------------------------

# glance at the dataset structure
glimpse(d)

# remove the columns don't use (date, street, country)
d <- d %>% select(price, bedrooms, bathrooms, sqft_living, sqft_lot, floors,
                  waterfront, view, condition, sqft_above, sqft_basement,
                  yr_built, yr_renovated, city, statezip)

d %>% select(price, bedrooms, bathrooms, sqft_living, sqft_lot, floors,
             waterfront, view, condition, sqft_above, sqft_basement,
             yr_built, yr_renovated) %>% 
    boxplot(horizontal = TRUE)

# alter the price = 0 to median price
med_price <- d %>% filter(price != 0) %>% 
    summarize(median_price = median(price)) %>% 
    .$median_price
ind <- d$price == 0
d$price[ind] <- med_price 

# create train set and test set
set.seed(1)
test_index <- createDataPartition(y = d$price, times = 1, p = 0.1, list = FALSE)
train_set <- d %>% slice(-test_index)
test_set <- d %>% slice(test_index)

# log transform to price, sqft_living, sqft_above
train_set <- train_set %>% mutate(price = log(price))
train_set <- train_set %>% mutate(sqft_living = log(sqft_living))
train_set <- train_set %>% mutate(sqft_above = log(sqft_above))

# preprocessing
train_set <- train_set %>% 
    mutate(floors = ifelse(floors == 1, 0, 1)) # floor 1(0) or more(1)

# select features
train_set <- train_set %>% select(price, bedrooms, bathrooms, sqft_living,
                                  sqft_above, floors, city)

# preprocessing for test set
# log transform to price, sqft_living, sqft_above
test_set <- test_set %>% mutate(price = log(price))
test_set <- test_set %>% mutate(sqft_living = log(sqft_living))
test_set <- test_set %>% mutate(sqft_above = log(sqft_above))

# preprocessing
test_set <- test_set %>% 
    mutate(floors = ifelse(floors == 1, 0, 1)) # floor 1(0) or more(1)

# randomforest

fit_rf <- randomForest(price ~ ., data = train_set)
predicted_prices <- predict(fit_rf, test_set)
plot(fit_rf)
RMSE(exp(predicted_prices), exp(test_set$price))

# save to rf_model
rf_model <- predicted_prices
# take off log transformation

RMSE(exp(predicted_prices), exp(test_set$price))
plot(exp(predicted_prices), exp(test_set$price))
plot(predicted_prices, test_set$price)
histogram(exp(predicted_prices))
histogram(exp(test_set$price))

varImpPlot(fit_rf)

