# load data and packages

# d <- read.csv(curl("https://raw.githubusercontent.com/rene-hiroki/house_price/master/raw_data.csv"))

library(dplyr)
library(ggplot2)
library(caret)
library(curl)
load("data.rda")
d <- data.rda
d %>% ggplot(aes(date,log(price), group = date)) + geom_boxplot()

# EDA ---------------------------------------------------------------------

# glimpse the dataset 
glimpse(d)

# histogram of price

plot(d$price)
d %>% ggplot(aes(log(price))) + geom_histogram()

# boxplot
boxplot(log(d$price))

# top 5 and worst 5 price
d %>% select(price) %>% arrange(desc(price)) %>% top_n(5)
d %>% select(price) %>% arrange((price)) %>% head(5)

# remove the top 2 price row as outlier 
ind <- d$price %in% c(26590000, 12899000)  
d <- d[!ind,]
nrow(d)

# check the row of price = 0
d %>% filter(price == 0)

# alter the price = 0 to median price
med_price <- d %>% filter(price != 0) %>% 
    summarize(median_price = median(price)) %>% 
    .$median_price
ind <- d$price == 0
d$price[ind] <- med_price 

# recheck the boxplot
boxplot(log(d$price))

# remove the worst 1 price row as outlier 
d %>% select(price) %>% arrange((price)) %>% head(5)
ind <- d$price == 7800  
d <- d[!ind,]

# recheck the boxplot
boxplot(log(d$price))

# create train set and test set
set.seed(1)
test_index <- createDataPartition(y = d$price, times = 1, p = 0.1, list = FALSE)
train_set <- d %>% slice(-test_index)
test_set <- d %>% slice(test_index)


# log transform to price, sqft_living, sqft_above
train_set <- train_set %>% mutate(price = log(price))
train_set <- train_set %>% mutate(sqft_living = log(sqft_living))
train_set <- train_set %>% mutate(sqft_above = log(sqft_above))

# tally
train_set %>% group_by(waterfront) %>% tally()
train_set %>% group_by(view) %>% tally()
train_set %>% group_by(floors) %>% tally()
train_set %>% group_by(condition) %>% tally()

# preprocessing
train_set <- train_set %>% 
    mutate(floors = ifelse(floors == 1, 0, 1)) # floor 1(0) or more(1)

train_set %>% ggplot(aes(x = bedrooms, y = price, group = bedrooms)) + geom_boxplot()
train_set %>% ggplot(aes(x = bathrooms, y = price, group = bathrooms)) + geom_boxplot()
train_set %>% ggplot(aes(x = sqft_living, y = price)) + geom_point()
#train_set %>% ggplot(aes(x = sqft_lot, y = price)) + geom_point()
train_set %>% ggplot(aes(x = floors, y = price, group = floors)) + geom_boxplot()
#train_set %>% ggplot(aes(x = waterfront, y = price, group = waterfront)) + geom_boxplot()
#train_set %>% ggplot(aes(x = view, y = price, group = view)) + geom_boxplot()
# train_set %>% ggplot(aes(x = condition, y = price, group = condition)) + geom_boxplot()
train_set %>% ggplot(aes(x = sqft_above, y = price)) + geom_point()
# train_set %>% ggplot(aes(x = log(sqft_basement), y = price)) + geom_point()
train_set %>% ggplot(aes(x = city, y = price, group = city)) + geom_boxplot()
train_set %>% ggplot(aes(x = statezip, y = price, group = statezip)) + geom_boxplot()
# train_set %>% ggplot(aes(x = yr_built, y = price, group = yr_built)) + geom_boxplot()
# train_set %>% filter(yr_renovated != 0) %>% 
    # ggplot(aes(x = yr_renovated, y = price, group = yr_renovated)) + geom_boxplot()

# select features
train_set <- train_set %>% select(price, bedrooms, bathrooms, sqft_living, floors)

# linear regression
fit_1 <- train_set %>% lm(price ~ sqft_living, data =.)
fit_2 <- train_set %>% lm(price ~ sqft_living + bathrooms, data =.)
# fit_3 <- train_set %>% lm(price ~ sqft_living + bedrooms, data =.)
fit_4 <- train_set %>% lm(price ~ sqft_living + bathrooms + floors, data =.)
cor(train_set)
summary(fit_1)
summary(fit_2)
summary(fit_3)
summary(fit_4) # good

# define the RMSE as this function
RMSE <- function(predicted_prices, true_prices){
    sqrt(mean((predicted_prices - true_prices)^2))
}


# preprocessing for test set
# log transform to price, sqft_living, sqft_above
test_set <- test_set %>% mutate(price = log(price))
test_set <- test_set %>% mutate(sqft_living = log(sqft_living))
test_set <- test_set %>% mutate(sqft_above = log(sqft_above))

# preprocessing
test_set <- test_set %>% 
    mutate(floors = ifelse(floors == 1, 0, 1)) # floor 1(0) or more(1)

# predict the price
predicted_prices <- predict(fit_2, test_set)
linear_model <- predicted_prices
# take off log transformation

RMSE(exp(predicted_prices), exp(test_set$price))
plot(exp(predicted_prices), exp(test_set$price))
plot(predicted_prices, test_set$price)
histogram(exp(predicted_prices))
histogram(exp(test_set$price))


RMSE(predicted_prices, test_set$price)
plot(predicted_prices, test_set$price)
histogram(predicted_prices)
histogram(test_set$price)

d %>% select(sqft_living) %>% arrange(desc(sqft_living)) %>% head
