# load data and packages
# comment out if you download data from github directly
# d <- read.csv(curl("https://raw.githubusercontent.com/rene-hiroki/house_price/master/raw_data.csv"))

library(curl)
library(dplyr)
library(ggplot2)
library(caret)
library(randomForest)
load("data.rda")
d <- data.rda

# EDA ---------------------------------------------------------------------

# glimpse the dataset 
glimpse(d)

# plot and histogram of price
plot(d$price)
hist(d$price)

# histogram of log(price)
d %>% ggplot(aes(log(price))) + 
    geom_histogram(bins = 40, color = I("black")) +
    labs(title = "Histogram of log(price)")

# boxplot of log(price)
boxplot(log(d$price))

# top 5 and worst 5 price
d %>% select(price) %>% arrange(desc(price)) %>% top_n(5)
d %>% select(price) %>% arrange((price)) %>% head(5)

# check the row of price = 0
d %>% filter(price == 0)

# alter the price = 0 to median price
med_price <- median(d$price)
ind <- d$price == 0
d$price[ind] <- med_price 

# recheck top 5 and worst 5 price
d %>% select(price) %>% arrange(desc(price)) %>% top_n(5)
d %>% select(price) %>% arrange((price)) %>% head(5)

# create train set and test set
set.seed(1)
test_index <- createDataPartition(y = d$price, times = 1, p = 0.1, list = FALSE)
train_set <- d %>% slice(-test_index)
test_set <- d %>% slice(test_index)

# tally
train_set %>% group_by(waterfront) %>% tally()
train_set %>% group_by(view) %>% tally()
train_set %>% group_by(floors) %>% tally()
train_set %>% group_by(condition) %>% tally()

#### predictor variables vs price graphs ####
train_set %>% ggplot(aes(x = bedrooms,         y = log(price), group = bedrooms))  + geom_boxplot() + labs(title = "Price vs Bedrooms")
train_set %>% ggplot(aes(x = bathrooms,        y = log(price), group = bathrooms)) + geom_boxplot() + labs(title = "Price vs Bathrooms")
train_set %>% ggplot(aes(x = log(sqft_living), y = log(price))) + geom_point() + geom_smooth(method = "lm") + labs(title = "Price vs Sqft_living")
train_set %>% ggplot(aes(x = log(sqft_above),  y = log(price))) + geom_point() + geom_smooth(method = "lm") + labs(title = "Price vs Sqft_above")
train_set %>% ggplot(aes(x = log(sqft_lot),   y = log(price))) + geom_point() + labs(title = "Price vs Sqft_lot")
train_set %>% ggplot(aes(x = floors,     y = log(price), group = floors)) + geom_boxplot() + labs(title = "Price vs Floors")
train_set %>% ggplot(aes(x = waterfront, y = log(price), group = waterfront)) + geom_boxplot() + labs(title = "Price vs Waterfront")
train_set %>% ggplot(aes(x = view,       y = log(price), group = view)) + geom_boxplot() + labs(title = "Price vs View")
train_set %>% ggplot(aes(x = condition,  y = log(price), group = condition)) + geom_boxplot() + labs(title = "Price vs Condition")
train_set %>% ggplot(aes(x = log(sqft_basement), y = log(price))) + geom_point() + labs(title = "Price vs Sqft_basement")
train_set %>% ggplot(aes(x = yr_built,   y = log(price), group = yr_built)) + geom_boxplot() + labs(title = "Price vs year_built") 
train_set %>% filter(yr_renovated != 0) %>% 
    ggplot(aes(x = yr_renovated, y = log(price), group = yr_renovated)) + geom_boxplot() + labs(title = "Price vs year_renovated")
train_set %>% ggplot(aes(x = date,     y = log(price), group = date)) + geom_boxplot() + coord_flip() + labs(title = "Date vs Price")
train_set %>% ggplot(aes(x = statezip, y = log(price), group = statezip)) + geom_boxplot() + coord_flip() + labs(title = "Statezip vs Price")

# select features
train_set <- train_set %>% select(price, bedrooms, bathrooms, sqft_living, sqft_above, city)

# define the RMSE as this function
RMSE <- function(predicted_prices, true_prices){
    sqrt(mean((predicted_prices - true_prices)^2))
    }

# correlation in trainset
cor(train_set[1:5])

# log transform to price and predictor variables
train_set <- train_set %>% mutate(price = log(price))
train_set <- train_set %>% mutate(sqft_living = log(sqft_living))
train_set <- train_set %>% mutate(sqft_above = log(sqft_above))
test_set <- test_set %>% mutate(price = log(price))
test_set <- test_set %>% mutate(sqft_living = log(sqft_living))
test_set <- test_set %>% mutate(sqft_above = log(sqft_above))


# multiple linear regression ----------------------------------------------

# multiple linear regression
fit_1 <- train_set %>% lm(price ~ sqft_living, data =.)
fit_2 <- train_set %>% lm(price ~ sqft_living + bathrooms, data =.)
fit_3 <- train_set %>% lm(price ~ sqft_living + bedrooms, data =.)
fit_4 <- train_set %>% lm(price ~ sqft_living + sqft_above + bathrooms, data =.)

# look each models
summary(fit_1)
summary(fit_2)
summary(fit_3)
summary(fit_4)


# predict the price
model_1_prices <- predict(fit_1, test_set)
model_2_prices <- predict(fit_2, test_set)

# make density plot
m1 = tibble(price = exp(model_1_prices), model = "model1")
m2 = tibble(price = exp(model_2_prices), model = "model2")
tp = tibble(price = exp(test_set$price), model = "test")
models_tibble <- bind_rows(m1,m2,tp)

models_tibble %>% ggplot(aes(price, color = model, fill = model)) +
    geom_density(alpha = 0.3, aes(y = ..scaled..)) +
    scale_x_continuous(labels = scales::dollar, breaks = seq(0,3000000,500000)) +
    labs(y = "", title = "Density plot for model1, model2 and test prices",
         color = "", fill = "") 

# calculate RMSE
m1_rmse <- RMSE(exp(model_1_prices), exp(test_set$price))
m2_rmse <- RMSE(exp(model_2_prices), exp(test_set$price))
m1_rmse
m2_rmse


# random forests ----------------------------------------------------------

# recreate the trainset and testset
# because the dataset we used are log transformed, random forests doesn't need it. 
set.seed(1)
test_index <- createDataPartition(y = d$price, times = 1, p = 0.1, list = FALSE)
train_set <- d %>% slice(-test_index)
test_set <- d %>% slice(test_index)

# select features
train_set <- train_set %>% 
    select(price, bedrooms, bathrooms, sqft_living, sqft_above, city)

# find best tuning parameter mtry
tuneRF(train_set[,-1], train_set[,1], doBest = TRUE)

# fit random forests
fit_rf <- randomForest(price ~ ., data = train_set, mtry = 1)
rf_model_prices <- predict(fit_rf, test_set)

plot(fit_rf)
varImpPlot(fit_rf, main = "Variable importance plot")

# density plot rf_model
rf = tibble(price = rf_model_prices, model = "rf_model")
tp = tibble(price = test_set$price , model = "test")
models_tibble <- bind_rows(rf,tp)

models_tibble %>% ggplot(aes(price, color = model, fill = model)) +
    geom_density(alpha = 0.3, aes(y = ..scaled..)) +
    scale_x_continuous(labels = scales::dollar, breaks = seq(0,3000000,500000)) +
    labs(y = "", title = "Density plot for random forests model and test prices",
         color = "", fill = "") 

# calculate RMSE
rf_rmse <- RMSE(rf_model_prices, test_set$price)
rf_rmse



# result ------------------------------------------------------------------

tibble(model = c("linear model1", "linear model2", "random forests model"),
       RMSE = c(m1_rmse, m2_rmse, rf_rmse)) %>% 
    knitr::kable()
