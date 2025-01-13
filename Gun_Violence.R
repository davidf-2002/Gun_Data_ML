# ---- Step 1: Load Libraries & Data

# install.packages("dplyr")
# install.packages("lubridate")
# install.packages("tidyr")
# install.packages("caret")

library(dplyr)
library(lubridate)
library(tidyr)
library(caret)

gun_data <- read.csv("C:/Users/david/Documents/Gun_violence.csv", stringsAsFactors = FALSE)


# ---- Step 2: Preprocess Data
# Convert date column to Date object for easier handling
gun_data$date <- dmy(gun_data$date)

# Replace empty strings with NA for better handling of missing values
gun_data[gun_data == ""] <- NA

# Remove unnecessary columns from the data set
gun_data <- gun_data %>%
  select(
    -address, -incident_url, -source_url, -congressional_district,
    -incident_url_fields_missing, -notes, -participant_name, -incident_id,
    -participant_relationship, -state_house_district, -state_senate_district,
    -sources, -latitude, -longitude, -location_description, -date, -city_or_county,
    -incident_characteristics, -participant_age, -participant_age_group, 
    -participant_status, -participant_gender, -participant_type, -state
  )

# Summarizing the number of NAs per column
colSums(is.na(gun_data))

# Remove rows with NA in critical columns (n_guns_involved, gun_stolen)
gun_data <- gun_data %>% drop_na(n_guns_involved, gun_stolen)


# ---- Step 3: Encode Categorical Variables 

# Define the gun type mapping according to the categories
gun_type_map <- list(
  Unknown_type = c("Unknown", ""),
  Pistol = c("Handgun", "9mm", "45 Auto", "40 SW", "44 Mag", "38 Spl", "380 Auto", "32 Auto", "357 Mag", "25 Auto", "10mm"),
  Shotgun = c("Shotgun", "12 gauge", "410 gauge", "16 gauge", "20 gauge", "28 gauge"),
  Rifle = c("22 LR", "223 Rem [AR-15]", "7.62 [AK-47]", "308 win", "Rifle", "30-30 Win", "30-06 Spr", "300 Win"),
  Other = c("Other")
)

# Clean and split the gun type data
gun_data$gun_type_cleaned <- gsub("[0-9]+::", "", gun_data$gun_type)
gun_data$gun_type_cleaned <- gsub("\\|", "||", gun_data$gun_type_cleaned)
gun_data <- gun_data[!grepl("^[0-9]+:", gun_data$gun_type_cleaned), ]
gun_data$gun_type_split <- strsplit(as.character(gun_data$gun_type_cleaned), "\\|\\|")

# Categorize gun types
gun_data$gun_type_category <- lapply(gun_data$gun_type_split, function(types) {
  categories <- unlist(lapply(types, function(type) {
    sapply(names(gun_type_map), function(category) if(type %in% gun_type_map[[category]]) category else NULL)
  }))
  unique(categories[!is.null(categories)])
})

# Create a binary matrix for the broader gun type categories
gun_data_categories <- do.call(rbind, lapply(gun_data$gun_type_category, function(categories) {
  as.integer(names(gun_type_map) %in% categories)
}))
colnames(gun_data_categories) <- names(gun_type_map)
gun_data_categories <- as.data.frame(gun_data_categories)


# Predefine the categories for gun_stolen status
stolen_categories <- c("Unknown", "Stolen", "Not-stolen")

# Process 'gun_stolen' by cleaning up the data
gun_data$gun_stolen_cleaned <- gsub("[0-9]+::", "", gun_data$gun_stolen)
gun_data$gun_stolen_cleaned <- gsub("\\|", "||", gun_data$gun_stolen_cleaned)
gun_data <- gun_data[!grepl("^[0-9]+:", gun_data$gun_stolen_cleaned), ]

# Replace empty strings with "Unknown" and split
gun_data$gun_stolen_cleaned <- ifelse(gun_data$gun_stolen_cleaned == "", "Unknown", gun_data$gun_stolen_cleaned)
gun_data$gun_stolen_split <- strsplit(as.character(gun_data$gun_stolen_cleaned), "\\|\\|")

# Create binary matrix data frame for gun stolen statuses
binary_df_gun_stolen <- data.frame(sapply(stolen_categories, function(category) {
  sapply(gun_data$gun_stolen_split, function(x) if(category %in% x) 1 else 0)
}))
colnames(binary_df_gun_stolen) <- ifelse(names(binary_df_gun_stolen) == "Unknown", "Unknown_status", names(binary_df_gun_stolen))

# Combine binary matrices from both 'gun_type' and 'gun_stolen' & remove original columns
gun_data_final <- cbind(gun_data, binary_df_gun_stolen, gun_data_categories)
gun_data_final <- gun_data_final[, !(colnames(gun_data_final) %in% c("gun_type", "gun_type_cleaned", "gun_type_split", "gun_stolen", "gun_stolen_cleaned", "gun_stolen_split", "gun_type_category"))]

print(gun_data_final)


# ---- Step 4: Feature Engineering
# Create n_casualties variable
gun_data_final <- mutate(gun_data_final, n_casualties = n_killed + n_injured)
gun_data_final <- mutate(gun_data_final %>% relocate(n_casualties, .after = n_injured))
gun_data_final <- select(gun_data_final, -n_killed, -n_injured)

# Create severity column based on n_casualties
gun_data_final$severity <- cut(
  gun_data_final$n_casualties,
  breaks = c(-1, 0, 1, Inf),
  labels = c(0, 1, 2),
  right = TRUE
)

gun_data_final$severity <- as.numeric(as.character(gun_data_final$severity))


# Remove n_casualties as this correlates to severity
gun_data_final <- select(gun_data_final, -n_casualties)


# ---- Step 5: Split Data

train_indices <- createDataPartition(gun_data_final$severity, p = 0.8, list = FALSE)
train_data <- gun_data_final[train_indices, ]
test_data <- gun_data_final[-train_indices, ]

# View summary to verify the new columns
summary(gun_data_final)





#------ XGBoost

#install.packages("xgboost")
library(xgboost)

# Calculate class frequencies
class_counts <- table(train_data$severity)
# Calculate class proportions
class_proportions <- class_counts / sum(class_counts)
# Calculate inverse proportions as weights
class_weights <- 1 / class_proportions
# Normalise weights so that the minimum weight is 1 (optional, for better scale handling)
class_weights <- class_weights / min(class_weights)
# Assign weights to each instance based on its class
train_weights <- sapply(train_data$severity, function(x) class_weights[as.character(x)])

dtrain <- xgb.DMatrix(data = as.matrix(train_data[-ncol(train_data)]), 
                      label = train_data$severity,
                      weight = train_weights
)

dtest <- xgb.DMatrix(data = as.matrix(test_data[-ncol(test_data)]), label = test_data$severity)

params <- list(
  objective = "multi:softmax",  # Use 'multi:softprob' for probability output
  num_class = length(unique(train_data$severity)),
  eta = 0.1,
  gamma = 0.1,
  max_depth = 8,
  min_child_weight = 0.5,
  subsample = 0.8,
  colsample_bytree = 0.8
)

model <- xgb.train(params = params, 
                   data = dtrain, 
                   nrounds = 100, 
                   watchlist = list(train = dtrain, test = dtest), 
                   early_stopping_rounds = 10)



pred <- predict(model, dtest)

# Ensure labels are factors for confusion matrix
predicted_labels_factor <- factor(pred, levels = c(0, 1, 2))
actual_labels_factor <- factor(test_data$severity, levels = c(0, 1, 2))

# Check distribution of predicted and actual labels
table(predicted_labels_factor)
table(actual_labels_factor)

confusionMatrix(predicted_labels_factor, actual_labels_factor)

# See which variables are the most important when determining the severity
importance_matrix <- xgb.importance(feature_names = colnames(train_data[-ncol(train_data)]), model = model)
print(importance_matrix)
xgb.plot.importance(importance_matrix)