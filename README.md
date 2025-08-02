# Restaurant-Recommender-System-using-ML-
Description: A recommendation engine that predicts which restaurants customers are most likely to order from, based on customer location, restaurant information, and historical order data.


Predictive Restaurant Recommender System
This project is a predictive recommendation engine designed to identify which restaurants customers are most likely to order from. It was built using customer location data, restaurant information, and historical order history.

Project Methodology
1. Data Preprocessing and Feature Engineering
Data Cleaning: The raw data was analyzed for missing values and inconsistencies. Key steps included:

Handling of missing values in columns like rating and description by either imputation or dropping.

Parsing and standardizing date and time formats from the order history.

Feature Creation: New features were engineered to provide the model with better signals:

Geospatial Features: The Haversine distance between each customer's location (CID, LOC_NUM) and every restaurant (VENDOR_ID) was calculated. This feature is crucial for capturing proximity as a key decision factor.

User Behavioral Features: From the order history, we extracted features like:

Average rating of restaurants a customer has previously ordered from.

Most frequent cuisine type a customer orders.

Recency of a customer's last order to capture their ordering frequency.

Vendor Features: Features related to the restaurants were also created, such as average vendor rating and number of past orders.

Challenges and Solutions
This section is where you can truly stand out by showing your problem-solving skills.

Challenge: The initial dataset was highly imbalanced, with a vast majority of possible customer-vendor combinations not resulting in an order (target = 0). This could lead a model to overpredict the negative class.

Solution: To address this, we employed techniques like undersampling the majority class or using class weights during model training to penalize misclassifications of the minority class more heavily.

Challenge: The vendor feature has a large number of unique categories, which is difficult for some models to handle directly.

Solution: Instead of one-hot encoding every vendor (which would create too many features), we transformed the vendor IDs into features like average_vendor_rating or vendor_order_count to give the model meaningful information without a massive increase in dimensionality.

Model Building and Evaluation
Model Selection: Given the nature of the problem as a binary classification task, we chose to use a Gradient Boosting Classifier (e.g., LightGBM or XGBoost). This model is effective with tabular data and can handle complex, non-linear relationships between features.

Training: The model was trained on the preprocessed and engineered dataset. The model's hyperparameters were tuned to optimize performance on the validation set.

Evaluation: The model's performance was evaluated using metrics like precision, recall, and the F1-score, which are more suitable than simple accuracy for imbalanced datasets.

Final Output
The final predictions for the test data are provided in the required format, showing the likelihood (0 or 1) of an order for each customer-vendor combination.
