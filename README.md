# Customer-Churn-Analysis
To predict whether a customer will leave (churn) or stay based on their behavior, and help companies take actions to reduce customer loss.
Dataset Information
Dataset: Telecom Customer Churn
Total records: 7043 customers
Target variable: Churn (Yes / No)
Important Features Used:
Tenure (how long customer stayed)
Monthly Charges
Total Charges
Contract Type
Payment Method
Tools & Technologies Used
•	Programming
                  Python
•	Libraries
                  pandas → data handling
                  numpy → numerical operations
                  scikit-learn → machine learning
                  pickle → model saving
                  streamlit → web app
 Machine Learning Model
Algorithm: Logistic Regression
Used with:
StandardScaler → for feature scaling
Pipeline → for better model structure
Project Workflow
•	Data Preprocessing
Removed unnecessary columns (customerID)
Converted TotalCharges to numeric
Handled missing values
Converted categorical data using one-hot encoding
•	 Model Building
Split features and target
Applied Logistic Regression
Used StandardScaler to improve performance
Trained model on full dataset
•	Model Saving
Saved model using pickle
Saved feature columns (columns.pkl) to fix prediction errors
•	Web App Development
Built an interactive app using Streamlit
Features:
User inputs customer data
Clicks Predict
Gets:
Churn / Not churn result
Probability score
Key Insights (From Analysis)
Customers with low tenure are more likely to churn
Month-to-month contracts have highest churn
High monthly charges increase churn risk
Electronic check users churn more
 Business Use Case
This project helps companies:
   Identify customers likely to leave
   Reduce customer churn
   Increase revenue retention
   Take early action (offers, support)
   Resume Description (Use This)
