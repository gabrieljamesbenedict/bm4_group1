#######################
# Importing libraries
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import pandas as pd
import seaborn as sns
import altair as alt
from wordcloud import WordCloud
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Importing Models
import joblib

# Images
from PIL import Image

#######################
# Page configuration
st.set_page_config(
    page_title="Retails Dataset", # Replace this with your Project's Title
    page_icon="assets/icon.png", # You may replace this with a custom icon or emoji related to your project
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

#######################

# Initialize page_selection in session state if not already set
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'about'  # Default page

# Function to update page_selection
def set_page_selection(page):
    st.session_state.page_selection = page

# Sidebar
with st.sidebar:

    # Sidebar Title (Change this with your project's title)
    st.title('Retails Dataset')

    # Page Button Navigation
    st.subheader("Pages")

    if st.button("About", use_container_width=True, on_click=set_page_selection, args=('about',)):
        st.session_state.page_selection = 'about'
    
    if st.button("Dataset", use_container_width=True, on_click=set_page_selection, args=('dataset',)):
        st.session_state.page_selection = 'dataset'

    if st.button("EDA", use_container_width=True, on_click=set_page_selection, args=('eda',)):
        st.session_state.page_selection = "eda"

    if st.button("Data Cleaning / Pre-processing", use_container_width=True, on_click=set_page_selection, args=('data_cleaning',)):
        st.session_state.page_selection = "data_cleaning"

    if st.button("Machine Learning", use_container_width=True, on_click=set_page_selection, args=('machine_learning',)): 
        st.session_state.page_selection = "machine_learning"

    if st.button("Prediction", use_container_width=True, on_click=set_page_selection, args=('prediction',)): 
        st.session_state.page_selection = "prediction"

    if st.button("Conclusion", use_container_width=True, on_click=set_page_selection, args=('conclusion',)):
        st.session_state.page_selection = "conclusion"

    # Project Members
    st.subheader("Members")
    st.markdown("1. Jelyka Dizon\n2. Erika Mariano\n3. Gabriel Loslos\n4. Sophia Olandag\n5. Thomas Kaden Zarta")

#######################
# Data

# Load data
df = pd.read_csv("data/retail_sales_dataset.csv")

#######################


######################
# Plots 
def pie_chart(column, width, height, key):

    # Generate a pie chart
    pie_chart = px.pie(df, names=df[column].unique(), values=df[column].value_counts().values)

    # Adjust the height and width
    pie_chart.update_layout(
        width=width,  # Set the width
        height=height  # Set the height
    )
    st.plotly_chart(pie_chart, use_container_width=True,  key=f"pie_chart_{key}")

def bar_plot(df, column, width, height, key):
    # Generate value counts as a DataFrame for plotting
    data = df[column].value_counts().reset_index()
    data.columns = [column, 'count']  # Rename for easy plotting

    # Create a bar plot using Plotly Express
    bar_plot = px.bar(data, x=column, y='count')

    # Update layout
    bar_plot.update_layout(
        width=width,
        height=height
    )

    # Display the plot in Streamlit
    st.plotly_chart(bar_plot, use_container_width=True, key=f"countplot_{key}")


######################


######################
# Importing models
model = joblib.load('assets/models/sales_prediction_model.joblib')
kmeans = joblib.load('assets/models/clustering_model.joblib')
rf_classifier = joblib.load('assets/models/random_forest_classifier.joblib')

features = ['Age', 'Gender', 'Date', 'Customer ID', 'Product Category', 'Quantity', 'Price per Unit', 'Total amount']
product_category = ['Beauty', 'Electronic', 'Clothing']

######################

# Pages

# About Page
if st.session_state.page_selection == "about":
   st.header (" 𝓲 About")
   st.markdown(""" 

    The Retail Sales Dataset is a synthetic dataset that simulates a realistic retail environment, which enables a deep dive into sales patterns and customer profiles. The dataset was originally made using the Numpy library, published in Kaggle, and owned by Mohammad Talib under a Public Domain license. 
                
    #### Pages
    1. `Dataset` - Contains the description of the dataset used in this project.
    2. `EDA` - Exploratory Data Analysis of the Retails dataset. Highlighting the three product categories (e.g., Beauty, Clothing, Electronics) for targeted marketing campaigns, customer’s behavior, and customer’s segmentation .
    3. `Data Cleaning / Pre-processing` - Data cleaning and pre-processing steps such as encoding the species column and splitting the dataset into training and testing sets. Includes graphs such as Pie Chart, Bar Chart, Scatter Plot, Box Plot, Bar Plot, Histogram, Heatmap, Violin Plot, and Line Plot. 
    4. `Machine Learning` - There are a total of 5 models, which focuses on supervised and unsupervised models. Whereas, supervised model trains the Random Forest Regressor, and Linear Regression. While an unsupervised model uses KMeans Clustering.
    5. `Prediction` - It predicts on the Total Amount based on other features, Product Category based on customer's demographic and their transaction details, and classifying customers by using customer segmentation using the two trained models.
    6. `Conclusion` - Summary of the insights and observations from the EDA and model training.

    """)

    ## might change the description of the about page - jelay

################################################################
#                         Dataset Page
################################################################

elif st.session_state.page_selection == "dataset":
    st.header("📊 Dataset")

    st.write("Retail Sales Dataset")
    st.markdown("""

  The Retail Sales Dataset provides a realistic exploration of demographics, customer purchasing behaviors, and analysis of sales trends. The dataset portrays a fictional retail environment, focusing on customer interactions and retail operations. Attributes of the dataset includes:  Transaction ID, Date, Customer ID, Gender, Age, Product Category, Quantity, Price per Unit, and Total Amount.
    
    **Content**  
    The dataset has **1000** rows containing **9 columns** regarding the customer's demographic and the Product Category (e.g., Beauty, Clothing and Electronics)
    `Link:` https://www.kaggle.com/datasets/mohammadtalib786/retail-sales-dataset/data?fbclid=IwZXh0bgNhZW0CMTEAAR1-Nw3xogR5b50rX6nWWaoTN8jNz8XbwIQIrywInvMUSOgI3dF7LvLyJpY_aem_pQqSow3d9e9SsGKjQuz6Fw           
                
    """)

    col_retail = st.columns((3, 3, 3), gap='medium')

    # dimensions wxh
    resize_dimensions = (500, 300)  

    with col_retail[0]:
        beauty_image = Image.open('assets/retails_picture/Beauty.jpg')
        beauty_image = beauty_image.resize(resize_dimensions)
        st.image(beauty_image, caption='Beauty Retail Store')

    with col_retail[1]:
        clothing_image = Image.open('assets/retails_picture/Clothing.jpg')
        clothing_image = clothing_image.resize(resize_dimensions)
        st.image(clothing_image, caption='Clothing Retail Store')

    with col_retail[2]:

        electronics_image = Image.open('assets/retails_picture/Electronics.jpg')
        virginica_image = electronics_image.resize(resize_dimensions)
        st.image(electronics_image, caption='Electronic Retail Store')
        
    # Display the dataset
    st.subheader("Dataset displayed as a Data Frame")
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Describe Statistics
    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe(), use_container_width=True)
    ## description
    st.markdown("""

The results from the descriptive statistics highlight key insights about the dataset. Firstly, the average age is 41.39 years with a standard deviation of 13.68, indicating moderate variation around the mean. The quantity of items purchased averages 2.51 with a standard deviation of 1.13, suggesting that the number of items bought is generally consistent. Moving on to price per unit, the mean is 179.89 with a standard deviation of 189.68, indicating significant variability in pricing. The total amount spent averages 456 with a high standard deviation of 559.99, reflecting substantial variation in transaction totals.

In terms of range, ages span from 18 to 64 years, while the quantity of items ranges from 1 to 4. The price per unit varies widely from 25 to 500, and total amounts range from 25 to 2000. This wide spread in both pricing and total amounts suggests diverse purchasing behaviors among customers.

The 25th, 50th, and 75th percentiles show gradual increases across all metrics, highlighting variability and potential trends within the data, useful for further analysis and understanding customer purchasing patterns.

    """)

################################################################
#                         EDA
################################################################

elif st.session_state.page_selection == "eda":
    st.header("📈 Exploratory Data Analysis (EDA)")


    col = st.columns((3, 3, 3), gap='medium')

    # Your content for the EDA page goes here

    with col[0]:
           with st.expander('Legend', expanded=True):
            st.write('''
                - Data: [Retails Dataset](https://www.kaggle.com/datasets/arshid/iris-flower-dataset).
                - :violet[**Pie Chart**]: Distribution of the Product Category.
                - :violet[**Bar Plot**]: Customer's Demographic mainly Gender and Age
                ''')
        
    st.markdown('#### Product Categroy Distribution')
    pie_chart("Product Category", 500, 350, 1)
            
    with col[1]:
        st.markdown('#### Gender Distribution')
        bar_plot(df, "Gender", 500, 300, 2)

        
    with col[2]:
        st.markdown('#### Age Distribution')
        bar_plot(df, "Age", 500, 300, 3)

# Data Cleaning Page
elif st.session_state.page_selection == "data_cleaning":
    st.header("🧼 Data Cleaning and Data Pre-processing")

    st.dataframe(df.head(), use_container_width=True, hide_index=True)

    st.markdown("""

    Description
         
    """)

    st.subheader ("Data Cleaning")
    
    # Checking for missing values
    missing_values = df.isnull().sum()

    # Display the missing values in Streamlit
    st.write("Displaying missing values in the dataset:")
    st.dataframe(missing_values, use_container_width=True, hide_index=True)

    st.markdown("""

    Description
         
    """)


    st.subheader("Data Pre-Processing")

    encoder = LabelEncoder()

    df['product_category_encoded'] = encoder.fit_transform(df['Product Category'])

    st.dataframe(df.head(), use_container_width=True, hide_index=True)

    st.markdown("""

    [Description]
         
    """)


    # Predicting the “Total Amount” that is based on the other features
    xdata = df[['product_category_encoded', 'Quantity', 'Price per Unit']]
    ydata = df['Total Amount']

    st.dataframe(df.head(), use_container_width=True, hide_index=True)

    st.markdown("""

    [Description]
         
    """)

    ######################################################################################

    # Classifying Customer segments, using “Age”, “Gender”, “Total Amount”, and “Product Category”, for targeted marketing campaigns
    # Default segment
    df['Customer Segment'] = 'Regular'

    # 1. Average Transaction Value
    df['AvgTransactionValue'] = df['Total Amount'] / df.groupby('Customer ID')['Transaction ID'].transform('nunique')

    # Encoding the Product Category
    encoder = LabelEncoder()
    df['product_category_encoded'] = encoder.fit_transform(df['Product Category'])

    # Adding one-hot encoding for Product Category
    product_category_dummies = pd.get_dummies(df['Product Category'], prefix='ProductCategory')
    df = pd.concat([df, product_category_dummies], axis=1)

    # Adding one-hot encoding for Gender, if it exists
    if 'Gender' in df.columns:
        gender_dummies = pd.get_dummies(df['Gender'], prefix='Gender')
        df = pd.concat([df, gender_dummies], axis=1)

    # Rule 1: High-Value Customers (using AvgTransactionValue)
    df.loc[df['AvgTransactionValue'] > 150, 'Customer Segment'] = 'High Value'  # Adjusted threshold

    # Rule 2: Loyal Customers (based on the number of unique transactions)
    loyal_customers = df.groupby('Customer ID')['Transaction ID'].nunique()
    loyal_customer_ids = loyal_customers[loyal_customers > 8].index
    df.loc[df['Customer ID'].isin(loyal_customer_ids), 'Customer Segment'] = 'Loyal'

    # Mapping Product Category and their encoded equivalent 
    
    unique_product = df['Product Category'].unique()
    unique_product_encoded = df['product_category_encoded'].unique()

    # Displaying the Dataframe
    product_mapping_df = pd.DataFrame({'Product Category': unique_product, 'product_category_encoded': unique_product_encoded})

    st.dataframe(product_mapping_df, use_container_width=True, hide_index=True)


    # Select features and target variable
    features = ['Age', 'Gender_Female', 'Gender_Male', 'AvgTransactionValue',
            'product_category_encoded'] + list(product_category_dummies.columns)

    X = df[features]
    y = df['Customer Segment']

    st.code("""

    # Select features and target variable
    features = ['Age', 'Gender_Female', 'Gender_Male', 'AvgTransactionValue',
            'product_category_encoded'] + list(product_category_dummies.columns)
            # Include one-hot encoded categories

    X = df[features]
    y = df['Customer Segment']
    """)

    st.markdown("""

    [Description]
         
    """)
    ######################################################################################

    # Predicting the “Product Category” that is based on the customer’s demographic and their transaction details
    # Handle missing values by forward filling
    df.ffill(inplace=True)  # Forward fill for missing values

    # Convert 'Date' to ordinal format
    df['Date'] = pd.to_datetime(df['Date']).apply(lambda date: date.toordinal())

    # Split data into features (X) and target variable (y)
    X = df.drop(['Product Category', 'Customer ID', 'Transaction ID', 'Gender', 'Customer Segment'], axis=1, errors='ignore')
    y = df['Product Category']

    st.code("""
     # Split data into features (X) and target variable (y)
    X = df.drop(['Product Category', 'Customer ID', 'Transaction ID', 'Gender', 'Customer Segment'], axis=1, errors='ignore')
    y = df['Product Category']
    """)

    st.markdown("""

    [Description]
         
    """)

    # Identify numerical features for scaling
    numerical_features = X.select_dtypes(include=['number']).columns
    scaler = StandardScaler()
    X[numerical_features] = scaler.fit_transform(X[numerical_features])

    # Display the DataFrame in Streamlit
    st.dataframe(df.head(), use_container_width=True, hide_index=True)

    ######################################################################################

    # To use the 'Date' feature to predict sales patterns over time
    # Convert 'Date' to datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Calculate total sales for each entry
    df['Total Sales'] = df['Quantity'] * df['Price per Unit']

    # Group by 'Date' to get daily total sales
    daily_sales = df.groupby('Date')['Total Sales'].sum().reset_index()

    # Display daily sales in Streamlit
    st.write("Daily Total Sales:")
    st.dataframe(daily_sales, use_container_width=True, hide_index=True)

    # Add the daily sales back to the main DataFrame if needed
    df = df.merge(daily_sales, on='Date', suffixes=('', '_Daily'))

    # Target variable: Future Sales (shifted 'Total Sales' on daily basis)
    daily_sales['Future_Sales'] = daily_sales['Total Sales'].shift(-1)
    daily_sales.dropna(inplace=True)  # Drop last row with no future sales value

    # Display updated DataFrame with 'Future_Sales'
    st.write("Daily Sales with 'Future_Sales' column added:")
    st.dataframe(daily_sales, use_container_width=True, hide_index=True)

    # Prepare for modeling by separating features (X) and target variable (y)
    X = daily_sales[['Date', 'Total Sales']]
    y = daily_sales['Future_Sales']

    # Convert 'Date' to ordinal for modeling
    X['Date'] = X['Date'].apply(lambda date: date.toordinal())

    st.write("Features (X):")
    st.dataframe(X, use_container_width=True, hide_index=True)
    st.write("Target variable (y): Future Sales")
    st.dataframe(y, use_container_width=True, hide_index=True)

    ######################################################################################
    # In order to segment customers according to their demographics and purchase patterns. Uses “Age”, “Total Amount spent”, “Frequency of purchases”, and “Product Category” preferences.
    # Feature engineering: Calculate 'Total Sales' for each entry    
    df['Total Sales'] = df['Quantity'] * df['Price per Unit']

    # Encode 'Gender' using LabelEncoder
    label_encoder = LabelEncoder()
    df['Gender'] = label_encoder.fit_transform(df['Gender'])  # Male=0, Female=1

    # Select features for segmentation: demographics and purchase patterns
    features = df[['Age', 'Gender', 'Quantity', 'Total Sales']]

    # Standardize features using numpy (alternative to StandardScaler)
    scaled_features = (features - features.mean()) / features.std()

    # Display the processed DataFrame in Streamlit
    st.write("Data after Feature Engineering and Encoding:")
    st.dataframe(df)

    # Finding the optimal number of clusters using the Elbow Method
    sse = []
    K_range = range(1, 11)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_features)
        sse.append(kmeans.inertia_)



    st.markdown ("""
    [Description]
    """)

    # Identify high-, medium-, and low-spending customer clusters based on their age and money spent.
    # Aggregate 'Total Amount' by 'Customer ID' to find overall spending per customer
    customer_data = df.groupby('Customer ID').agg({
        'Age': 'first',  # Assuming age remains constant for each customer
        'Total Amount': 'sum'  # Total amount spent per customer
    }).reset_index()

    # Rename 'Total Amount' to 'CLV'
    customer_data.rename(columns={'Total Amount': 'CLV'}, inplace=True)

    # Scale features
    scaler = StandardScaler()
    customer_data[['Age', 'CLV']] = scaler.fit_transform(customer_data[['Age', 'CLV']])

    # Perform KMeans clustering to segment customers into high, medium, and low-value groups
    kmeans = KMeans(n_clusters=3, random_state=42)
    customer_data['Cluster'] = kmeans.fit_predict(customer_data[['Age', 'CLV']])

    # Map cluster labels to high, medium, and low CLV based on cluster centroids
    cluster_centers = kmeans.cluster_centers_
    cluster_labels = ['Low CLV', 'Medium CLV', 'High CLV']
    sorted_clusters = sorted(range(len(cluster_centers)), key=lambda k: cluster_centers[k][1])  # Sort by CLV centroid
    customer_data['CLV Segment'] = customer_data['Cluster'].map({sorted_clusters[0]: 'Low CLV',
                                                                sorted_clusters[1]: 'Medium CLV',
    
                                                                sorted_clusters[2]: 'High CLV'})
    
###############################################################################################



    st.subheader("Train-Test Split")
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # Drop rows with NaN values in y_train and y_test

    st.code("""

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                
    """)

    # Finding the optimal number of clusters using the Elbow Method [Unsupervised Model]


    st.subheader("X_train")
    st.dataframe(X_train, use_container_width=True, hide_index=True)

    st.subheader("X_test")
    st.dataframe(X_test, use_container_width=True, hide_index=True)

    st.subheader("y_train")
    st.dataframe(y_train, use_container_width=True, hide_index=True)

    st.subheader("y_test")
    st.dataframe(y_test, use_container_width=True, hide_index=True)

    st.markdown("[Description]")
    
    st.subheader ("Visualization for Unsupervised Model")

    # Plot the Elbow Method
    st.write("Elbow Method for Optimal K")
    plt.figure(figsize=(8, 6))
    plt.plot(K_range, sse, marker='o')
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Sum of Squared Distances (SSE)")
    plt.title("Elbow Method to Determine Optimal K")
    st.pyplot(plt)

    # Visualize the clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=customer_data, x='Age', y='CLV', hue='CLV Segment', palette='coolwarm', s=100)
    plt.title('Customer Segmentation Based on CLV and Age')
    plt.xlabel('Age (Standardized)')
    plt.ylabel('Customer Lifetime Value (Standardized)')
    plt.legend(title='CLV Segment')
    st.pyplot(plt)



# Machine Learning Page
elif st.session_state.page_selection == "machine_learning":
    st.header("🤖 Machine Learning")

    st.subheader("Model Evaluation")

    st.write(" **Predicting the “Total Amount” that is based on the other features** ")

    st.code("""

    # Train the models
    model = LinearRegression()
    model.fit(xtrain, ytrain)

    # Predict the test set
    ypred = model.predict(xtest)

    # Calculate accuracy of model
    accuracy = model.score(xtest, ytest)
    print("Accuracy:", accuracy)
                
    """)
    st.write("Accuracy: 0.8559361812944614")

    predicted_vs_actual_total_amount = Image.open('assets/graphs [images]/Predicted vs. Actual Total Amount.png')
    st.image(predicted_vs_actual_total_amount, caption='Predicted vs Actual Total Amount')

    st.markdown(
        """
The scatter plot shows how well the model predicts the total amount that clients will spend. The anticipated total amount is displayed on the y-axis, and the actual total amount is represented on the x-axis. To indicate accurate predictions, the points should ideally be in tight alignment with the diagonal line. With an accuracy score of 0.8559, the model appears to be fairly accurate in forecasting consumer spending patterns.

        """
    )

    st.write("### Classifying Customer segments, using “Age”, “Gender”, “Total Amount”, and “Product Category”, for targeted marketing campaigns. ")

    st.code("""

    rf_classifier = RandomForestClassifier(random_state=42)
    rf_classifier.fit(X_train, y_train)

    y_pred = rf_classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
                
    """)

    distribution_of_customer_segment = Image.open('assets/graphs [images]/Distribution of Customer Segments.png')
    st.image(distribution_of_customer_segment, caption='Distribution of Customer Segment')

    st.write("Accuracy: 1.00")


    st.markdown ("""
                 The distribution of clients across four different segments is shown in the bar chart that is provided. With a somewhat greater count than the other segments, Segment 2 seems to be the most populated. While Segment 1 has the fewest clients, Segments 0 and 3 have comparable numbers.

With an output of 1.00, the model appears to have determined that Segment 1 is the most important or pertinent segment for the analysis or prediction task at hand. This might suggest that Segment 1 clients display particular traits or behaviors that are especially relevant to the goal of the model.
                 """)
    
    st.markdown(" ### Predicting the “Product Category” that is based on the customer’s demographic and their transaction details")
 
    st.code("""

    # Make predictions on the test set
    y_pred = rf_classifier.predict(X_test_scaled)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
                    
    """)

    confusion_matrix_product_category = Image.open('assets/graphs [images]/Confusion Matrix for Product Category Prediction.png')
    st.image(confusion_matrix_product_category, caption='Confusion Matrix for Product Category Prediction')

    st.write("Accuracy: 1.0")

    st.markdown ("""
The confusion matrix illustrates how well the model predicts the appropriate product category. Correct predictions are represented by diagonal elements, but wrong classifications are shown by off-diagonal elements. The model is said to accurately predict the product category for every instance in the test set if its accuracy is 1.00. This shows that the model successfully classifies products by using transaction information and customer demographics.
                 """)
    
    st.markdown(" ### To use the 'Date' feature to predict sales patterns over time")
    # Convert 'Date' to datetime format
    df['Date'] = pd.to_datetime(df['Date'])

        # Calculate total sales for each entry
    df['Total Sales'] = df['Quantity'] * df['Price per Unit']

        # Group by 'Date' to get daily total sales
    daily_sales = df.groupby('Date')['Total Sales'].sum().reset_index()

        # Plot daily sales with a rolling average
    plt.figure(figsize=(14, 7))
    plt.plot(daily_sales['Date'], daily_sales['Total Sales'], label='Daily Sales', alpha=0.5)
    plt.plot(
            daily_sales['Date'],
            daily_sales['Total Sales'].rolling(window=7).mean(),
            label='7-Day Rolling Average',
            color='orange',
            linewidth=2,
        )
    plt.xlabel('Date')
    plt.ylabel('Total Sales')
    plt.title('Daily Sales with 7-Day Rolling Average')
    plt.legend()
    st.pyplot(plt)

    
    st.code("""

    # Make predictions on the test set
    y_pred = model.predict(X_test_scaled)

    # Evaluate the model (Mean Squared Error)
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)
                        
    """)

    actual_vs_predicted_image = Image.open('assets/graphs [images]/Actual vs Predicted Sales.png')
    st.image( actual_vs_predicted_image, caption='Actual vs Predicted Sales')

    st.write("Mean Squared Error: 86550.090760611")
    st.markdown("""
The difference between actual and anticipated sales data is shown in the accompanying line plot. The orange line shows the model's anticipated values, and the blue line shows the actual sales data. A considerable disparity between the two is indicated by the high Mean Squared Error (MSE) of 86,550.09, which implies that the model's predictions are not particularly reliable. Numerous factors, including feature engineering, model complexity, data quality, and model selection, may be to blame for this. It is advised to experiment with various models, examine extra features, improve data preparation, and take regularization strategies into consideration in order to enhance the model's performance.
                """)
    
    st.markdown(" ### In order to segment customers according to their demographics and purchase patterns. Uses “Age”, “Total Amount spent”, “Frequency of purchases”, and “Product Category” preferences.")


    st.code("""

    # Display the centroid of each cluster to understand the segment characteristics
    centroids = kmeans.cluster_centers_ * features.std().values + features.mean().values  # Reverse scaling
    centroid_df = pd.DataFrame(centroids, columns=features.columns)
    print("Centroids of each customer segment:\n", centroid_df)
                        
    """)

    customer_segmentation_demographic = Image.open('assets/graphs [images]/Customer Segmentation based on Demographics and Purchase Patterns.png')
    st.image(customer_segmentation_demographic, caption='Customer Segmentation based on Demographics and Purchase Patterns')

    st.write("Centroids of each customer segment:"
          "Age        Gender  Quantity  Total Sales"
"0  29.727723  1.000000e+00  2.217822   234.876238"
"1  38.797386  5.032680e-01  3.673203  1554.248366"
"2  41.764151 -1.498801e-15  2.344340   266.155660"
"3  53.495192  1.000000e+00  2.302885   264.206731")
    st.markdown (""" 
The link between the number of clusters and the associated Sum of Squared Errors (SSE) is depicted in the Elbow Method figure. A graph's **"elbow point"** indicates the ideal amount of clusters, where the pace at which SSE decreases starts to level out. The elbow point in this instance seems to be approximately four segments. Four different client categories have been discovered by the clustering analysis based on the following factors: **age, gender, quantity purchased, and total sales.**

`Segment 1:`  Younger customers with moderate spending.

`Segment 2: ` Middle-aged customers with higher spending.

`Segment 3:`  Older customers with moderate spending.

`Segment 4: ` Older male customers with moderate spending.

Businesses can boost customer satisfaction and sales by customizing their product offerings and marketing strategies to target particular consumer groups by having a thorough grasp of these segments. For instance, **Segment 3** can be supplied expensive products, whereas **Segment 1** might be targeted with promotions for younger consumers.
                """)

    st.markdown(" ### Identify high-, medium-, and low-spending customer clusters based on their age and money spent.")

    st.code("""

    # Map cluster labels to high, medium, and low CLV based on cluster centroids
    cluster_centers = kmeans.cluster_centers_
    cluster_labels = ['Low CLV', 'Medium CLV', 'High CLV']
    sorted_clusters = sorted(range(len(cluster_centers)), key=lambda k: cluster_centers[k][1])  # Sort by CLV centroid
    customer_data['CLV Segment'] = customer_data['Cluster'].map({sorted_clusters[0]: 'Low CLV',
                                                             sorted_clusters[1]: 'Medium CLV',
                                                             sorted_clusters[2]: 'High CLV'})
            
    # Display the resulting clusters
    print("Customer CLV Segmentation Results:")
    print(customer_data[['Customer ID', 'Age', 'CLV', 'CLV Segment']].head())
                        
    """)

    st.write("Customer CLV Segmentation Results:"
  "Customer ID       Age       CLV CLV Segment"
"0     CUST001 -0.534352 -0.550362     Low CLV"
"1     CUST002 -1.118896  0.963495    High CLV"
"2     CUST003  0.634737 -0.764083  Medium CLV"
"3     CUST004 -0.315148  0.072991     Low CLV"
"4     CUST005 -0.826624 -0.639412     Low CLV")
    
    customer_segmentation_clv = Image.open('assets/graphs [images]/Customer Segmentation Based on CLV and Age.png')
    st.image(customer_segmentation_clv, caption='Customer Segmentation Based on CLV and Age.png')
    
    st.markdown ("""
The scatter plot shows how customers are divided into groups according to their age and Customer Lifetime Value (CLV). Three separate clusters—Low CLV, High CLV, and Medium CLV—have been discovered by the model. This segmentation offers important information on consumer purchasing and behavior. Customers in the High CLV category, for example, are typically older and have a history of spending more money. Businesses can optimize customer value and retention by customizing their product offers and marketing methods by comprehending these segments.
                 """)

# Prediction Page
elif st.session_state.page_selection == "prediction":
    st.header("👀 Prediction")

    col_pred = st.columns((1.5, 3, 3), gap='medium')

    # Initialize session state for clearing results
    if 'clear' not in st.session_state:
        st.session_state.clear = False

    with col_pred[0]:
        with st.expander('Options', expanded=True):
            show_dataset = st.checkbox('Show Dataset')
            show_classes = st.checkbox('Show All Classes')
            show_beauty = st.checkbox('Show Beauty')
            show_electronic = st.checkbox('Show Electronic')
            show_clothing = st.checkbox('Show Clothing')

            clear_results = st.button('Clear Results', key='clear_results')

            if clear_results:

                st.session_state.clear = True

    with col_pred[1]:
        st.markdown("#### 🌲 Decision Tree Classifier")
        
        # Input boxes for the features
        rf_Age = st.number_input('Age', min_value=0.0, max_value=10.0, step=0.1, key='dt_Age', value=0.0 if st.session_state.clear else st.session_state.get('dt_Age', 0.0))
        rf_Gender = st.number_input('Sepal Width', min_value=0.0, max_value=10.0, step=0.1, key='dt_Gender', value=0.0 if st.session_state.clear else st.session_state.get('dt_Gender', 0.0))
        #dt_petal_width = st.number_input('Petal Width', min_value=0.0, max_value=10.0, step=0.1, key='dt_petal_width', value=0.0 if st.session_state.clear else st.session_state.get('dt_petal_width', 0.0))
        #dt_petal_length = st.number_input('Petal Length', min_value=0.0, max_value=10.0, step=0.1, key='dt_petal_length', value=0.0 if st.session_state.clear else st.session_state.get('dt_petal_length', 0.0))

        classes_list = ['Beauty', 'Electronic', 'Clothing']
        
        # Button to detect the Iris species
        if st.button('Detect', key='dt_detect'):
            # Prepare the input data for prediction
            rf_input_data = [[rf_Age, rf_Gender]]
            
            # Predict the Iris species
            rf_prediction = rf_classifier.predict(rf_input_data)
            
            # Display the prediction result
            st.markdown(f'The predicted Iris species is: `{classes_list[rf_prediction[0]]}`')

# Conclusions Page
elif st.session_state.page_selection == "conclusion":
    st.header("📝 Conclusion")

    st.markdown("""
The analysis of the graphs allows for the deduction of the following important conclusions 📊:



1.   **Customer Spending Patterns:** Overall spending behavior cannot be explained by a single dominant component, although there are notable differences in purchasing habits depending on gender and product category.

2.   **Product Category Preference:** Beauty items are bought by customers of all ages, but younger audiences are more interested in clothing and electronics.

3.   **Age and Spending:** Spending habits are not greatly influenced by age alone, suggesting that lifestyle, income, or personal preferences are more important considerations.

4.   **Gender and Spending:** Spending habits are somewhat influenced by gender; differences are seen in the cosmetics sector but not in clothing or electronics.


Overall, these results point to a complicated interaction between variables affecting consumer spending patterns. Deeper understanding of these trends may be possible with additional research, such as consumer segmentation or in-depth product pricing study.

                
Moreover, the analysis's supervised and unsupervised models offer insightful information about consumer behavior and industry trends in retails dataset.

**Supervised Model:**
* **Sales Prediction:** Based on the model, it accurately forecasts future sales and businesses are able to estimate demand and make appropriate plans.
* **Customer Segmentation:** By dividing up the client base into discrete groups, the concept makes it possible to run focused advertising campaigns and provide individualized customer service.
* **Product Category Prediction:** The model helps with inventory management and product recommendations by correctly predicting product categories


**Unsupervised Model:**
* **Customer Segmentation:** Targeted marketing and product recommendations are made possible by the clustering analysis, which separates different customer segments according to demographics and purchasing trends.


Businesses can learn a lot about market trends and consumer behavior by using both supervised and unstructured models. This allows them to make data-driven decisions for better business performance, manage inventory, improve customer experience, and personalize marketing.

""")