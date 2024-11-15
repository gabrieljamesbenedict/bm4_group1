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
#supervised_model_filename = joblib.load('assests/models/sales_prediction_model.joblib')
#unsupervised_model_filename = joblib.load('assests/models/clustering_model.joblib')
#rf_classifier_filename = joblib.load('assests/models/random_forest_classifier.joblib')

## not sure with the features and the list for the mean time
## features = ['Age', 'Gender_Female', 'Gender_Male', 'AvgTransactionValue', 'ProductCategory_Encoded']
## species_list = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
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

  The Retail Sales Dataset is a synthetic dataset that simulates a realistic retail environment, which enables a deep dive into sales patterns and customer profiles. The dataset was originally made using the Numpy library, published in Kaggle, and owned by Mohammad Talib under a Public Domain license. It provides a realistic exploration of demographics, customer purchasing behaviors, and analysis of sales trends. The dataset portrays a fictional retail environment, focusing on customer interactions and retail operations. Attributes of the dataset includes:  Transaction ID, Date, Customer ID, Gender, Age, Product Category, Quantity, Price per Unit, and Total Amount.
    
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

    # Your content for the MACHINE LEARNING page goes here

# Prediction Page
elif st.session_state.page_selection == "prediction":
    st.header("👀 Prediction")

    # Your content for the PREDICTION page goes here

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