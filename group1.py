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
    2. `EDA` - Exploratory Data Analysis of the Retails dataset. Highlighting the three product category (e.g., Beauty, Clothing, Electronics) for targeted marketing campaigns and customer segmentation.
    3. `Data Cleaning / Pre-processing` - Data cleaning and pre-processing steps such as encoding the species column and splitting the dataset into training and testing sets. Includes graphs such as Pie Chart, Bar Chart, Scatter Plot, Box Plot, Bar Plot, Histogram, Heatmap, Violin Plot, and Line Plot. 
    4. `Machine Learning` - Training one supervised model, the Random Forest Regressor, and training one unsupervised model, the KMeans clustering. This includes Supervised and Unsupervised Model evaluation, feature importance and segmentation.
    5. `Prediction` - It predicts on the Total Amount based on other features, Product Category based on customer's demographic and their transaction details, and classifying customers by using customer segmentation using the two trained models.
    6. `Conclusion` - Summary of the insights and observations from the EDA and model training.

    """)

    ## might change the description of the about page - jelay

# Dataset Page
elif st.session_state.page_selection == "dataset":
    st.header("📊 Dataset")

    st.write("Retails Dataset")
    st.markdown("""

  The retails sales dataset focuses on simulating a dynamic retail environment such that it will have an in-depth analysis of customers behavior and its essential attributes such as customer’s interaction and retail operations. Data-Driven Retail Insights: Leveraging Machine Learning for Customer Understanding and Business Optimization

    **Content**  
    The dataset has **1000** rows containing **9 columns** regarding the customer's demographic and the Product Category (e.g., Beauty, Clothing and Electronics)
    `Link:` https://www.kaggle.com/datasets/mohammadtalib786/retail-sales-dataset/data?fbclid=IwZXh0bgNhZW0CMTEAAR1-Nw3xogR5b50rX6nWWaoTN8jNz8XbwIQIrywInvMUSOgI3dF7LvLyJpY_aem_pQqSow3d9e9SsGKjQuz6Fw           
                
    """)

    col_retail = st.columns((3, 3, 3), gap='medium')

    # Define the new dimensions (width, height)
    resize_dimensions = (500, 300)  # Example dimensions, adjust as needed

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
    ## include the description - jelay



# EDA Page
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

    bibibibi
         
    """)

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