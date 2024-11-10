#######################
# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px

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
dataset = pd.read_csv("data/retail_sales_dataset.csv")

#######################

# Pages

# About Page
if st.session_state.page_selection == "about":
    st.header (" 𝓲 About")
    st.markdown(""" 

    The retails sales dataset focuses on simulating a dynamic retail environment such that it will have an in-depth analysis of customers behavior and its essential attributes such as customer’s interaction and retail operations. Data-Driven Retail Insights: Leveraging Machine Learning for Customer Understanding and Business Optimization

    #### Pages
    1. `Dataset` - Contains the description of the dataset used in this project.
    2. `EDA` - Exploratory Data Analysis of the Retails dataset. Highlighting the three product category (e.g., Beauty, Clothing, Electronics) for targeted marketing campaigns and customer segmentation. Includes graphs such as 
    3. `Data Cleaning / Pre-processing` - Data cleaning and pre-processing steps such as encoding the species column and splitting the dataset into training and testing sets. Includes graphs such as Pie Chart, Bar Chart, Scatter Plot, Box Plot, Bar Plot, Histogram, Heatmap, Violin Plot, and Line Plot. 
    4. `Machine Learning` - Training one supervised model, the Random Forest Regressor, and training one unsupervised model, the KMeans clustering. This includes Supervised and Unsupervised Model evaluation, feature importance and segmentation.
    5. `Prediction` - It predicts on the Total Amount based on other features, Product Category based on customer's demographic and their transaction details, and classifying customers by using customer segmentation using the two trained models.
    6. `Conclusion` - Summary of the insights and observations from the EDA and model training.

    """)
# Dataset Page
elif st.session_state.page_selection == "dataset":
    st.header("📊 Dataset")

    st.write("Retails Dataset")
    st.write("")

    # Your content for your DATASET page goes here

# EDA Page
elif st.session_state.page_selection == "eda":
    st.header("📈 Exploratory Data Analysis (EDA)")


    col = st.columns((1.5, 4.5, 2), gap='medium')

    # Your content for the EDA page goes here

    with col[0]:
        st.markdown('#### Graphs Column 1')


    with col[1]:
        st.markdown('#### Graphs Column 2')
        
    with col[2]:
        st.markdown('#### Graphs Column 3')

# Data Cleaning Page
elif st.session_state.page_selection == "data_cleaning":
    st.header("🧼 Data Cleaning and Data Pre-processing")

    # Your content for the DATA CLEANING / PREPROCESSING page goes here

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