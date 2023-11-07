import streamlit as st
st. set_page_config(layout="wide")

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import pandas as pd 
from matplotlib import pyplot as plt
import plotly.express as px
import plotly.io as pio
import numpy as np 
import joblib
import seaborn as sns
from scipy.cluster.hierarchy import cut_tree

pio.templates.default = 'plotly'

# plt.style.use("ggplot")
plt.axis('off')
plt.grid(visible = False)

countries = ['United Kingdom', 'France', 'Australia', 'Netherlands', 'Germany',
       'Norway', 'EIRE', 'Switzerland', 'Spain', 'Poland', 'Portugal',
       'Italy', 'Belgium', 'Lithuania', 'Japan', 'Iceland',
       'Channel Islands', 'Denmark', 'Cyprus', 'Sweden', 'Austria',
       'Israel', 'Finland', 'Greece', 'Singapore', 'Lebanon',
       'United Arab Emirates', 'Saudi Arabia', 'Czech Republic', 'Canada',
       'Unspecified', 'Brazil', 'USA', 'European Community', 'Bahrain',
       'Malta', 'RSA']


data             = pd.read_csv("./CleanedData.csv")

encoder          = joblib.load('./encoder.joblib')

kmeans           = joblib.load("./kmeans.joblib")
mergings         = joblib.load("./mergings.joblib")


kmeans2          = joblib.load("./kmeans2.joblib")
mergings2        = joblib.load("./mergings2.joblib")


scaler           = joblib.load("./scaler.joblib")
pca              = joblib.load("./pca.joblib")

pca_data         = pca.transform(data[['AmountSpent','Frequency','Recency','Country']])

selData          = data[['AmountSpent','Frequency','Recency']]
Cluster_3D       = kmeans.predict(selData.values)
Cluster_3DHeir   = cut_tree(mergings, n_clusters=4).reshape(-1, )

cluster_labels   = cut_tree(mergings, n_clusters=4).reshape(-1, )
cluster_labels2  = cut_tree(mergings, n_clusters=4).reshape(-1, )


pcaPredsDf       = pd.DataFrame(np.c_[pca_data, kmeans.predict(pca_data), cluster_labels], columns=['AmountSpent','Frequency','Recency','Cluster_Labels', "Cluster_Labels_Heir"])
selDataDf        = pd.DataFrame(np.c_[selData, kmeans2.predict(selData), cluster_labels2], columns=['AmountSpent','Frequency','Recency','Cluster_Labels', "Cluster_Labels_Heir"])

df = pcaPredsDf


def predict(data, modelType = "PCA KMEANS"):
    pred = kmeans.predict(data)

    if modelType=="PCA HEIRARCHICAL":
        pred = cut_tree(mergings, n_clusters=4).reshape(-1, )
    return pred
html_temp = """
<h3>
Project Title: The Development of a Customer Segmentation System Based on Purchasing Behavior
</h3>

<h4>
Project Overview:
</h4>

<p>
Welcome to our innovative project aimed at revolutionizing customer segmentation through the power of data science. In this endeavor, we focus on understanding and classifying customers by their purchasing behavior, using advanced techniques such as K-Means and Hierarchical Clustering.

<strong>Why Customer Segmentation Matters:</strong>

Customer segmentation is the cornerstone of personalized marketing and customer satisfaction. By dissecting the diverse purchasing behaviors of your clientele, businesses can tailor their marketing strategies, create targeted promotions, and enhance customer experiences.
</p>
<h4><b>
Our Approach:
</b>
</h4>

<p>
<b>Data</b> Collection: We gather extensive data on customer transactions, preferences, and behaviors to create a rich dataset.

<b>Data Preprocessing</b>: We clean and prepare the data for analysis, ensuring accuracy and reliability.

<b>K-Means Clustering</b>: Employing the K-Means algorithm, we divide customers into distinct segments based on their purchasing habits, allowing businesses to understand and engage with them more effectively.

<b>Hierarchical Clustering</b>: This technique refines our segmentation further, providing a hierarchical view of customer groups for deeper insights.

Key Benefits:
<ul>
<li>Improved Targeting: Precisely target customers with tailored marketing strategies and promotions.</li>
<li>Enhanced Customer Experiences: Understand and address unique needs and preferences.</li>
<li>Optimized Product Development: Develop products that cater to specific customer segments.</li>
<li>Increased ROI: Maximize the effectiveness of your marketing budget by reaching the right audience.</li>
</p>
<h4>
Why Choose Our Solution:
</h4>
Our team of expert data scientists is dedicated to creating a customer segmentation system that aligns with your business goals. We aim to empower your organization with data-driven insights that drive success.
</p>
Get Started:

<i>
Join us on this journey of transforming customer segmentation by harnessing the power of data. Unlock the potential of personalized marketing and exceptional customer experiences with our cutting-edge solutions.
</i>
</p>
"""

def displayGraph(plotType, df, combos, graph):
       if plotType == '2D':

        columns = combos.split('/')

        if graph == "Non-Interactive":

            col1, col2, col3 = st.columns([1,4,1])

            fig = plt.figure(figsize=(20, 12))  # Adjust the figure size here

            plt.subplot(211)
            sns.scatterplot(x=columns[0], y=columns[1], hue='Cluster_Labels', data=df, palette='Set1')
            plt.title(f'{columns[0]} vs. {columns[1]} Normal Kmeans')


            plt.subplot(212)
            sns.scatterplot(x=columns[0], y=columns[1], hue='Cluster_Labels_Heir', data=df, palette='Set1')
            plt.title(f'{columns[0]} vs. {columns[1]} Heirarchical')
            col2.pyplot(fig)


        if graph == "Interactive":
            col1, col2, col3 = st.columns([1,4,1])

            fig = px.scatter(df, x=columns[0], y=columns[1], color='Cluster_Labels',
                title=f'Clustering by {columns[0]} and {columns[1]}',
                labels={columns[0]: columns[0], columns[1]: columns[1], 'Cluster_Labels': 'Cluster'})

            fig.update_layout(
                xaxis=dict(title=columns[0], title_font=dict(size=14)),
                yaxis=dict(title=columns[1], title_font=dict(size=14)),
                width=800,
                height=600
                )
            col2.plotly_chart(fig)

            fig2 = px.scatter(df, x=columns[0], y=columns[1], color='Cluster_Labels_Heir',
                title=f'Clustering by {columns[0]} and {columns[1]}',
                labels={columns[0]: columns[0], columns[1]: columns[1], 'Cluster_Labels_Heir': 'Cluster'})

            fig2.update_layout(
                xaxis=dict(title=columns[0], title_font=dict(size=14)),
                yaxis=dict(title=columns[1], title_font=dict(size=14)),
                width=800,
                height=600
                )
            col2.plotly_chart(fig2)

        if plotType == '3D':
            col1, col2, col3 = st.columns([1,4,1])
            fig = px.scatter_3d(df, x='AmountSpent', y='Frequency', z='Recency', color='Cluster_Labels',
                        labels={'AmountSpent': 'AmountSpent', 'Frequency': 'Frequency', 'Recency': 'Recency', 'Cluster_3D': 'Cluster'})

            fig.update_layout(
                scene=dict(
                    xaxis_title='Amount',
                    yaxis_title='Frequency',
                    zaxis_title='Recency',
                ),
                title='Clustering by Amount, Frequency, and Recency For Normal KMeans',
                width=1000,
                height=800
            )
            
            col2.plotly_chart(fig)

            fig2 = px.scatter_3d(df, x='AmountSpent', y='Frequency', z='Recency', color='Cluster_Labels_Heir',
                        labels={'AmountSpent': 'AmountSpent', 'Frequency': 'Frequency', 'Recency': 'Recency', 'Cluster_3D': 'Cluster'})

            fig2.update_layout(
                scene=dict(
                    xaxis_title='Amount',
                    yaxis_title='Frequency',
                    zaxis_title='Recency',
                ),
                title='Clustering by Amount, Frequency, and Recency For Heirarchical Cluster',
                width=1000,
                height=800
            )
            
            col2.plotly_chart(fig2)

col1, col2, col3 = st.columns([1,4,1])
col2.title("CUSTOMER SEGMENTATION")
fig = plt.figure(figsize=(7, 3))  # Adjust the figure size here
sns.scatterplot(x='AmountSpent', y='Frequency', hue='Cluster_Labels', data=pcaPredsDf, palette='Set1')
# plt.title('Amount Spent vs. Frequency')

plt.tight_layout()
col2.pyplot(fig)

nav = st.sidebar.radio("Navigation",["Home", "Data Exploration","Prediction"])

if nav == "Home":

    st.markdown(html_temp, unsafe_allow_html=True)


if nav == 'Data Exploration':

    plotType       = st.radio("Plot Type",["2D","3D"])
    dataToExplore  = st.radio("Data to Explore",["PCA DATA","Original Data"])
    datas          = {'PCA DATA': pcaPredsDf, 'Original Data':selDataDf}
    df             = datas[dataToExplore]

    combos = st.radio("COMBO", ["AmountSpent/Frequency","AmountSpent/Recency","Frequency/Recency"])
    graph = st.selectbox("What kind of Graph ? ",["Non-Interactive","Interactive"])

    # typeOfCluster  = st.radio("Tye of Cluster",["Normal Kmeans", "Hierarchical Cluster"])

    displayGraph(plotType, df, combos, graph)
    
if nav == "Prediction":
    st.header("PREDICT CUSTOMER CLASSES")
    predType = st.radio("Select Prediction Type",["Upload Data","Predict Single"])
    model    = "PCA KMEANS"
    models   = {"PCA KMEANS": kmeans, 'PCA HEIRARCHICAL':mergings}
    



    if predType == 'Upload Data':
        dat  = st.file_uploader("Upload your csv data", type=['csv','txt'])
        

        if dat is not None:
            dat_df = pd.read_csv(dat)

            if 'AmountSpent' in dat_df.columns and 'Frequency' in dat_df.columns and 'Recency' in dat_df.columns and 'Country' in dat_df.columns:
                cdf = dat_df[['AmountSpent','Frequency','Recency','Country']]
                cdf = scaler.transform(cdf.values)
                cdf = pca.transform(pd.DataFrame(cdf, columns=['AmountSpent','Frequency','Recency','Country']))


                if st.button("Predict"):
                    preds = predict(cdf, model)

                    cdfredsdf = pd.DataFrame(np.c_[cdf, preds], columns=['AmountSpent','Frequency','Recency','Cluster_Labels'])
                    

            else:
                f"EMSURE YOUR DATA IS IN THE RIGHT ORDER. MAKE SURE IT HAS THE COLUMNS \n ['AmountSpent','Frequency','Recency','Country']. AVAILABLE COLUMNS ARE {dat.columns}"

        
        plotType       = st.radio("Plot Type",["2D","3D"])
                    
        combos = st.radio("COMBO", ["AmountSpent/Frequency","AmountSpent/Recency","Frequency/Recency"])
        graph = st.selectbox("What kind of Graph ? ",["Non-Interactive","Interactive"])

        displayGraph(plotType, df, combos, graph)

        

    if predType == 'Predict Single':
        val1 = st.number_input('AmountSpent', 0.0)
        val2 = st.number_input('Frequency', 0.0)
        val3 = st.number_input('Recency', 0.0)
        val4 = st.selectbox("Country",countries)

        val4 = encoder.transform(np.array([[val4]])).tolist()[0][0]

        # f'[{val1,val2,val3,val4}]'

        arr = np.array([[val1,val2,val3,val4]])
        arr = scaler.transform(arr)
        arr = pca.transform(arr)
        
        if st.button("PREDICT"):
            st.success(f"The customer belongs to segment {predict(arr, model)}")
            
