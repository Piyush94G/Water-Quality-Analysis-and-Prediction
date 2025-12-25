#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_excel('water_quality.xlsx')


# In[2]:


df.sample(5)


# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


# Find the number of duplicates in the DataFrame
num_duplicates = df.duplicated().sum()

# Print the number of duplicates
print("Number of duplicates:", num_duplicates)


# In[7]:


# Calculate the number of points for each class in 'Quality'
quality_counts = df['Quality'].value_counts()

# Print the counts for each class
print(quality_counts)


# In[8]:


import matplotlib.pyplot as plt

# Define the colors for each quality category
colors = ['skyblue', 'gold', 'salmon', 'lightgreen']

# Plot the bar graph
plt.figure(figsize=(12, 6))
quality_counts.plot(kind='bar', color=colors, edgecolor='black')

# Add labels and title
plt.title('Number of Points for Each Quality Class', fontsize=16)
plt.xlabel('Quality', fontsize=14)
plt.ylabel('Count', fontsize=14)

# Add text labels for each bar
for i, count in enumerate(quality_counts):
    plt.text(i, count + 10, str(count), ha='center', va='bottom', fontsize=12)

# Customize the appearance
plt.xticks(rotation=45, fontsize=12)  # Rotate x-labels and set font size
plt.yticks(fontsize=12)  # Set font size for y-axis labels
plt.grid(axis='y', linestyle='--', alpha=0.5)  # Add horizontal grid lines

plt.ylim(0, max(quality_counts) * 1.3)  # Set the y-axis limit to extend a bit above the maximum count

plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.show()


# In[9]:


label_mapping_manual = {
    'Agriculture Water': 2,
    'Bathing Water': 1,
    'Drinking Water': 0,
    'Polluted Water': 3
}

# Transform the 'Quality' column using the manual label mapping
df['Quality_Encoded_Manual'] = df['Quality'].map(label_mapping_manual)

# Print the mapping of original labels to manually encoded values
print("Manual Label Mapping:")
print(label_mapping_manual)


# In[10]:


import matplotlib.pyplot as plt

label_mapping = {
    'Agriculture Water': 2,
    'Bathing Water': 1,
    'Drinking Water': 0,
    'Polluted Water': 3
}

plt.figure(figsize=(8, 6))
plt.barh(list(label_mapping.keys()), list(label_mapping.values()), color='skyblue')
plt.xlabel('Encoded Value')
plt.ylabel('Quality')
plt.title('Label Mapping')
plt.gca().invert_yaxis()  # Invert y-axis to show 'Agriculture Water' at the top
plt.show()


# In[11]:


import matplotlib.pyplot as plt
import seaborn as sns

cols = ['Quality', 'DissolvedOxygen', 'pH', 'BOD', 'Nitrate']
Corr = df[cols].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(Corr, annot=True, cmap='cividis')
plt.title('HeatMap')
plt.show()


# In[12]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import classification_report


# In[13]:


prediction_feature = [  'DissolvedOxygen','pH',  'BOD', 'Nitrate',]

targeted_feature = 'Quality'

len(prediction_feature)


# In[14]:


x = df[[ 'DissolvedOxygen','pH',  'BOD', 'Nitrate',]]
x


# In[15]:


y= df['Quality']


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)


# In[17]:


sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[18]:


def model_building(model, X_train, X_test, y_train, y_test):
    """
    Model Fitting, Prediction And Other stuff
    return ('score', 'accuracy_score', 'predictions' )
    """

    model.fit(X_train, y_train)
    score = model.score(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(predictions, y_test)

    return (score, accuracy, predictions)


# In[19]:


models_list = {
    "LogisticRegression": LogisticRegression(),
    "RandomForestClassifier": RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=5),
    "DecisionTreeClassifier": DecisionTreeClassifier(criterion='entropy', random_state=0),
    "SVC": SVC(),
}


# In[20]:


print(lzist(models_list.values()))


# In[21]:


def cm_metrix_graph(cm):
    sns.heatmap(cm, annot=True, fmt="d")
    plt.show()


# In[22]:


df_prediction = []
confusion_matrixs = []
df_prediction_cols = [ 'model_name', 'score', 'accuracy_score' , "accuracy_percentage"]

for name, model in zip(list(models_list.keys()), list(models_list.values())):

    (score, accuracy, predictions) = model_building(model, X_train, X_test, y_train, y_test )

    print("\n\nClassification Report of '"+ str(name), "'\n")

    print(classification_report(y_test, predictions))

    df_prediction.append([name, score, accuracy, "{0:.2%}".format(accuracy)])

    # For Showing Metrics
    confusion_matrixs.append(confusion_matrix(y_test, predictions))


df_pred = pd.DataFrame(df_prediction, columns=df_prediction_cols)


# In[115]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 2))
for index, cm in enumerate(confusion_matrixs):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap='cividis')
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.title(f'Confusion Matrix {index+1}')
    plt.tight_layout(pad=2)
    plt.show()


# In[24]:


df_pred


# In[ ]:





# In[25]:


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[['DissolvedOxygen', 'pH', 'BOD', 'Nitrate']], df['Quality'], test_size=0.3, random_state=42)

# Train the SVC model
svc_model = SVC()
svc_model.fit(X_train, y_train)


# In[ ]:





# # 2017

# In[26]:


import pandas as pd
df_2017 = pd.read_excel('2017_collective.xlsx')
df_2017


# In[29]:


predictions = svc_model.predict(df_2017[['DissolvedOxygen', 'pH', 'BOD', 'Nitrate']])

# Add predictions to a new column in the dataset
df_2017['Predicted_Quality'] = predictions

# Display the updated dataset with predictions
df_2017


# In[30]:


# Calculate the number of points for each class in 'Quality'
quality_counts = df_2017['Predicted_Quality'].value_counts()

# Print the counts for each class
print(quality_counts)


# In[ ]:





# # 2018

# In[31]:


import pandas as pd
df_2018 = pd.read_excel('2018_collective.xlsx')
df_2018


# In[32]:


predictions = svc_model.predict(df_2018[['DissolvedOxygen', 'pH', 'BOD', 'Nitrate']])

# Add predictions to a new column in the dataset
df_2018['Predicted_Quality'] = predictions

# Display the updated dataset with predictions
df_2018


# In[33]:


# Calculate the number of points for each class in 'Quality'
quality_counts = df_2018['Predicted_Quality'].value_counts()

# Print the counts for each class
print(quality_counts)


# In[ ]:





# In[ ]:





# # 2019

# In[34]:


import pandas as pd
df_2019 = pd.read_excel('2019_collective.xlsx')
df_2019


# In[35]:


predictions = svc_model.predict(df_2019[['DissolvedOxygen', 'pH', 'BOD', 'Nitrate']])

# Add predictions to a new column in the dataset
df_2019['Predicted_Quality'] = predictions

# Display the updated dataset with predictions
df_2019


# In[36]:


# Calculate the number of points for each class in 'Quality'
quality_counts = df_2019['Predicted_Quality'].value_counts()

# Print the counts for each class
print(quality_counts)


# In[ ]:





# In[ ]:





# # 2020

# In[39]:


import pandas as pd
df_2020 = pd.read_excel('2020_collective.xlsx')
df_2020


# In[40]:


predictions = svc_model.predict(df_2020[['DissolvedOxygen', 'pH', 'BOD', 'Nitrate']])

# Add predictions to a new column in the dataset
df_2020['Predicted_Quality'] = predictions

# Display the updated dataset with predictions
df_2020


# In[41]:


# Calculate the number of points for each class in 'Quality'
quality_counts = df_2020['Predicted_Quality'].value_counts()

# Print the counts for each class
print(quality_counts)


# In[ ]:





# In[ ]:





# # 2021

# In[42]:


import pandas as pd
df_2021 = pd.read_excel('2021_collective.xlsx')
df_2021


# In[43]:


predictions = svc_model.predict(df_2021[['DissolvedOxygen', 'pH', 'BOD', 'Nitrate']])

# Add predictions to a new column in the dataset
df_2021['Predicted_Quality'] = predictions

# Display the updated dataset with predictions
df_2021


# In[44]:


# Calculate the number of points for each class in 'Quality'
quality_counts = df_2021['Predicted_Quality'].value_counts()

# Print the counts for each class
print(quality_counts)


# In[ ]:





# In[ ]:





# # MAP

# # 2017

# In[45]:


import folium
from folium.plugins import MarkerCluster

# Create a map centered on India
india_map = folium.Map(location=[20.5937, 78.9629], zoom_start=5)

# Define a style function for the GeoJSON layer
def style_function(feature):
    return {
        'fillColor': 'lightgreen',
        'color': 'lightgreen',
        'weight': 2
    }

# Add the GeoJSON data for India's map
folium.GeoJson(
    'states_india.geojson',
    name='india_map',
    style_function=style_function
).add_to(india_map)

# Define a mapping for the quality classes
quality_mapping = {
    'Agriculture Water': 2,
    'Bathing Water': 1,
    'Drinking Water': 0,
    'Polluted Water': 3
}

# Define colors for different quality classes
colors = ["green", 'yellow', 'blue', 'red']

# Create a MarkerCluster layer
marker_cluster = MarkerCluster().add_to(india_map)

# Iterate over the rows of df_2017 and add markers to the map
for index, row in df_2017.iterrows():
    quality_label = row['Predicted_Quality']
    color = colors[quality_mapping.get(quality_label, 0)]  # Get color based on Predicted_Quality
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=f"{row['location']}, {row['state']}",
        icon=folium.Icon(color=color)
    ).add_to(marker_cluster)

# Display the map
india_map


# In[ ]:





# # 2018

# In[46]:


import folium
from folium.plugins import MarkerCluster

# Create a map centered on India
india_map = folium.Map(location=[20.5937, 78.9629], zoom_start=5)

# Define a style function for the GeoJSON layer
def style_function(feature):
    return {
        'fillColor': 'lightgreen',
        'color': 'lightgreen',
        'weight': 2
    }

# Add the GeoJSON data for India's map
folium.GeoJson(
    'states_india.geojson',
    name='india_map',
    style_function=style_function
).add_to(india_map)

# Define a mapping for the quality classes
quality_mapping = {
    'Agriculture Water': 2,
    'Bathing Water': 1,
    'Drinking Water': 0,
    'Polluted Water': 3
}

# Define colors for different quality classes
colors = ["green", 'yellow', 'blue', 'red']

# Create a MarkerCluster layer
marker_cluster = MarkerCluster().add_to(india_map)

# Iterate over the rows of df_2018 and add markers to the map
for index, row in df_2018.iterrows():
    quality_label = row['Predicted_Quality']
    color = colors[quality_mapping.get(quality_label, 0)]  # Get color based on Predicted_Quality
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=f"{row['location']}, {row['state']}",
        icon=folium.Icon(color=color)
    ).add_to(marker_cluster)

# Display the map
india_map


# In[ ]:





# # 2019

# In[47]:


import folium
from folium.plugins import MarkerCluster

# Create a map centered on India
india_map = folium.Map(location=[20.5937, 78.9629], zoom_start=5)

# Define a style function for the GeoJSON layer
def style_function(feature):
    return {
        'fillColor': 'lightgreen',
        'color': 'lightgreen',
        'weight': 2
    }

# Add the GeoJSON data for India's map
folium.GeoJson(
    'states_india.geojson',
    name='india_map',
    style_function=style_function
).add_to(india_map)

# Define a mapping for the quality classes
quality_mapping = {
    'Agriculture Water': 2,
    'Bathing Water': 1,
    'Drinking Water': 0,
    'Polluted Water': 3
}

# Define colors for different quality classes
colors = ["green", 'yellow', 'blue', 'red']

# Create a MarkerCluster layer
marker_cluster = MarkerCluster().add_to(india_map)

# Iterate over the rows of df_2019 and add markers to the map
for index, row in df_2019.iterrows():
    quality_label = row['Predicted_Quality']
    color = colors[quality_mapping.get(quality_label, 0)]  # Get color based on Predicted_Quality
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=f"{row['location']}, {row['state']}",
        icon=folium.Icon(color=color)
    ).add_to(marker_cluster)

# Display the map
india_map


# In[ ]:





# # 2020

# In[48]:


import folium
from folium.plugins import MarkerCluster

# Create a map centered on India
india_map = folium.Map(location=[20.5937, 78.9629], zoom_start=5)

# Define a style function for the GeoJSON layer
def style_function(feature):
    return {
        'fillColor': 'lightgreen',
        'color': 'lightgreen',
        'weight': 2
    }

# Add the GeoJSON data for India's map
folium.GeoJson(
    'states_india.geojson',
    name='india_map',
    style_function=style_function
).add_to(india_map)

# Define a mapping for the quality classes
quality_mapping = {
    'Agriculture Water': 2,
    'Bathing Water': 1,
    'Drinking Water': 0,
    'Polluted Water': 3
}

# Define colors for different quality classes
colors = ["green", 'yellow', 'blue', 'red']

# Create a MarkerCluster layer
marker_cluster = MarkerCluster().add_to(india_map)

# Iterate over the rows of df_2020 and add markers to the map
for index, row in df_2020.iterrows():
    quality_label = row['Predicted_Quality']
    color = colors[quality_mapping.get(quality_label, 0)]  # Get color based on Predicted_Quality
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=f"{row['location']}, {row['state']}",
        icon=folium.Icon(color=color)
    ).add_to(marker_cluster)

# Display the map
india_map


# In[ ]:





# # 2021

# In[49]:


import folium
from folium.plugins import MarkerCluster

# Create a map centered on India
india_map = folium.Map(location=[20.5937, 78.9629], zoom_start=5)

# Define a style function for the GeoJSON layer
def style_function(feature):
    return {
        'fillColor': 'lightgreen',
        'color': 'lightgreen',
        'weight': 2
    }

# Add the GeoJSON data for India's map
folium.GeoJson(
    'states_india.geojson',
    name='india_map',
    style_function=style_function
).add_to(india_map)

# Define a mapping for the quality classes
quality_mapping = {
    'Agriculture Water': 2,
    'Bathing Water': 1,
    'Drinking Water': 0,
    'Polluted Water': 3
}

# Define colors for different quality classes
colors = ["green", 'yellow', 'blue', 'red']

# Create a MarkerCluster layer
marker_cluster = MarkerCluster().add_to(india_map)

# Iterate over the rows of df_2021 and add markers to the map
for index, row in df_2021.iterrows():
    quality_label = row['Predicted_Quality']
    color = colors[quality_mapping.get(quality_label, 0)]  # Get color based on Predicted_Quality
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=f"{row['location']}, {row['state']}",
        icon=folium.Icon(color=color)
    ).add_to(marker_cluster)

# Display the map
india_map


# In[ ]:





# In[ ]:





# In[54]:


import pandas as pd
df_complete = pd.read_excel('complete.xlsx')
df_complete


# In[59]:





# In[114]:


import matplotlib.pyplot as plt
import seaborn as sns

# Filter the data for the years 2019, 2020, and 2021
covid_period = df_complete[df_complete['year'].isin([2019, 2020, 2021])]

# Define a custom color palette
custom_palette = {
    'Agriculture Water': 'green',
    'Bathing Water': 'gold',
    'Drinking Water': 'blue',
    'Polluted Water': 'red'
}

# Plot the water quality distribution during the COVID-19 period
plt.figure(figsize=(9, 4))  # Adjust the figsize here
sns.countplot(data=covid_period, x='year', hue='Predicted_Quality', palette=custom_palette, edgecolor='black')
plt.title("Water Quality Distribution During COVID-19 Period")
plt.xlabel("Year")
plt.ylabel("Count")
plt.legend(title='Predicted Quality', loc='upper right')
plt.tight_layout()
plt.show()


# In[ ]:




