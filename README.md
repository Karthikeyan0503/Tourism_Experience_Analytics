# Tourism Prediction Analytics with Machine Learning

## Project Overview
This project involves combining multiple CSV files into a single dataset, cleaning the dataset, and training machine learning models for three key objectives.

1. **Recommendations:** Personalized attraction suggestions.
2. **Regression:** Predicting attraction ratings.
3. **Classification:** User visit mode prediction.
4. **Visualization:** Data Visualizations for Personalized Attraction Recommendation

Additionally, a **Streamlit application** has been built to provide a user-friendly interface for interacting with the trained models.

---

## Objective 1: Recommendations - Personalized Attraction Suggestions

### **Objective:**
Develop a recommendation system to suggest tourist attractions based on a user's historical preferences and similar users’ preferences.

### **Use Cases:**
- Travel platforms can guide users toward attractions they are most likely to enjoy, increasing engagement.
- Destination management organizations can promote attractions aligned with user interests.

### **Types of Recommendation Approaches:**
- **Collaborative Filtering:** Recommend attractions based on similar users' ratings and preferences.
- **Content-Based Filtering:** Suggest attractions similar to those already visited by the user based on features like attraction type, location, and amenities.
- **Hybrid Systems:** Combine collaborative and content-based methods for improved accuracy.

### **Inputs (Features):**
- **User visit history:** Attractions visited, ratings given.
- **Attraction features:** Type, location, popularity.
- **Similar user data:** Travel patterns and preferences.

### **Output:**
- Ranked list of recommended attractions.

## Objective 2: Regression - Predicting Attraction Ratings

### **Objective:**
Develop a regression model to predict the rating a user might give to a tourist attraction based on historical data, user demographics, and attraction features.

### **Use Cases:**
- Travel platforms can estimate user satisfaction for specific attractions.
- Identify attractions likely to receive lower ratings and take corrective actions.
- Personal travel guides can enhance user experience by suggesting attractions aligned with preferences.

### **Possible Inputs (Features):**
- **User demographics:** Continent, region, country, city.
- **Visit details:** Year, month, mode of visit (e.g., business, family, friends).
- **Attraction attributes:** Type (e.g., beaches, ruins), location, previous average ratings.

### **Target:**
- Predicted rating (on a scale, e.g., 1-5).

---

## Objective 3: Classification - User Visit Mode Prediction

### **Objective:**
Create a classification model to predict the mode of visit (e.g., business, family, couples, friends) based on user and attraction data.

### **Use Cases:**
- Travel platforms can personalize marketing campaigns based on user visit mode predictions.
- Hotels and attraction organizers can plan resources more efficiently.

### **Inputs (Features):**
- **User demographics:** Continent, region, country, city.
- **Attraction characteristics:** Type, popularity, previous visitor demographics.
- **Historical visit data:** Month, year, previous visit modes.

### **Target:**
- Visit mode categories (e.g., Business, Family, Couples, Friends, etc.).

---

## Objective 4: Visualization - Data Visualizations for Personalized Attraction Recommendation

### **Objective:**
Focuses on visualizing key aspects of the tourist attraction dataset to provide insights that support the development and evaluation of our personalized attraction recommendation system.

### **Use Cases:**
- **Understand Data Distributions:** Visualize the distribution of user ratings to understand user preferences and rating patterns.
- **Identify Popular Trends:** Highlight the most popular attraction types, visit modes, and cities to understand common tourist interests.
- **Explore Feature Correlations:** Investigate correlations between numerical features to identify potential relationships that can inform recommendation algorithms.
- **Provide Visual Insights:** Offer clear and informative visualizations to facilitate data understanding and communication of findings.

### **Types of Visualizations Approaches:**
1.  **Distribution of Ratings:**
    * A histogram showing the frequency of different rating values.
    * This helps understand the overall sentiment of users towards attractions.

2.  **Top 10 Attraction Types:**
    * A bar plot displaying the top 10 most frequent attraction types.
    * This reveals the most popular categories of attractions.

3.  **Top 10 Visit Modes:**
    * A bar plot showing the top 10 most common visit modes.
    * This provides insights into how users prefer to experience attractions.

4.  **Correlation Heatmap:**
    * A heatmap illustrating the correlation matrix of numerical features.
    * This helps identify relationships between features such as ratings, popularity, and other numerical attributes.

5.  **Top 10 Cities:**
    * A barplot showing the top 10 cities with most attractions or visits.
    * This shows the most popular destination cities.

### **Output:**
- Gain a deeper understanding of the dataset's characteristics.
- Identify patterns and trends that can be leveraged in the recommendation system.
- Communicate key findings to stakeholders.
- Aid in debugging the recommendation systems, by visually representing the data the system is working with.

---

## Streamlit Application

A **Streamlit app** has been developed to provide a user-friendly interface for interacting with the machine learning models. Users can input details and receive predictions for:
- Recommended attractions.
- Attraction ratings.
- Visit modes.
- Visualization on attraction recommendation.

---

## Technologies Used

- **Python** (Pandas, NumPy, Scikit-Learn, Pickle, Scipy) for data processing and model training.
- **Streamlit** (Streamlit,base64,pandas,pickle,matplotlib,seaborn,numpy)for building the interactive web application.

---

## How to Run the Project

1. Clone the repository from GitHub.
2. Install required dependencies using `pip install -r requirements.txt`.
4. Run the Streamlit application using `streamlit run app.py`.

---
