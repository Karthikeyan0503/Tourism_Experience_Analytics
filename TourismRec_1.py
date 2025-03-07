import base64
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
def load_data():
    try:
        with open('rf_regressor.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('recomClassifier.pkl', 'rb') as f:
            rc = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        with open('rclabel_encoder.pkl', 'rb') as f:
            rclabel_encoder = pickle.load(f)
        with open('user_mat_sim.pkl', 'rb') as f:
            user_mat_sim = pickle.load(f)
        with open('user_item_matrix.pkl', 'rb') as f:
            user_item_matrix = pickle.load(f)
        with open('user_recom.pkl', 'rb') as f:
            user_recom = pickle.load(f)
        with open('df.pkl', 'rb') as f:
            df = pickle.load(f)
        with open('mode_df.pkl', 'rb') as f:
            mode_df = pickle.load(f)
        with open('type_df.pkl', 'rb') as f:
            type_df = pickle.load(f)
        return model, rc, label_encoder, rclabel_encoder, user_mat_sim, user_item_matrix, user_recom, df, mode_df, type_df
    except EOFError:
        st.error("Error: One or more pickle files are corrupted. Please recreate them.")
        return None, None, None, None, None, None, None, None, None, None #return none so that the program does not try to use the broken data.
    except FileNotFoundError:
        st.error("Error: One or more pickle files are missing.")
        return None, None, None, None, None, None, None, None, None, None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None, None, None, None, None, None, None, None, None, None

model, rc, label_encoder, rclabel_encoder, user_mat_sim, user_item_matrix, user_recom, df, mode_df, type_df = load_data()

features = ['UserId', 'CityName', 'Attraction', 'VisitMode', 'AttractionType']
RegClassfeatures = ['CityName', 'Rating', 'Attraction']


loaded_data = load_data()

if loaded_data[0] is not None:
    model, rc, label_encoder, rclabel_encoder, user_mat_sim, user_item_matrix, user_recom, df, mode_df, type_df = loaded_data
else:
    st.stop() 

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('Classified.jpg')
    
# Recommendation functions (same as before)
def collabfill(user_id, user_mat_sim, user_item_matrix, user_recom):
    num_users = user_mat_sim.shape[0]

    if user_id < 1 or user_id > num_users:
        st.error(f"Error: User ID {user_id} is out of range. Valid range: 1 to {num_users}")
        return pd.DataFrame()

    simuser = user_mat_sim[user_id - 1]
    sim_user_ids = np.argsort(simuser)[::-1][1:6]
    sim_user_rating = user_item_matrix.iloc[sim_user_ids].mean(axis=0)
    rec_dest_id = sim_user_rating.sort_values(ascending=False).head(5).index
    rec = user_recom[user_recom['AttractionId'].isin(rec_dest_id)][['Attraction', 'VisitMode', 'Rating']].drop_duplicates().head(5)

    return rec

def recom_dest(user_input, model, label_encoder, features, df):
    encodeddata = {}
    for i in features:
        if i in label_encoder:
            encodeddata[i] = label_encoder[i].transform([user_input[i]])[0]
        else:
            encodeddata[i] = user_input[i]

    input_df = pd.DataFrame([encodeddata])
    pred_rate = model.predict(input_df)[0]
    return int(pred_rate)

def recomClassifier_dest(user_input, model, label_encoder, features, df):
    encodeddata = {}
    for i in features:
        if i in label_encoder:
            encodeddata[i] = label_encoder[i].transform([user_input[i]])[0]
        else:
            encodeddata[i] = user_input[i]

    input_df = pd.DataFrame([encodeddata])
    input_df = input_df[features]  # Reorder columns to match RegClassfeatures
    print("Input DataFrame Columns:", input_df.columns)
    pred_rate = model.predict(input_df)[0]
    return pred_rate

# Streamlit App
st.title("Tourism Prediction Analysis")

def recommendation():
    st.header("Personalized Attraction Suggestions")
  
    with st.form(key='recommendation_form'):
        user_id = st.number_input("Enter User ID", min_value=1, step=1)
        city = st.text_input("Enter City Name (Optional)")
        attraction = st.text_input("Enter Attraction Name (Optional)")
        mode = st.selectbox("Select Visit Mode (Optional)", mode_df['VisitMode'].unique())
        attraction_type = st.selectbox("Select Attraction Type (Optional)", type_df['AttractionType'].unique())
        submit_button = st.form_submit_button(label='Recommend')

    if submit_button:
        recom_data = collabfill(user_id, user_mat_sim, user_item_matrix, user_recom)        
        if not recom_data.empty:
            st.subheader("Recommended Destinations:")
            st.dataframe(recom_data)            
        else:
            st.warning("No recommendations found for this user.")

    
def regression():
    st.header("Predicting Attraction Ratings")
    st.write("Predict the rating a user might give to a tourist attraction based on historical data.")
    with st.form(key='regression_form'):
        user_id_reg = st.number_input("Enter User ID", min_value=1, step=1)
        city_reg = st.text_input("Enter City Name")
        attraction_reg = st.text_input("Enter Attraction Name")
        mode_reg = st.selectbox("Select Visit Mode", mode_df['VisitMode'].unique(), key='reg_mode')
        attraction_type_reg = st.selectbox("Select Attraction Type", type_df['AttractionType'].unique(), key='reg_type')
        continent_reg = st.text_input("Enter Continent (Optional)")
        region_reg = st.text_input("Enter Region (Optional)")
        country_reg = st.text_input("Enter Country (Optional)")
        year_reg = st.number_input("Enter Year (Optional)", min_value=1900, max_value=2100, value=2023, step=1)
        month_reg = st.number_input("Enter Month (Optional)", min_value=1, max_value=12, value=1, step=1)

        submit_button_reg = st.form_submit_button(label='Predict Rating')

    if submit_button_reg:
        user_input = {
            'UserId': user_id_reg,
            'CityName': city_reg,
            'Attraction': attraction_reg,
            'VisitMode': mode_reg,
            'AttractionType': attraction_type_reg,
            'Continent': continent_reg,
            'Region': region_reg,
            'Country': country_reg,
            'Year': year_reg,
            'Month': month_reg,
        }
        
        # Assuming you have a separate regression model and features for this task
        # and that the recom_dest function works with the correct model and features.
        predicted_rating_reg = recom_dest(user_input, model, label_encoder, features, df)
        st.subheader(f"Predicted Rating: {predicted_rating_reg}")

def classification():
    st.header("Classification")
    with st.form(key='classification_form'):
        #c_user_id = st.number_input("Enter User ID", min_value=1, step=1, key='cuserid')
        c_city = st.text_input("Enter City Name", key='ccity')
        c_rating = st.number_input("Enter Rating", min_value=1, max_value=5, step=1, key='cmode')
        c_attraction = st.text_input("Enter Attraction Name", key='cattraction')
        #c_attraction_type = st.selectbox("Select Attraction Type", type_df['AttractionType'].unique(), key='cattractiontype')
        submit_button_class = st.form_submit_button(label="Predict Visit Mode")

    if submit_button_class:
        user_input = {
            #'UserId': c_user_id,
            'CityName': c_city,
            'Rating': c_rating,
            'Attraction': c_attraction,
            #'AttractionType': c_attraction_type
        }
        # print("User Input:", user_input)
        # print("RegClassfeatures:", features)

        pred_visitmode = recomClassifier_dest(user_input, rc, rclabel_encoder,RegClassfeatures,df)
        st.subheader("Predicted Visit Mode:")
        st.write(pred_visitmode)

def visualizations():
    st.header("Data Visualizations")

    # 1. Distribution of Ratings
    st.subheader("Distribution of Ratings")
    fig1, ax1 = plt.subplots()
    sns.histplot(df['Rating'], bins=5, kde=True, ax=ax1)
    st.pyplot(fig1)

    # 2. Top 10 Attraction Types
    st.subheader("Top 10 Attraction Types")
    fig2, ax2 = plt.subplots()
    top_attraction_types = type_df['AttractionType'].value_counts().head(10)
    sns.barplot(x=top_attraction_types.index, y=top_attraction_types.values, ax=ax2)
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig2)

    # 3. Top 10 Visit Modes
    st.subheader("Top 10 Visit Modes")
    fig3, ax3 = plt.subplots()
    top_visit_modes = mode_df['VisitMode'].value_counts().head(10)
    sns.barplot(x=top_visit_modes.index, y=top_visit_modes.values, ax=ax3)
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig3)

    # 4. Correlation Heatmap
    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=np.number)
    corr_matrix = numeric_df.corr()
    fig4, ax4 = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax4)
    st.pyplot(fig4)

    # 5. Top 10 cities
    st.subheader("Top 10 Cities")
    fig5, ax5 = plt.subplots()
    top_cities = df['CityName'].value_counts().head(10)
    sns.barplot(x=top_cities.index, y=top_cities.values, ax=ax5)
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig5)

def about():
    st.header("About the App")
    st.write("This app provides tourism recommendations and predictions using machine learning models.")
    st.write("It includes features for recommending destinations based on user preferences, predicting ratings, and classifying visit modes.")
    st.write("The models are trained on a tourism dataset, and the app uses Streamlit for its user interface.")

def help():
    st.header("Help")
    st.write("For recommendations, enter the User ID, City Name, Attraction Name, select Visit Mode, and Attraction Type, then click 'Recommend'.")
    st.write("For classification, enter the User ID, City Name, Rating, Attraction Name, and select Attraction Type, then click 'Predict Visit Mode'.")
    st.write("The 'Visualizations' section displays various graphs generated from the dataset.")

def main():
    option = st.sidebar.radio("Select an Option", ["Recommendation", "Regression", "Classification", "Visualizations", "About", "Help"], index=0)

    if option == "Recommendation":
        recommendation()
    elif option == "Regression":
        regression()
    elif option == "Classification":
        classification()
    elif option == "Visualizations":
        visualizations()
    elif option == "About":
        about()
    elif option == "Help":
        help()
  
if __name__ == "__main__":
    # ... (your load_data() and data loading logic) ...
    main()


