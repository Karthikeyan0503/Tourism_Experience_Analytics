import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np
import base64

# Load & Training Models
features = ['UserId', 'CityName', 'Attraction', 'VisitMode', 'AttractionType']
RegClassfeatures = ['UserId', 'CityName', 'Rating', 'Attraction', 'AttractionType']
model = pickle.load(open('rf_regressor.pkl', 'rb'))
rc = pickle.load(open('recomClassifier.pkl', 'rb'))
label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))
rclabel_encoder = pickle.load(open('rclabel_encoder.pkl', 'rb'))

# Load datasets
user_df = pd.read_excel("Tourism dataset\\User.xlsx")
city_df = pd.read_excel("Tourism dataset\\City.xlsx")
continent_df = pd.read_excel("Tourism dataset\\Continent.xlsx")
country_df = pd.read_excel("Tourism dataset\\Country.xlsx")
item_df = pd.read_excel("Tourism dataset\\Item.xlsx")
mode_df = pd.read_excel("Tourism dataset\\Mode.xlsx")
region_df = pd.read_excel("Tourism dataset\\Region.xlsx")
transaction_df = pd.read_excel("Tourism dataset\\Transaction.xlsx")
type_df = pd.read_excel("Tourism dataset\\Type.xlsx")

# EDA
citydetail_df = pd.merge(user_df, city_df, on='CityId', how='inner')
citydetail_df.drop("CountryId_y", axis=1, inplace=True)
citydetail_df.rename(columns={'ContenentId': 'ContentId', 'CountryId_x': 'CountryId'}, inplace=True)
countrydetail_df = pd.merge(citydetail_df, country_df, on='CountryId', how='inner')
countrydetail_df.drop("RegionId_y", axis=1, inplace=True)
countrydetail_df.rename(columns={'RegionId_x': 'RegionId'}, inplace=True)
countrydetail_df = pd.merge(countrydetail_df, region_df, on='RegionId', how='inner')
countrydetail_df.drop("ContentId_y", axis=1, inplace=True)
countrydetail_df.rename(columns={'ContentId_x': 'ContentId'}, inplace=True)
continent_df.rename(columns={'ContenentId': 'ContentId'}, inplace=True)
userdetails_df = pd.merge(countrydetail_df, continent_df, on='ContentId', how='inner')
transactiondetails_df = pd.merge(transaction_df, item_df, on='AttractionId', how='inner')
transactiondetails_df.rename(columns={'VisitMode': 'VisitModeId'}, inplace=True)
transactiondetails_df = pd.merge(transactiondetails_df, mode_df, on='VisitModeId', how='inner')
transactiondetails_df = pd.merge(transactiondetails_df, type_df, on='AttractionTypeId', how='inner')

user_recom = transactiondetails_df
user_recom = user_recom.sample(50000)

user_item_matrix = user_recom.pivot_table('Rating', ['UserId'], 'AttractionId')
user_item_matrix.fillna(0, inplace=True)
user_mat_sim = cosine_similarity(user_item_matrix)


df = pd.merge(userdetails_df, transactiondetails_df, on='UserId', how='inner')
df = df.sample(50000)
df.drop("ContentId", axis=1, inplace=True)
df.drop("RegionId", axis=1, inplace=True)
df.drop("CountryId", axis=1, inplace=True)
df.drop("CityId", axis=1, inplace=True)
df.drop("Country", axis=1, inplace=True)
df.drop("Region", axis=1, inplace=True)
df.drop("Contenent", axis=1, inplace=True)
df.drop("TransactionId", axis=1, inplace=True)
df.drop("VisitYear", axis=1, inplace=True)
df.drop("VisitMonth", axis=1, inplace=True)
df.drop("VisitModeId", axis=1, inplace=True)
df.drop("AttractionId", axis=1, inplace=True)
df.drop("AttractionAddress", axis=1, inplace=True)
df.drop("AttractionCityId", axis=1, inplace=True)
df.drop("AttractionTypeId", axis=1, inplace=True)

print("Unique users in original transaction_df:", transaction_df['UserId'].nunique())
print("Unique users in user_recom:", user_recom['UserId'].nunique())
print("Max user in user_recom", user_recom['UserId'].max())
print("Unique users in df:", df['UserId'].nunique())
print("Max user in df", df['UserId'].max())
print("Max user in user_item_matrix index", user_item_matrix.index.max())

col = ['UserId', 'CityName', 'Attraction', 'VisitMode', 'AttractionType', 'Rating']
df = df[col]

col1 = ['UserId', 'CityName', 'Rating', 'Attraction', 'AttractionType', 'VisitMode']
data = df[col1]

# Collaborative Filter
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

# Recommendation function using regression model
def recom_dest(user_input, model, label_encoder, features, data):
    encodeddata = {}
    for i in features:
        if i in label_encoder:
            encodeddata[i] = label_encoder[i].transform([user_input[i]])[0]
        else:
            encodeddata[i] = user_input[i]

    input_df = pd.DataFrame([encodeddata])
    pred_rate = model.predict(input_df)[0]
    return int(pred_rate)

# Recommendation function using Classifier model
def recomClassifier_dest(user_input, model, label_encoder, features, data):
    encodeddata = {}
    for i in features:
        if i in label_encoder:
            encodeddata[i] = label_encoder[i].transform([user_input[i]])[0]
        else:
            encodeddata[i] = user_input[i]

    input_df = pd.DataFrame([encodeddata])
    pred_rate = model.predict(input_df)[0]
    return pred_rate




# Streamlit App
st.title("Tourism Prediction Analysis")

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


if "page" not in st.session_state:
    st.session_state.page = "Recommendation"  # Default to Recommendation

# Sidebar Buttons
if st.sidebar.button("**Recommend 🌟**"):
    st.session_state.page = "Recommendation"
if st.sidebar.button("**Classify 📊**"):
    st.session_state.page = "Classification"


# Content Display based on Session State
if st.session_state.page == "Recommendation":
    st.header("Recommendation")
    with st.form(key='recommendation_form'):
        user_id = st.number_input("Enter User ID", min_value=1, step=1)
        city = st.text_input("Enter City Name")
        attraction = st.text_input("Enter Attraction Name")
        mode = st.selectbox("Select Visit Mode", mode_df['VisitMode'].unique())
        attraction_type = st.selectbox("Select Attraction Type", type_df['AttractionType'].unique())
        submit_button = st.form_submit_button(label='Recommend')

    if submit_button:
        user_input = {'UserId': user_id, 'CityName': city, 'Attraction': attraction, 'VisitMode': mode, 'AttractionType': attraction_type}
        recom_data = collabfill(user_id, user_mat_sim, user_item_matrix, user_recom)
        pred_rating = recom_dest(user_input, model, label_encoder, features, df)
        if not recom_data.empty:
            st.subheader("Recommended Destinations:")
            st.dataframe(recom_data)
            st.subheader(f"Predicted Rating: {pred_rating}")
        else:
            st.warning("No recommendations found for this user.")

elif st.session_state.page == "Classification":
    st.header("Classification")
    with st.form(key='classification_form'):
        c_user_id = st.number_input("Enter User ID", min_value=1, step=1, key='cuserid')
        c_city = st.text_input("Enter City Name", key='ccity')
        c_rating = st.number_input("Enter Rating", min_value=1, max_value=5, step=1, key='cmode')
        c_attraction = st.text_input("Enter Attraction Name", key='cattraction')
        c_attraction_type = st.selectbox("Select Attraction Type", type_df['AttractionType'].unique(), key='cattractiontype')
        submit_button_class = st.form_submit_button(label="Predict Visit Mode")

    if submit_button_class:
        user_input = {
            'UserId': c_user_id,
            'CityName': c_city,
            'Rating': c_rating,
            'Attraction': c_attraction,
            'AttractionType': c_attraction_type
        }

        pred_visitmode = recomClassifier_dest(user_input, rc, rclabel_encoder, RegClassfeatures, data)
        st.subheader("Predicted Visit Mode:")
        st.write(pred_visitmode)

