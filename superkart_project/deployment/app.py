import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# grab the model we pushed from training
model_path = hf_hub_download(
    repo_id="iamsubha/superkart-sales-model",
    filename="best_superkart_model_v1.joblib",
)
model = joblib.load(model_path)

st.title("SuperKart Sales Forecast")
st.write("Enter product and store details to estimate total sales for that product at that store.")

# ---- product inputs ----
st.subheader("Product")
product_weight = st.number_input("Product Weight", min_value=1.0, max_value=30.0, value=12.5)
product_sugar  = st.selectbox("Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
product_area   = st.number_input("Allocated Display Area (ratio)",
                                 min_value=0.0, max_value=1.0, value=0.05)
product_type   = st.selectbox("Product Type", [
    "Frozen Foods", "Dairy", "Canned", "Baking Goods", "Health and Hygiene",
    "Snack Foods", "Meat", "Household", "Hard Drinks", "Fruits and Vegetables",
    "Breads", "Soft Drinks", "Breakfast", "Others", "Starchy Foods", "Seafood",
])
product_cat = st.selectbox("Product Category", ["Food", "Non-Consumable", "Drinks"])
product_mrp = st.number_input("MRP", min_value=1.0, max_value=500.0, value=150.0)

# ---- store inputs ----
st.subheader("Store")
store_age   = st.slider("Store Age (years)", 0, 50, 15)
store_size  = st.selectbox("Store Size", ["Small", "Medium", "High"])
store_city  = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
store_type  = st.selectbox("Store Type",
                           ["Departmental Store", "Supermarket Type1",
                            "Supermarket Type2", "Food Mart"])

# shape input to match the training columns
input_df = pd.DataFrame([{
    "Product_Weight": product_weight,
    "Product_Sugar_Content": product_sugar,
    "Product_Allocated_Area": product_area,
    "Product_Type": product_type,
    "Product_MRP": product_mrp,
    "Store_Size": store_size,
    "Store_Location_City_Type": store_city,
    "Store_Type": store_type,
    "Product_Category": product_cat,
    "Store_Age": store_age,
}])

if st.button("Predict Sales"):
    pred = model.predict(input_df)[0]
    st.success(f"Predicted Total Sales: ₹ {pred:,.2f}")
