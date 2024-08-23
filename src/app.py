# TODO: add other augmentation methods and their settings
import streamlit as st

from schemas import ImagePairs

st.title("Augmentation")

st.write(
    "This is a simple app to demonstrate the use of the augmentation package."
)

dataset = ImagePairs.load("images/merged_no_aug.json", "images").pairs

# get parameters from the sidebar
with st.sidebar:
    st.selectbox("Select augmentation", ["Blur"], disabled=True)
    selected_fileName = st.selectbox(
        "Select intent",
        [intent.filename for intent in dataset],
    )

# Get the selected data
selected_data = next(
    (intent for intent in dataset if intent.filename == selected_fileName),
    None,
)

# sanity check
assert selected_data is not None, "Intent not found"

with st.expander("Original Data"):
    selected_data.st_write()

with st.expander("Augmented images"):
    pass
