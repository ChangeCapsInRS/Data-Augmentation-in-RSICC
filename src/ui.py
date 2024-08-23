import subprocess

import streamlit as st

import augmentation_methods  # Import the functionality from the original file

# Set up Streamlit UI
st.title("Augmentation method selector")

# Create checkboxes for each augmentation method
checkboxes = []
for method in augmentation_methods.AUGMENTATION_METHODS:
    checkbox = st.checkbox(method.__name__)
    checkboxes.append(checkbox)

# Create a button to start augmentation
start_button = st.button("Start augmentation")
if start_button:
    selected_methods = [
        augmentation_methods.AUGMENTATION_METHODS[i]
        for i in range(len(augmentation_methods.AUGMENTATION_METHODS))
        if checkboxes[i]
    ]
    selected_method_names = [method.__name__ for method in selected_methods]
    print("Selected augmentation methods:", selected_method_names)
    # Call the augmentation function from augmentation.py with selected_methods
    subprocess.call(
        ["python", "src/augment.py", "-o", "output", "--augmenter-names"]
        + selected_method_names
    )
    st.info("Augmentation is completed.")
