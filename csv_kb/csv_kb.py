import streamlit as st
import random
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import warnings
sns.set_theme(color_codes=True)

import pathlib
import textwrap
import google.generativeai as genai
import os

sns.set_theme(color_codes=True)

st.title("Make your data talk")
st.write("they never lied, with Google Generative AI")

# Upload the CSV file
uploaded_file = st.file_uploader("Upload CSV file:")

# Calculate the number of rows for subplots
def calculate_num_rows(num_cols):
    if num_cols == 0:
        return 1
    return (num_cols + 2) // 3

# Function to convert text to Markdown with indentation
def to_markdown(text):
    text = text.replace('•', '  *')
    return st.markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

# Check if the file is uploaded
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, low_memory=False, encoding='latin-1')

    st.write("Original DataFrame:")
    st.dataframe(df)

    # ── Countplot Barchart ──────────────────────────────────────────────────
    st.write("**Countplot Barchart**")
    cat_vars = [col for col in df.select_dtypes(include='object').columns
                if df[col].nunique() > 1 and df[col].nunique() <= 10]

    if len(cat_vars) == 0:
        st.info("No categorical columns with 2–10 unique values found.")
    else:
        num_cols = len(cat_vars)
        num_rows = calculate_num_rows(num_cols)
        fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5 * num_rows))
        axs = axs.flatten()
        for i, var in enumerate(cat_vars):
            top_values = df[var].value_counts().head(10).index
            filtered_df = df.copy()
            filtered_df[var] = df[var].apply(lambda x: x if x in top_values else 'Other')
            sns.countplot(x=var, data=filtered_df, ax=axs[i])
            axs[i].set_title(var)
            axs[i].tick_params(axis='x', rotation=90)
        for i in range(num_cols, len(axs)):
            fig.delaxes(axs[i])
        fig.tight_layout()
        st.pyplot(fig)
        fig.savefig("plot4.png")

    # ── Histoplot ───────────────────────────────────────────────────────────
    st.write("**Histoplot**")
    num_vars = [col for col in df.select_dtypes(include=['int', 'float']).columns]

    if len(num_vars) == 0:
        st.info("No numeric columns found.")
    else:
        num_cols = len(num_vars)
        num_rows = (num_cols + 2) // 3
        fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5 * num_rows))
        axs = axs.flatten()
        for i, var in enumerate(num_vars):
            sns.histplot(df[var], ax=axs[i], kde=True)
            axs[i].set_title(var)
            axs[i].set_xlabel('')
        for i in range(num_cols, len(axs)):
            fig.delaxes(axs[i])
        fig.tight_layout()
        st.pyplot(fig)
        fig.savefig("plot7.png")

    # ── Target variable & column selection ─────────────────────────────────
    target_variable = st.selectbox("Select target variable:", df.columns)
    columns_for_analysis = st.multiselect(
        "Select columns for analysis:",
        [col for col in df.columns if col != target_variable]
    )

    if st.button("Process"):
        target_variable_data = df[target_variable]
        columns_for_analysis_data = df[columns_for_analysis]

        st.write("Target Variable DataFrame:")
        st.dataframe(df[[target_variable]])

        st.write("Columns for Analysis DataFrame:")
        st.dataframe(df[columns_for_analysis])

        df = pd.concat([target_variable_data, columns_for_analysis_data], axis=1)
        st.write("Columns for Analysis and Target Variable DataFrame:")
        st.dataframe(df)

        # Drop columns with >25% nulls
        null_percentage = df.isnull().sum() / len(df)
        columns_to_drop = null_percentage[null_percentage > 0.25].index
        df.drop(columns=columns_to_drop, inplace=True)

        # Fill <25% nulls with median
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if null_percentage[col] <= 0.25:
                    if df[col].dtype in ['float64', 'int64']:
                        df[col].fillna(df[col].median(), inplace=True)

        # Lowercase all object columns
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.lower()

        st.write("Cleaned Dataset")
        st.dataframe(df)

        # ── Multiclass Barplot ──────────────────────────────────────────────
        st.write("**Multiclass Barplot**")
        cat_vars = df.select_dtypes(include=['object']).columns.tolist()
        if target_variable in cat_vars:
            cat_vars.remove(target_variable)

        if len(cat_vars) == 0:
            st.info("No categorical columns available for multiclass barplot.")
        else:
            num_cols = len(cat_vars)
            num_rows = (num_cols + 2) // 3
            fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5 * num_rows))
            axs = axs.flatten()
            for i, var in enumerate(cat_vars):
                top_categories = df[var].value_counts().nlargest(10).index
                filtered_df = df[df[var].notnull() & df[var].isin(top_categories)]
                sns.countplot(x=var, hue=target_variable, data=filtered_df, ax=axs[i])
                axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=90)
            for i in range(num_cols, len(axs)):
                fig.delaxes(axs[i])
            fig.tight_layout()
            st.pyplot(fig)
            fig.savefig("plot2.png")

        # ── Multiclass Histplot ─────────────────────────────────────────────
        st.write("**Multiclass Histplot**")
        int_vars = df.select_dtypes(include=['int', 'float']).columns.tolist()
        int_vars = [col for col in int_vars if col != target_variable]

        if len(int_vars) == 0:
            st.info("No numeric columns available for multiclass histplot.")
        else:
            num_cols = len(int_vars)
            num_rows = (num_cols + 2) // 3
            fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5 * num_rows))
            axs = axs.flatten()
            for i, var in enumerate(int_vars):
                sns.histplot(data=df, x=var, hue=target_variable, kde=True, ax=axs[i])
                axs[i].set_title(var)
            for i in range(num_cols, len(axs)):
                fig.delaxes(axs[i])
            fig.tight_layout()
            st.pyplot(fig)
            fig.savefig("plot3.png")

    # ── Merged plot grid ────────────────────────────────────────────────────
    plot_paths = ["plot1.png", "plot2.png", "plot3.png", "plot4.png"]
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    for i, plot_path in enumerate(plot_paths):
        row = i // 2
        col = i % 2
        if os.path.isfile(plot_path):
            try:
                img = plt.imread(plot_path)
                axs[row, col].imshow(img)
                axs[row, col].axis('off')
            except Exception as e:
                axs[row, col].text(0.5, 0.5, 'Error loading image', ha='center', va='center')
                axs[row, col].axis('off')
        else:
            axs[row, col].text(0.5, 0.5, 'File not found', ha='center', va='center')
            axs[row, col].axis('off')
    plt.tight_layout()
    plt.savefig("merged_plots.png")

    # ── Gemini vision setup ─────────────────────────────────────────────────
    import textwrap
    from markdown import Markdown

    def to_markdown(text):
        text = text.replace('•', '  *')
        return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

    genai.configure(api_key=input())

    import PIL.Image

    img = PIL.Image.open("merged_plots.png")
    model = genai.GenerativeModel('gemini-pro-vision')
    response = model.generate_content(img)

    def response_generator():
        text = response.text
        for word in text.split():
            yield word + " "
            time.sleep(0.05)

    # ── Chat interface ──────────────────────────────────────────────────────
    st.title("Chat with your Data")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask Your Data"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        img = PIL.Image.open("merged_plots.png")
        model = genai.GenerativeModel('gemini-pro-vision')
        response = model.generate_content([prompt, img], stream=True)
        response.resolve()

        response_text = response.text
        to_markdown(response_text)
        st.write(response.text)

        st.session_state.messages.append({"role": "assistant", "content": response_text})