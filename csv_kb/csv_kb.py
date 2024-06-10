import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import google.generativeai as genai
import PIL.Image
import os

sns.set_theme(color_codes=True)

st.title("Make your data talk")
st.write("they never lied, with Google Generative AI")

# Upload the CSV file
uploaded_file = st.file_uploader("Upload CSV file:")

# Check if the file is uploaded
if uploaded_file is not None:
    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(uploaded_file, low_memory=False, encoding='latin-1')

    # Show the original DataFrame
    st.write("Original DataFrame:")
    st.dataframe(df)

    st.write("**Countplot Barchart**")

    # Get the names of all columns with data type 'object' (categorical columns) excluding 'Country'
    cat_vars = [col for col in df.select_dtypes(include='object').columns if df[col].nunique() > 1 and df[col].nunique() <= 10]

    # Create a figure with subplots
    num_cols = len(cat_vars)
    num_rows = (num_cols + 2) // 3
    fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5*num_rows))
    axs = axs.flatten()

    # Create a countplot for the top 10 values of each categorical variable using Seaborn
    for i, var in enumerate(cat_vars):
        top_values = df[var].value_counts().head(10).index
        filtered_df = df.copy()
        filtered_df[var] = df[var].apply(lambda x: x if x in top_values else 'Other')
        sns.countplot(x=var, data=filtered_df, ax=axs[i])
        axs[i].set_title(var)
        axs[i].tick_params(axis='x', rotation=90)

    # Remove any extra empty subplots if needed
    if num_cols < len(axs):
        for i in range(num_cols, len(axs)):
            fig.delaxes(axs[i])

    # Adjust spacing between subplots
    fig.tight_layout()

    # Show plots using Streamlit
    st.pyplot(fig)
    fig.savefig("plot4.png")

    st.write("**Histoplot**")
    # Get the names of all columns with data type 'int' or 'float'
    num_vars = [col for col in df.select_dtypes(include=['int', 'float']).columns]

    # Create a figure with subplots
    num_cols = len(num_vars)
    num_rows = (num_cols + 2) // 3
    fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5*num_rows))
    axs = axs.flatten()

    # Create a histplot for each numeric variable using Seaborn
    for i, var in enumerate(num_vars):
        sns.histplot(df[var], ax=axs[i], kde=True)
        axs[i].set_title(var)
        axs[i].set_xlabel('')

    # Remove any extra empty subplots if needed
    if num_cols < len(axs):
        for i in range(num_cols, len(axs)):
            fig.delaxes(axs[i])

    # Adjust spacing between subplots
    fig.tight_layout()

    # Show plots using Streamlit
    st.pyplot(fig)
    fig.savefig("plot7.png")

    # Select target variable
    target_variable = st.selectbox("Select target variable:", df.columns)

    # Select columns for analysis
    columns_for_analysis = st.multiselect("Select columns for analysis:", [col for col in df.columns if col != target_variable])

    # Process button
    if st.button("Process"):
        # Select the target variable and columns for analysis from the original DataFrame
        target_variable_data = df[target_variable]
        columns_for_analysis_data = df[columns_for_analysis]

        # Display target variable in a dataframe
        target_variable_df = df[[target_variable]]
        st.write("Target Variable DataFrame:")
        st.dataframe(target_variable_df)

        # Display columns for analysis in a dataframe
        columns_for_analysis_df = df[columns_for_analysis]
        st.write("Columns for Analysis DataFrame:")
        st.dataframe(columns_for_analysis_df)

        # Concatenate target variable and columns for analysis into a single DataFrame
        df = pd.concat([target_variable_data, columns_for_analysis_data], axis=1)
        st.write("Columns for Analysis and Target Variable DataFrame:")
        st.dataframe(df)

        try:
            # Drop columns with null values more than 25%
            null_percentage = df.isnull().sum() / len(df)
            columns_to_drop = null_percentage[null_percentage > 0.25].index
            df.drop(columns=columns_to_drop, inplace=True)
        
            # Fill missing values below 25% with median
            for col in df.columns:
                if df[col].isnull().sum() > 0:
                    if null_percentage[col] <= 0.25:
                        if df[col].dtype in ['float64', 'int64']:
                            median_value = df[col].median()
                            df[col].fillna(median_value, inplace=True)
        
            # Convert object datatype columns to lowercase
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].str.lower()
        except Exception as e:
            print(f"Error during data cleaning and preprocessing: {e}")

        st.write("Cleaned Dataset")
        st.dataframe(df)

        st.write("**Multiclass Barplot**")
        # Get the names of all columns with data type 'object' (categorical variables)
        cat_vars = df.select_dtypes(include=['object']).columns.tolist()

        # Exclude 'Country' from the list if it exists in cat_vars
        if target_variable in cat_vars:
            cat_vars.remove(target_variable)

        # Create a figure with subplots, but only include the required number of subplots
        num_cols = len(cat_vars)
        num_rows = (num_cols + 2) // 3  # To make sure there are enough rows for the subplots
        fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5*num_rows))
        axs = axs.flatten()

        # Create a count plot for each categorical variable
        for i, var in enumerate(cat_vars):
            top_categories = df[var].value_counts().nlargest(10).index
            filtered_df = df[df[var].notnull() & df[var].isin(top_categories)]  # Exclude rows with NaN values in the variable
            sns.countplot(x=var, hue=target_variable, data=filtered_df, ax=axs[i])
            tick_labels = [item.get_text() for item in axs[i].get_xticklabels()]
            axs[i].set_xticks(range(len(tick_labels)))
            axs[i].set_xticklabels(tick_labels, rotation=90)

        # Remove any remaining blank subplots
        for i in range(num_cols, len(axs)):
            fig.delaxes(axs[i])

        # Adjust spacing between subplots
        fig.tight_layout()

        # Show plot
        st.pyplot(fig)
        fig.savefig("plot2.png")

        st.write("**Multiclass Histplot**")
        # Get the names of all columns with data type 'int'
        int_vars = df.select_dtypes(include=['int', 'float']).columns.tolist()
        int_vars = [col for col in int_vars if col != target_variable]

        # Create a figure with subplots
        num_cols = len(int_vars)
        num_rows = (num_cols + 2) // 3  # To make sure there are enough rows for the subplots
        fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5*num_rows))
        axs = axs.flatten()
        
        # Create a histogram for each integer variable with hue='Attrition'
        for i, var in enumerate(int_vars):
            top_categories = df[var].value_counts().nlargest(10).index
            filtered_df = df[df[var].notnull() & df[var].isin(top_categories)]
            sns.histplot(data=df, x=var, hue=target_variable, kde=True, ax=axs[i])
            axs[i].set_title(var)
        
        # Remove any extra empty subplots if needed
        if num_cols < len(axs):
            for i in range(num_cols, len(axs)):
                fig.delaxes(axs[i])
        
        # Adjust spacing between subplots
        fig.tight_layout()
        
        # Show plot
        st.pyplot(fig)
        fig.savefig("plot3.png")

    # Define the paths to the saved plots
    plot_paths = ["plot4.png", "plot7.png", "plot2.png", "plot3.png"]
    existing_plots = []
    for plot_path in plot_paths:
        if os.path.exists(plot_path):
            existing_plots.append(plot_path)
        else:
            print(f"File '{plot_path}' not found.")
    
    if existing_plots:
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 15))
        for i, plot_path in enumerate(existing_plots):
            try:
                row = i // 2
                col = i % 2
                img = plt.imread(plot_path)
                axs[row, col].imshow(img)
                axs[row, col].axis('off')
            except Exception as e:
                print(f"Error while reading '{plot_path}': {e}")
    else:
        print("No plot files found.")

    # Create a new figure
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 15))

    # Iterate over each plot path and place it in the corresponding subplot
    for i, plot_path in enumerate(plot_paths):
        row = i // 2
        col = i % 2
        img = plt.imread(plot_path)
        axs[row, col].imshow(img)
        axs[row, col].axis('off')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Save the merged plot
    fig.savefig("merged_plots.png")

    # Streamed response emulator
    def to_markdown(text):
        text = text.replace('•', '  *')
        return text

    genai.configure()

    img = PIL.Image.open("merged_plots.png")
    model = genai.GenerativeModel('gemini-pro-vision')
    response = model.generate_content(img)

    st.title("Chat with your Data")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask Your Data"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Google Gemini response based on user's question
        img = PIL.Image.open("merged_plots.png")
        model = genai.GenerativeModel('gemini-pro-vision')
        response = model.generate_content([prompt, img], stream=True)
        response.resolve()

        # Format and display the response
        response_text = response.text
        response_markdown = to_markdown(response_text)
        st.write(response_markdown)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response_text})
