# === Core Python ===
import os
import math
import json
import shutil
import base64
import warnings
import logging
from typing import Dict, List

# === Data Handling ===
import pandas as pd
import polars as pl

# === NLP & Text Processing ===
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# === Visualization ===
import matplotlib.pyplot as plt
import missingno as msno
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# === Sklearn ===
from sklearn.impute import KNNImputer

# === PII Detection (Presidio) ===
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

# === Other Libraries ===
from langchain_anthropic import ChatAnthropic
import difflib
import pyfiglet
import getpass

# === Rust compute backend ===
# CPU-bound work (text cleaning, symbol stripping, IQR outliers, skewness,
# correlation/multicollinearity, JSON detection) is delegated to this compiled
# Rust extension for speed and real (GIL-free) parallelism.
from . import _rustcore

# === Setup & Logging Cleanup ===
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)  # required by the WordNet lemmatizer
nltk_logger = logging.getLogger('nltk')
nltk_logger.setLevel(logging.ERROR)


# English stopword list, materialised once and reused by the Rust text backend.
_STOPWORDS = list(stopwords.words("english"))


def _to_f64_list(series):
    """Convert a Polars numeric Series to a list of floats, mapping nulls to NaN
    so the Rust backend (which ignores NaN) sees the same data Polars would."""
    return [float(x) if x is not None else float("nan") for x in series.to_list()]






def print_section_header(title):
    """Prints a formatted section header with dashes."""
    print("\n" + "-" * 100)
    print(f"{title.center(100)}")
    print("-" * 100 + "\n")




def print_header(title):
    """Prints a formatted section header with ASCII art font centered in the terminal."""
    ascii_banner = pyfiglet.figlet_format(title, font="slant")  # Choose a large font
    terminal_width = shutil.get_terminal_size().columns  # Get terminal width

    # Split the banner into lines and center each line
    for line in ascii_banner.split("\n"):
        print(line.center(terminal_width))  



def load_data(file_path):
    """Load data from different formats using polars."""

    print_header("Automated Cleaning")
    print_header("by DataSpoof")
    print_section_header("Loading Data")
    if file_path.endswith('.csv'):
        return pl.read_csv(file_path,infer_schema_length=10000,try_parse_dates=True,ignore_errors=False)
    elif file_path.endswith('.tsv'):
        return pl.read_csv(file_path, separator='\t')
    elif file_path.endswith('.json'):
        return pl.read_json(file_path)
    elif file_path.endswith('.parquet'):
        return pl.read_parquet(file_path)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        raise ValueError("Polars does not support direct Excel file reading. Convert to CSV or Parquet.")
    else:
        raise ValueError("Unsupported file format")




contractions = {"ain't": "am not", "aren't": "are not", "can't": "cannot", 
"can't've": "cannot have", "'cause": "because", "could've": "could have", 
"couldn't": "could not", "couldn't've": "could not have", "didn't": "did not", 
"doesn't": "does not", "don't": "do not", "hadn't": "had not", "hadn't've": "had not have",
"hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'd've": "he would have",
"he'll": "he will", "he's": "he is", "how'd": "how did", "how'll": "how will",
"how's": "how is", "i'd": "i would", "i'll": "i will", "i'm": "i am", "i've": "i have",
"isn't": "is not", "it'd": "it would", "it'll": "it will", "it's": "it is",
"let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have",
"mightn't": "might not", "must've": "must have", "mustn't": "must not",
"needn't": "need not", "oughtn't": "ought not", "shan't": "shall not",
"sha'n't": "shall not", "she'd": "she would", "she'll": "she will", "she's": "she is",
"should've": "should have", "shouldn't": "should not", "that'd": "that would",
"that's": "that is", "there'd": "there had", "there's": "there is", "they'd": "they would",
"they'll": "they will", "they're": "they are", "they've": "they have", "wasn't": "was not",
"we'd": "we would", "we'll": "we will", "we're": "we are", "we've": "we have",
"weren't": "were not", "what'll": "what will", "what're": "what are", "what's": "what is",
"what've": "what have", "where'd": "where did", "where's": "where is", "who'll": "who will",
"who's": "who is", "won't": "will not", "wouldn't": "would not", "you'd": "you would",
"you'll": "you will", "you're": "you are", "wfh": "work from home", "wfo": "work from office",
"idk": "i do not know", "brb": "be right back", "btw": "by the way", "tbh": "to be honest",
"omw": "on my way", "lmk": "let me know", "fyi": "for your information",
"imo": "in my opinion", "smh": "shaking my head", "nvm": "never mind",
"ikr": "i know right", "fr": "for real", "rn": "right now", "gg": "good game",
"dm": "direct message", "afaik": "as far as i know", "bff": "best friends forever",
"ftw": "for the win", "hmu": "hit me up", "ggwp": "good game well played"}





def preprocess_text(text, remove_stopwords=True):
    """Clean a single text value.

    The heavy work (lowercasing, contraction expansion, URL/mention/special-char/
    emoji stripping, tokenization and stopword removal) runs in the Rust backend.
    WordNet lemmatization stays in Python so its output matches the original.
    """
    if text is None or not isinstance(text, str):
        return ""
    stop = _STOPWORDS if remove_stopwords else []
    cleaned = _rustcore.preprocess_texts([text], stop, remove_stopwords)[0]
    lemmatizer = WordNetLemmatizer()
    return " ".join(lemmatizer.lemmatize(word) for word in cleaned.split())




def detect_column_types_and_process_text(df):
    """
    Detect whether string columns in a Polars DataFrame are likely categorical, text, or JSON.
    
    This function categorizes string columns as:
    - 'categorical': Columns with relatively few unique values compared to total rows
    - 'text': Columns with many unique values and longer string lengths
    - 'json': Columns containing valid JSON objects/arrays
    
    Parameters:
    -----------
    df : pl.DataFrame
        The input Polars DataFrame to analyze
        
    Returns:
    --------
    tuple:
        (DataFrame with analysis results, set of categorical columns, set of text columns, set of json columns)
    """
    print_section_header("Checking whether column is string or text or json and peprocess text column")


    # Get only string columns
    string_cols = [col for col in df.columns if df[col].dtype == pl.Utf8]
    
    if not string_cols:
        return (pl.DataFrame({"column": [], "type": [], "unique_ratio": [], "avg_length": []}), 
                set(), set(), set())
    
    results = []
    categorical_cols = set()
    text_cols = set()
    json_cols = set()
    
    # Number of rows in the dataframe
    row_count = df.height
    
    for col in string_cols:
        # Calculate metrics
        unique_count = df[col].n_unique()
        unique_ratio = unique_count / row_count if row_count > 0 else 0
        
        # Calculate average string length using a more reliable approach
        try:
            # Try different string length methods depending on Polars version
            avg_length = df.select(
                pl.mean(pl.col(col).cast(pl.Utf8).str.length()).alias("avg_length")
            ).item()
        except AttributeError:
            try:
                # Alternative approach using string_length expression
                avg_length = df.select(
                    pl.mean(pl.string_length(pl.col(col))).alias("avg_length")
                ).item()
            except:
                # Fallback to a manual calculation if needed
                non_null = df.filter(pl.col(col).is_not_null())
                if non_null.height > 0:
                    avg_length = sum(len(str(x)) for x in non_null[col].to_list()) / non_null.height
                else:
                    avg_length = 0
        
        # Check if column contains JSON
        is_json = False
        json_sample_count = min(30, df.height)  # Check up to 30 rows for efficiency
        
        if json_sample_count > 0:
            # Take a sample of non-null values
            sample_values = df.filter(pl.col(col).is_not_null()).head(json_sample_count)[col].to_list()
            
            # JSON detection heuristic (Rust backend): >=50% of sampled non-null
            # values both look like a JSON object/array and parse successfully.
            is_json = _rustcore.detect_json_sample([str(v).strip() for v in sample_values])
        
        # Determine column type
        if is_json:
            col_type = "json"
            json_cols.add(col)
        elif unique_ratio < 0.2 or (unique_count < 50 and avg_length < 20):
            col_type = "categorical"
            categorical_cols.add(col)
        else:
            col_type = "text"
            text_cols.add(col)
            # Batch the whole column through the Rust backend in one call
            # (parallel, GIL released), then lemmatize in Python for parity.
            cleaned = _rustcore.preprocess_texts(df[col].to_list(), _STOPWORDS, True)
            lemmatizer = WordNetLemmatizer()
            processed = [" ".join(lemmatizer.lemmatize(w) for w in c.split()) for c in cleaned]
            df = df.with_columns(pl.Series(col, processed))
 
            
        results.append({
            "column": col,
            "type": col_type,
            "unique_ratio": round(unique_ratio, 3),
            "avg_length": round(avg_length, 1) if avg_length is not None else None,
            "unique_count": unique_count,
            "is_json": is_json
        })
    
    result_df = pl.DataFrame(results)
    
    # Print the sets in the desired format
    print(f"Categorical columns are {categorical_cols}")
    print(f"Text columns are {text_cols}") 
    print(f"Json columns are {json_cols}")
    
    return df













def check_data_types(df):
    """Check the data types of columns."""
    print_section_header("Checking Data Types")
    print(df.dtypes)
    return df.dtypes

def replace_symbols_and_convert_to_float(df):
    """Replace symbols and convert to float using Polars."""
    print_section_header("Handling Symbols & Conversion")
    
    # Identify columns containing unwanted symbols
    problematic_cols = [
        col for col in df.columns 
        if df[col].cast(pl.Utf8, strict=False).str.contains(r'[\$,₹,-]', literal=False).any()
    ]
    
    if problematic_cols:
        print("Columns containing $, ₹, -, or , before processing:", problematic_cols)

    # Replace symbols and convert to float
    df = df.with_columns([
        pl.col(col)
        .str.replace_all(r'[\$,₹,-]', '')  # Remove $, ₹, - symbols
        .cast(pl.Float64, strict=False)  # Convert to float (coerce errors)
        .alias(col)
        for col in problematic_cols
    ])
    
    return df



def replace_symbols(df):
    """Strip currency/separator symbols ($ ₹ , - •) from string columns.

    Symbol removal runs in the Rust backend (parallel, per value)."""

    # Identify columns containing unwanted symbols
    problematic_cols = [
        col for col in df.columns
        if df[col].cast(pl.Utf8, strict=False).str.contains(r'[\$,₹,-]', literal=False).any()
    ]

    for col in problematic_cols:
        values = df[col].cast(pl.Utf8, strict=False).to_list()
        stripped = _rustcore.strip_symbols(values)  # removes $ ₹ , - •
        df = df.with_columns(pl.Series(col, stripped))

    return df



def fix_incorrect_data_types(df):
    for col in df.columns:
        try:
            # Attempt to convert column to numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except Exception as e:
            # Log any errors for debugging purposes
            print(f"Could not convert column {col} due to: {e}")
    return df


def fix_spelling_errors_in_columns(df):
    """Fix spelling errors in column names by interacting with the user."""
    
    print("Checking for spelling errors in column names:")
    for idx, col in enumerate(df.columns, start=1):
        print(f"{idx}. {col}")

    print("\n" + "-" * 40)
    
    incorrect_columns = input("Enter the index of the columns with incorrect spelling (comma-separated), or press Enter to skip: ").strip()

    if not incorrect_columns:
        print("No changes made.")
        return df

    incorrect_columns = incorrect_columns.split(',')
    incorrect_columns = [df.columns[int(i.strip()) - 1] for i in incorrect_columns if i.strip().isdigit()]

    corrected_columns = {}
    
    for col in incorrect_columns:
        suggestion = difflib.get_close_matches(col, df.columns, n=1, cutoff=0.8)
        if suggestion and suggestion[0] != col:
            print(f"Suggested correction for '{col}': {suggestion[0]}")

        correct_spelling = input(f"Enter the correct spelling for '{col}' (or press Enter to keep it unchanged): ").strip()
        
        if correct_spelling:
            corrected_columns[col] = correct_spelling

    # Rename columns
    df = df.rename(corrected_columns)
    
    print("Updated Column Names:")
    print("-" * 40)
    print(df.columns)
    
    return df

# Initialize logging
logging.basicConfig(filename="corrections.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# System prompt for categorical value correction
CATEGORICAL_CORRECTION_PROMPT = """
You are a data validation expert. Your task is to correct spelling errors and inconsistencies in categorical values.
- Ensure all values are standardized and correctly spelled.
- Return the corrected values **in the same order** as provided.
- Use a bullet-point format (one value per line).

Example input:
Column: ProductCategory
Values: ['elecronics', 'fashon', 'Electronics', 'fasihon', 'home_appl']

Expected output:
- Electronics
- Fashion
- Electronics
- Fashion
- Home Appliances

Example input:
Column: ProductCategory
Values: ['good', 'bad', 'goo', 'ba']

Expected output:
- good
- bad
- good
- bad

"""



def fix_spelling_errors_in_categorical(df):
    """Fix spelling errors in categorical columns using Claude AI or manual input."""
    print_section_header("Fix spelling errors in categorical columns")
    
    user_choice = input("Do you want to correct spelling errors in categorical columns? (yes/no): ").strip().lower()
    if user_choice not in ["yes", "y"]:
        print("Skipping spell-checking.")
        return df

    method_choice = input("Choose correction method: (1) Automatic (Claude AI) (2) Manual: ").strip()
    
    if method_choice == "1":
        api_key = getpass.getpass("Enter your Claude API key: ")
        model = ChatAnthropic(model="claude-3-7-sonnet-latest", api_key=api_key)
    else:
        model = None  # No AI model needed for manual correction

    categorical_columns = [col for col in df.columns if df[col].dtype == pl.Utf8]

    for col in categorical_columns:
        unique_values = df[col].drop_nulls().unique().to_list()
        
        if method_choice == "1":
            corrected_values = correct_categorical_values(model, col, unique_values)
        else:
            corrected_values = manual_correction(col, unique_values)
        
        correction_map = dict(zip(unique_values, corrected_values))
        
        print(f"\nColumn: {col}\nOriginal Values: {unique_values}\nCorrected Values: {corrected_values}")

        # Log corrections
        for old_value, new_value in correction_map.items():
            if old_value != new_value:
                logging.info(f"Column: {col} | '{old_value}' -> '{new_value}'")

        df = df.with_columns(pl.col(col).replace(correction_map).alias(col))
    
    return df


def correct_categorical_values(model, column_name: str, values: list) -> list:
    """Corrects categorical column values using Claude AI."""
    text = f"Column: {column_name}\nValues: {', '.join(values)}"
    model_input = [
        {"role": "system", "content": CATEGORICAL_CORRECTION_PROMPT},
        {"role": "user", "content": text},
    ]
    response = model.invoke(model_input)

    # Ensure response is properly formatted and split into a clean list
    corrected_values = response.content.strip().split("\n")

    # If response length is incorrect, return original values
    if len(corrected_values) != len(values):
        logging.warning(f"Unexpected response format for column '{column_name}'. Received: {response.content}")
        return values  # Return original values if there's an issue

    return corrected_values


def manual_correction(column_name: str, values: list) -> list:
    """Allows user to manually correct categorical values."""
    corrected_values = []
    print(f"\nManual correction for column: {column_name}")
    
    for val in values:
        new_val = input(f"Correct '{val}' (press enter to keep it unchanged): ").strip()
        corrected_values.append(new_val if new_val else val)
    
    return corrected_values



def handle_negative_values(df):
    """Handle negative values by printing column names with negatives and replacing them with absolute values."""
    print_section_header("Checking for Negative Values")

    # Identify numerical columns in Polars
    numeric_cols = [col for col, dtype in df.schema.items() if dtype in [pl.Float64, pl.Int64, pl.Int32, pl.Float32]]

    # Iterate over numerical columns and check for negatives (Rust backend)
    for col in numeric_cols:
        if _rustcore.has_negative(_to_f64_list(df[col])):
            print(f"Column '{col}' contains negative values.")

            # Replace negative values with their absolute counterparts
            df = df.with_columns(pl.col(col).abs().alias(col))

    return df



def handle_missing_values(df):
    """Handle missing values with visualization."""
    print_section_header("Checking for missing values and fixing it")
    missing_columns = [col for col in df.columns if df[col].null_count() > 0]
    
    if missing_columns:
        print("Columns containing missing values:", missing_columns)
    else:
        print("No missing values found.")
    
    plt.figure(figsize=(10, 6))
    msno.bar(df.to_pandas())

    # Select only numeric columns
    num_df = df.select(pl.col(pl.Float64, pl.Int64))
    
    # Convert to NumPy
    num_array = num_df.to_numpy()

    # Check shape before applying imputer
    print(f"Shape of num_df: {num_array.shape}")

    imputer = KNNImputer(n_neighbors=5)
    imputed_values = imputer.fit_transform(num_array)

    # Check shape after imputation
    print(f"Shape after imputation: {imputed_values.shape}")

    # Ensure we don't go out of bounds
    min_columns = min(len(num_df.columns), imputed_values.shape[1])

    df = df.with_columns([pl.Series(num_df.columns[i], imputed_values[:, i]) for i in range(min_columns)])

    categorical_cols = df.select(pl.col(pl.Utf8, pl.Categorical)).columns
    for col in categorical_cols:
        mode_value = df[col].mode().to_list()[0]
        df = df.with_columns(df[col].fill_null(mode_value))
    
    return df


def handle_duplicates(df):
    """Handle duplicate records."""
    print_section_header("Checking for duplicate values and fixing it")
    duplicate_count = df.is_duplicated().sum()
    if duplicate_count > 0:
        print(f"Duplicate rows found: {duplicate_count}. Dropping duplicates...")
        df = df.unique()
    else:
        print("No duplicate rows found.")
    return df

def check_outliers(df):
    """Check for outliers using IQR method."""
    numerical_cols = df.select(pl.col(pl.Float64, pl.Int64)).columns
    outliers = {}
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers[col] = df.filter((df[col] < lower_bound) | (df[col] > upper_bound))
    return outliers

def remove_outliers(df):
    """Remove outliers using the IQR method for numerical columns (Rust backend)."""
    numerical_cols = df.select(pl.col(pl.Float64, pl.Int64)).columns
    for col in numerical_cols:
        keep_mask = _rustcore.iqr_keep_mask(_to_f64_list(df[col]))
        df = df.filter(pl.Series(keep_mask))
    return df



def check_problem_type(df, target_col):
    """Determine whether the problem is classification or regression."""
    unique_values = df[target_col].n_unique()
    
    if df[target_col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64] and unique_values > 10:
        print(f"Detected a regression problem (continuous target column '{target_col}'). Skipping class balancing.")
        return "regression"
    else:
        print(f"Detected a classification problem (categorical/discrete target column '{target_col}').")
        return "classification"

def check_and_handle_imbalance(df, target_col):
    """
    Check for class imbalance and handle it using user-selected undersampling or oversampling.
    
    Parameters:
    - df (pl.DataFrame): The input Polars dataframe
    - target_col (str): The name of the target column
    
    Returns:
    - pl.DataFrame: A balanced dataframe or the original if skipped
    """
    
    if check_problem_type(df, target_col) == "regression":
        return df  # Skip class balancing if it's a regression problem
    
    # Original Class Distribution
    class_counts = df[target_col].value_counts().sort("count")
    min_count, max_count = class_counts["count"].min(), class_counts["count"].max()
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    print("Original Class Distribution:")
    print(class_counts)

    if imbalance_ratio > 1.5:
        print(f"\nThe target column '{target_col}' is **imbalanced**.")
        
        # Ask the user for input
        method = input("\nChoose a balancing method - 'oversampling', 'undersampling', or press Enter to skip: ").strip().lower()
        
        if method == "":
            print("\nSkipping class balancing.")
            return df

        balanced_df = []

        if method == "oversampling":
            max_samples = max_count
            for label in class_counts[target_col].to_list():
                subset = df.filter(df[target_col] == label)
                additional_samples = subset.sample(n=max_samples - len(subset), with_replacement=True)
                balanced_df.append(subset)
                balanced_df.append(additional_samples)

        elif method == "undersampling":
            min_samples = min_count
            for label in class_counts[target_col].to_list():
                subset = df.filter(df[target_col] == label)
                subset = subset.sample(n=min_samples)  # Undersample to match the smallest class
                balanced_df.append(subset)

        else:
            raise ValueError("\nInvalid method. Please choose 'oversampling', 'undersampling', or press Enter to skip.")

        # Combine all balanced samples and shuffle
        df = pl.concat(balanced_df).sample(fraction=1.0, shuffle=True)

        # Print New Class Distribution
        print("\nBalanced Class Distribution:")
        print(df[target_col].value_counts().sort("count"))

    return df

def check_skewness(df):
    """Check skewness in numerical columns."""
    return df.select(pl.col(pl.Float64, pl.Int64)).skew()

def fix_skewness(df):
    """Fix skewness using log transformation (skewness computed in Rust)."""
    numerical_cols = df.select(pl.col(pl.Float64, pl.Int64)).columns
    for col in numerical_cols:
        if _rustcore.skewness(_to_f64_list(df[col])) > 1:
            df = df.with_columns((df[col] + 1).log().alias(col))
    return df

def check_multicollinearity(df, threshold=0.7):
    """Check for multicollinearity and drop highly correlated features.

    The correlation matrix and drop-list are computed in the Rust backend."""
    num_df = df.select(pl.col(pl.Float64, pl.Int64))
    names = num_df.columns

    to_drop = []
    if len(names) >= 2:
        columns = [_to_f64_list(num_df[c]) for c in names]
        to_drop = _rustcore.collinear_to_drop(columns, names, threshold)

    if to_drop:
        print(f"Dropping highly correlated columns: {', '.join(to_drop)}")
        df = df.drop(to_drop)
    else:
        print("No highly correlated features found above the threshold.")

    return df



def check_cardinality(df: pl.DataFrame):
    """
    Check the cardinality (number of unique values) of categorical columns and remove columns with only one unique value.

    Parameters:
        df (pl.DataFrame): The input DataFrame.

    Returns:
        pl.DataFrame: The DataFrame after removing low-cardinality columns.
        dict: A dictionary with column names as keys and their cardinality as values.
    """
    print_section_header("Checking for Cardinality")
    
    # Select categorical columns
    categorical_cols = [col for col in df.columns if df[col].dtype in [pl.Utf8, pl.Categorical]]
    print(f"Categorical columns found: {categorical_cols}")
    
    # Calculate cardinality
    cardinality = {col: df[col].n_unique() for col in categorical_cols}
    print(f"Cardinality of categorical columns:\n{cardinality}")
    
    # Remove columns with only one unique value
    low_cardinality_cols = [col for col, count in cardinality.items() if count == 1]
    if low_cardinality_cols:
        print(f"\nRemoving columns with only one unique value: {low_cardinality_cols}")
        df = df.drop(low_cardinality_cols)
    else:
        print("\nNo columns removed because there are no low cardinality columns")
    
    return df, cardinality

def save_cleaned_data(df: pl.DataFrame, file_name="cleaned_data.csv", quantize=True):
    """
    Save cleaned DataFrame to a CSV file with optional quantization for float and integer columns.
    
    Parameters:
    - df (pl.DataFrame): The input Polars dataframe.
    - file_name (str): The name of the CSV file.
    - quantize (bool): Whether to quantize numeric columns (default: True).
    
    Returns:
    - None
    """
    
    print("\n🔹 Saving cleaned data...")

    if quantize:
        # Convert float64 -> float32, int64 -> int32 to reduce size
        float_cols = [col for col, dtype in zip(df.columns, df.dtypes) if dtype in [pl.Float64, pl.Float32]]
        int_cols = [col for col, dtype in zip(df.columns, df.dtypes) if dtype in [pl.Int64, pl.Int32]]

        df = df.with_columns([df[col].cast(pl.Float32) for col in float_cols])
        df = df.with_columns([df[col].cast(pl.Int32) for col in int_cols])

    # Save to CSV
    df.write_csv(file_name)
    print(f"✅ Cleaned data saved to {file_name}")
    return df


def save_dashboard(html_content, filename="dashboard.html"):
    OUTPUT_DIR = "output/eda/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, filename), "w", encoding="utf-8") as f:
        f.write(html_content)

def generate_dashboard(df: pl.DataFrame, background_image_path: str = None):
    """Render an interactive, enterprise-style EDA dashboard.

    Writes ``output/eda/dashboard.html`` with a collapsible sidebar (Overview /
    Univariate / Bivariate / Multivariate), a hamburger toggle, KPI cards, an
    interactive data-quality panel, a data preview, and Plotly charts. The
    Univariate tab has a column picker; the Bivariate tab has X / Y / Color
    pickers that plot the chosen pair on demand. Charts lazy-render per section.
    """
    if df is None:
        raise ValueError(
            "generate_dashboard received None. Pass a DataFrame, e.g. the value "
            "returned by clean_data(df) or load it with load_data('cleaned_data.csv')."
        )

    df_pandas = df.to_pandas()
    num_cols = df.select(pl.col(pl.Float64, pl.Int64)).columns
    cat_cols = df.select(pl.col(pl.Utf8)).columns
    all_cols = num_cols + cat_cols
    PALETTE = ["#3b82f6", "#22d3ee", "#a78bfa", "#f472b6", "#fbbf24", "#34d399", "#f87171"]

    def esc(s):
        return (str(s).replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;").replace('"', "&quot;"))

    def f64(series):
        return [float(x) if x is not None else float("nan") for x in series.to_list()]

    chart_json = []  # (cid, section, plotly_json_str)

    def _register(fig, section, tall=False, col=None):
        fig.update_layout(
            template="plotly_dark", autosize=True,
            margin=dict(l=48, r=20, t=52, b=42),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1", size=12),
            title=dict(font=dict(size=14, color="#e5e7eb")),
            colorway=PALETTE, legend=dict(bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor="rgba(148,163,184,0.12)"),
            yaxis=dict(gridcolor="rgba(148,163,184,0.12)"),
        )
        cid = "c%d" % len(chart_json)
        chart_json.append((cid, section, fig.to_json()))
        dc = (' data-col="%s"' % esc(col)) if col is not None else ""
        return '<div class="card"%s><div id="%s" class="plot%s"></div></div>' % (
            dc, cid, " tall" if tall else "")

    # ---- univariate charts (one card per column, tagged for filtering) -------
    uni_cards = []
    for col in cat_cols:
        vc = df_pandas[col].value_counts().reset_index()
        vc.columns = [col, "count"]
        uni_cards.append(_register(px.bar(vc.head(20), x=col, y="count",
                                          title=f"{col} — distribution"), "univariate", col=col))
    for col in num_cols:
        uni_cards.append(_register(px.histogram(df_pandas, x=col, title=f"{col} — histogram"),
                                   "univariate", col=col))
        uni_cards.append(_register(px.box(df_pandas, y=col, title=f"{col} — box plot"),
                                   "univariate", col=col))
    uni_grid = '<div class="chart-grid" id="uni-grid">' + "".join(uni_cards) + '</div>'

    # ---- multivariate (correlation heatmap) ----------------------------------
    figures_multivariate = []
    if len(num_cols) > 1:
        corr = df_pandas[num_cols].corr()
        hm = go.Figure(data=go.Heatmap(
            z=corr.values, x=list(corr.columns), y=list(corr.index),
            colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
            text=corr.round(2).values, texttemplate="%{text}",
            hovertemplate="%{x} vs %{y}: %{z:.2f}<extra></extra>"))
        hm.update_layout(title="Correlation heatmap")
        figures_multivariate.append(hm)
    multi_body = ('<div class="chart-grid single">'
                  + "".join(_register(f, "multivariate", tall=True) for f in figures_multivariate)
                  + '</div>') if figures_multivariate else \
        '<p class="empty">Need at least two numeric columns for a correlation heatmap.</p>'

    # ---- embedded data for on-demand bivariate plotting ----------------------
    cap = 5000
    sub = df_pandas.head(cap)
    DATA = {}
    for c in num_cols:
        DATA[c] = [None if (v is None or (isinstance(v, float) and v != v)) else float(v)
                   for v in sub[c].tolist()]
    for c in cat_cols:
        DATA[c] = [None if v is None else str(v) for v in sub[c].tolist()]
    COLTYPES = {**{c: "num" for c in num_cols}, **{c: "cat" for c in cat_cols}}
    data_js = json.dumps(DATA).replace("</", "<\\/")
    coltypes_js = json.dumps(COLTYPES).replace("</", "<\\/")

    def _opts(cols, selected=None):
        return "".join('<option value="%s"%s>%s</option>'
                       % (esc(c), " selected" if c == selected else "", esc(c)) for c in cols)

    bi_x_default = num_cols[0] if num_cols else (all_cols[0] if all_cols else "")
    bi_y_default = (num_cols[1] if len(num_cols) > 1 else
                    (cat_cols[0] if cat_cols else (all_cols[0] if all_cols else "")))

    uni_controls = (
        '<div class="controls"><div class="ctrl"><label>Column</label>'
        '<select id="uni-select" onchange="filterUni(this.value)">'
        '<option value="__all__">All columns</option>' + _opts(all_cols) + '</select></div></div>')
    uni_body = uni_controls + uni_grid

    bi_body = (
        '<div class="controls">'
        '<div class="ctrl"><label>X axis</label><select id="bi-x">' + _opts(all_cols, bi_x_default) + '</select></div>'
        '<div class="ctrl"><label>Y axis</label><select id="bi-y">' + _opts(all_cols, bi_y_default) + '</select></div>'
        '<div class="ctrl"><label>Color / group</label><select id="bi-c">'
        '<option value="__none__">None</option>' + _opts(cat_cols) + '</select></div>'
        '<button class="btn" onclick="plotBi()">Plot</button></div>'
        '<div class="card"><div id="bi-plot" class="plot tall"></div></div>')

    # ---- KPIs ----------------------------------------------------------------
    n = df.height or 1
    total_cells = (df.height * df.width) or 1
    missing_cells = sum(df[c].null_count() for c in df.columns)
    try:
        dup_rows = int(df.is_duplicated().sum())
    except Exception:
        dup_rows = 0
    kpis = [
        ("Rows", f"{df.height:,}"), ("Columns", f"{df.width:,}"),
        ("Numeric", f"{len(num_cols)}"), ("Categorical", f"{len(cat_cols)}"),
        ("Missing", f"{missing_cells / total_cells:.1%}"), ("Duplicates", f"{dup_rows:,}"),
    ]
    kpi_html = "".join(
        '<div class="kpi"><div class="kpi-val">' + v + '</div><div class="kpi-lbl">' + k + '</div></div>'
        for k, v in kpis)

    # ---- data-quality issue detection ---------------------------------------
    issues = []
    for c in df.columns:
        nc = df[c].null_count()
        if nc > 0:
            pct = nc / n
            sev = "high" if pct > 0.3 else ("medium" if pct > 0.05 else "low")
            issues.append((sev, f"Missing values · {c}", f"{pct:.1%}",
                           f"{nc:,} of {n:,} rows are null in “{c}”."))
    if dup_rows > 0:
        sev = "high" if dup_rows / n > 0.05 else "medium"
        issues.append((sev, "Duplicate rows", f"{dup_rows:,}",
                       f"{dup_rows:,} fully duplicated rows — consider dropping them."))
    for c in df.columns:
        if df[c].n_unique() <= 1:
            issues.append(("medium", f"Constant column · {c}", "1 value",
                           f"“{c}” has a single unique value and carries no information."))
    for c in df.columns:
        ratio = df[c].n_unique() / n
        if ratio > 0.95 and df.height > 10:
            issues.append(("low", f"Possible identifier · {c}", f"{ratio:.0%} unique",
                           f"“{c}” is nearly unique per row — likely an ID with no predictive value."))
    for c in num_cols:
        vals = [v for v in df[c].to_list() if v is not None]
        if len(vals) >= 8:
            lo, hi = _rustcore.iqr_bounds(f64(df[c]))
            out = sum(1 for v in vals if v < lo or v > hi)
            if out > 0:
                pct = out / len(vals)
                sev = "high" if pct > 0.1 else ("medium" if pct > 0.02 else "low")
                issues.append((sev, f"Outliers · {c}", f"{out:,}",
                               f"{out:,} values fall outside the IQR fences [{lo:.2f}, {hi:.2f}]."))
    for c in num_cols:
        if _rustcore.has_negative(f64(df[c])):
            issues.append(("low", f"Negative values · {c}", "present",
                           f"“{c}” contains negative values — verify whether that is expected."))
    for c in num_cols:
        sk = _rustcore.skewness(f64(df[c]))
        if sk == sk and abs(sk) > 1:
            issues.append(("low", f"Skewed distribution · {c}", f"skew {sk:.1f}",
                           f"“{c}” is highly skewed — a log transform may help."))
    for c in cat_cols:
        raw = df[c].to_list()
        cleaned = _rustcore.normalize_whitespace([str(v) if v is not None else None for v in raw])
        bad = sum(1 for a, b in zip(raw, cleaned) if a is not None and str(a) != b)
        if bad > 0:
            issues.append(("low", f"Whitespace issues · {c}", f"{bad:,}",
                           f"{bad:,} values in “{c}” have leading/trailing or repeated whitespace."))
    for c in cat_cols:
        nn = df[c].drop_nulls()
        u = nn.n_unique()
        lu = nn.str.to_lowercase().n_unique()
        if u > lu:
            issues.append(("medium", f"Inconsistent casing · {c}", f"{u - lu} merge(s)",
                           f"{u} distinct values in “{c}” collapse to {lu} when lowercased "
                           f"(values that differ only in letter case)."))

    sev_rank = {"high": 0, "medium": 1, "low": 2}
    issues.sort(key=lambda x: sev_rank[x[0]])
    cH = sum(1 for i in issues if i[0] == "high")
    cM = sum(1 for i in issues if i[0] == "medium")
    cL = sum(1 for i in issues if i[0] == "low")
    if issues:
        items = ""
        for sev, title, metric, detail in issues:
            items += (
                '<div class="q-item %s" data-sev="%s" onclick="this.classList.toggle(\'open\')">'
                '<div class="q-row"><span class="q-dot"></span>'
                '<span class="q-title">%s</span><span class="q-badge">%s</span>'
                '<span class="q-caret">&#9662;</span></div>'
                '<div class="q-detail">%s</div></div>'
            ) % (sev, sev, esc(title), esc(metric), esc(detail))
        chips = (
            '<button class="chip active" onclick="filterQ(\'all\',this)">All %d</button>'
            '<button class="chip" onclick="filterQ(\'high\',this)"><span class="sd high"></span>High %d</button>'
            '<button class="chip" onclick="filterQ(\'medium\',this)"><span class="sd medium"></span>Medium %d</button>'
            '<button class="chip" onclick="filterQ(\'low\',this)"><span class="sd low"></span>Low %d</button>'
        ) % (len(issues), cH, cM, cL)
        quality_html = (
            '<div class="panel"><div class="q-head"><h3>Data quality issues '
            '<span>(%d found)</span></h3><div class="q-filters">%s</div></div>'
            '<div class="q-list">%s</div></div>') % (len(issues), chips, items)
    else:
        quality_html = ('<div class="panel ok"><h3>&#10003; No major data quality '
                        'issues detected</h3></div>')

    preview_html = df_pandas.head(8).to_html(index=False, border=0, classes="preview")
    overview_body = (
        '<div class="kpi-row">' + kpi_html + '</div>' + quality_html +
        '<div class="panel"><h3>Data preview <span>(first 8 rows)</span></h3>'
        '<div class="table-wrap">' + preview_html + '</div></div>')

    topbar_bg = ""
    if background_image_path:
        try:
            with open(background_image_path, "rb") as fh:
                enc = base64.b64encode(fh.read()).decode()
            topbar_bg = ("background-image:linear-gradient(rgba(18,21,29,.85),rgba(18,21,29,.85)),"
                         "url('data:image/png;base64," + enc + "');background-size:cover;")
        except Exception:
            topbar_bg = ""

    specs_js = "{" + ",".join('"%s":%s' % (cid, js) for cid, _, js in chart_json) + "}"
    secmap_js = "{" + ",".join('"%s":"%s"' % (cid, sec) for cid, sec, _ in chart_json) + "}"

    css = """
:root{--bg:#0f1116;--panel:#161a22;--card:#1a1f2b;--border:#232a37;--text:#e5e7eb;--muted:#9aa4b2;--accent:#3b82f6;}
*{box-sizing:border-box}
body{margin:0;font-family:'Segoe UI',system-ui,Roboto,Helvetica,Arial,sans-serif;background:var(--bg);color:var(--text);}
.app{display:flex;min-height:100vh;}
.sidebar{width:232px;background:#12151d;border-right:1px solid var(--border);padding:18px 12px;transition:width .22s ease;overflow:hidden;position:sticky;top:0;height:100vh;}
.app.collapsed .sidebar{width:66px;}
.brand{display:flex;align-items:center;gap:10px;padding:6px 8px 20px;font-weight:700;font-size:18px;white-space:nowrap;}
.brand .logo{color:var(--accent);font-size:20px;}
.app.collapsed .brand-name{display:none;}
nav{display:flex;flex-direction:column;gap:4px;}
.nav-item{display:flex;align-items:center;gap:13px;padding:11px 12px;border-radius:10px;color:var(--muted);cursor:pointer;text-decoration:none;font-size:14px;white-space:nowrap;transition:background .15s,color .15s;}
.nav-item:hover{background:#1b2230;color:var(--text);}
.nav-item.active{background:var(--accent);color:#fff;}
.nav-item .ico{width:18px;text-align:center;font-size:15px;flex:none;}
.app.collapsed .nav-item .txt{display:none;}
.main{flex:1;min-width:0;display:flex;flex-direction:column;}
.topbar{display:flex;align-items:center;gap:16px;padding:16px 24px;border-bottom:1px solid var(--border);background:#12151d;position:sticky;top:0;z-index:10;}
.toggle{background:#1b2230;border:1px solid var(--border);color:var(--text);width:40px;height:40px;border-radius:10px;font-size:18px;cursor:pointer;flex:none;}
.toggle:hover{background:#232c3d;}
.title{font-size:18px;font-weight:700;}
.subtitle{font-size:13px;color:var(--muted);margin-top:2px;}
.content{padding:22px 24px;max-width:1600px;}
.section{display:none;}
.section.active{display:block;animation:fade .25s ease;}
@keyframes fade{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
h2{font-size:19px;margin:2px 0 18px;}
.kpi-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px;margin-bottom:22px;}
.kpi{background:linear-gradient(160deg,#1b2230,#161a22);border:1px solid var(--border);border-radius:14px;padding:18px;}
.kpi-val{font-size:26px;font-weight:700;color:#fff;}
.kpi-lbl{font-size:12px;color:var(--muted);margin-top:4px;text-transform:uppercase;letter-spacing:.05em;}
.panel{background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:18px;margin-bottom:22px;}
.panel h3{margin:0 0 14px;font-size:15px;}
.panel h3 span{color:var(--muted);font-weight:400;font-size:13px;}
.panel.ok h3{color:#34d399;margin:0;}
.table-wrap{overflow-x:auto;}
table.preview{border-collapse:collapse;width:100%;font-size:13px;}
table.preview th{background:#1b2230;color:#cbd5e1;text-align:left;padding:9px 12px;position:sticky;top:0;}
table.preview td{padding:8px 12px;border-top:1px solid var(--border);color:#c7cdd8;white-space:nowrap;}
table.preview tr:hover td{background:#171c26;}
.controls{display:flex;gap:16px;flex-wrap:wrap;align-items:flex-end;margin-bottom:18px;background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:14px 16px;}
.ctrl{display:flex;flex-direction:column;gap:6px;}
.ctrl label{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;}
.controls select{background:#12151d;border:1px solid var(--border);color:var(--text);padding:9px 12px;border-radius:9px;font-size:14px;min-width:170px;}
.btn{background:var(--accent);border:none;color:#fff;padding:10px 20px;border-radius:9px;font-size:14px;cursor:pointer;height:39px;}
.btn:hover{background:#2f6fd6;}
.q-head{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;}
.q-filters{display:flex;gap:8px;flex-wrap:wrap;}
.chip{display:inline-flex;align-items:center;gap:7px;background:#1b2230;border:1px solid var(--border);color:var(--muted);padding:6px 13px;border-radius:20px;font-size:12px;cursor:pointer;transition:.15s;}
.chip:hover{color:var(--text);}
.chip.active{background:var(--accent);color:#fff;border-color:var(--accent);}
.sd{width:8px;height:8px;border-radius:50%;display:inline-block;}
.sd.high{background:#f87171;}.sd.medium{background:#fbbf24;}.sd.low{background:#38bdf8;}
.q-list{display:flex;flex-direction:column;gap:8px;margin-top:14px;}
.q-item{border:1px solid var(--border);border-left:3px solid var(--muted);border-radius:10px;background:#141922;cursor:pointer;overflow:hidden;transition:border-color .15s;}
.q-item:hover{border-color:#2d3646;}
.q-item.high{border-left-color:#f87171;}.q-item.medium{border-left-color:#fbbf24;}.q-item.low{border-left-color:#38bdf8;}
.q-row{display:flex;align-items:center;gap:12px;padding:12px 14px;}
.q-dot{width:9px;height:9px;border-radius:50%;background:var(--muted);flex:none;}
.q-item.high .q-dot{background:#f87171;}.q-item.medium .q-dot{background:#fbbf24;}.q-item.low .q-dot{background:#38bdf8;}
.q-title{flex:1;font-size:14px;color:var(--text);}
.q-badge{background:#232c3d;color:#cbd5e1;padding:3px 11px;border-radius:20px;font-size:12px;white-space:nowrap;}
.q-caret{color:var(--muted);transition:transform .2s;flex:none;}
.q-item.open .q-caret{transform:rotate(180deg);}
.q-detail{max-height:0;opacity:0;padding:0 14px;color:var(--muted);font-size:13px;transition:max-height .2s ease,opacity .2s ease,padding .2s ease;}
.q-item.open .q-detail{max-height:160px;opacity:1;padding:0 14px 13px 38px;}
.chart-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(440px,1fr));gap:16px;}
.chart-grid.single{grid-template-columns:1fr;}
.card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:8px;box-shadow:0 1px 3px rgba(0,0,0,.3);transition:transform .15s,box-shadow .15s;}
.card:hover{transform:translateY(-2px);box-shadow:0 10px 26px rgba(0,0,0,.35);}
.plot{height:360px;width:100%;}
.plot.tall{height:560px;}
.empty{color:var(--muted);padding:44px;text-align:center;border:1px dashed var(--border);border-radius:14px;}
@media(max-width:760px){.sidebar{position:fixed;z-index:60;left:0;}.app.collapsed .sidebar{width:0;padding:0;border:none;}.content{padding:16px;}}
"""

    js = """
var SPECS=__SPECS__, SECMAP=__SECMAP__, RENDERED={};
var DATA=__DATA__, COLTYPES=__COLTYPES__, CO=["#3b82f6","#22d3ee","#a78bfa","#f472b6","#fbbf24","#34d399","#f87171"];
var biInited=false;
function renderSection(sec){
  Object.keys(SECMAP).forEach(function(cid){
    if(SECMAP[cid]===sec && !RENDERED[cid] && document.getElementById(cid)){
      var s=SPECS[cid];
      Plotly.newPlot(cid, s.data, s.layout, {responsive:true, displayModeBar:false});
      RENDERED[cid]=1;
    }
  });
}
function showSection(id, el){
  var s=document.querySelectorAll('.section');for(var i=0;i<s.length;i++){s[i].classList.remove('active');}
  document.getElementById(id).classList.add('active');
  var n=document.querySelectorAll('.nav-item');for(var j=0;j<n.length;j++){n[j].classList.remove('active');}
  if(el){el.classList.add('active');}
  renderSection(id);
  if(id==='bivariate' && !biInited){biInited=true; plotBi();}
  setTimeout(function(){window.dispatchEvent(new Event('resize'));},80);
}
function toggleSidebar(){
  document.getElementById('app').classList.toggle('collapsed');
  setTimeout(function(){window.dispatchEvent(new Event('resize'));},240);
}
function filterQ(sev, el){
  var chips=el.parentNode.querySelectorAll('.chip');for(var i=0;i<chips.length;i++){chips[i].classList.remove('active');}
  el.classList.add('active');
  var items=document.querySelectorAll('.q-item');
  for(var j=0;j<items.length;j++){items[j].style.display=(sev==='all'||items[j].dataset.sev===sev)?'':'none';}
}
function filterUni(v){
  var cards=document.querySelectorAll('#uni-grid .card');
  for(var i=0;i<cards.length;i++){cards[i].style.display=(v==='__all__'||cards[i].dataset.col===v)?'':'none';}
  setTimeout(function(){window.dispatchEvent(new Event('resize'));},60);
}
function _uniq(a){var o=[],s={};for(var i=0;i<a.length;i++){var v=a[i];if(v!==null&&v!==undefined&&!s[v]){s[v]=1;o.push(v);}}return o;}
function _baseLayout(t,xt,yt){return {title:t,template:'plotly_dark',paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{color:'#cbd5e1',size:12},margin:{l:55,r:20,t:52,b:48},colorway:CO,xaxis:{title:xt,gridcolor:'rgba(148,163,184,0.12)'},yaxis:{title:yt,gridcolor:'rgba(148,163,184,0.12)'}};}
function plotBi(){
  var x=document.getElementById('bi-x').value, y=document.getElementById('bi-y').value, c=document.getElementById('bi-c').value;
  if(!x||!y){return;}
  var tx=COLTYPES[x], ty=COLTYPES[y], traces=[], layout=_baseLayout(x+'  vs  '+y,x,y), k,i;
  if(tx==='num'&&ty==='num'){
    if(c&&c!=='__none__'){
      var gs=_uniq(DATA[c]).slice(0,10); layout.showlegend=true;
      for(k=0;k<gs.length;k++){var ix=[];for(i=0;i<DATA[c].length;i++){if(DATA[c][i]===gs[k])ix.push(i);}
        traces.push({type:'scatter',mode:'markers',name:String(gs[k]),marker:{size:6,color:CO[k%CO.length]},
          x:ix.map(function(i){return DATA[x][i];}),y:ix.map(function(i){return DATA[y][i];})});}
    } else { traces.push({type:'scatter',mode:'markers',marker:{size:6,color:CO[0]},x:DATA[x],y:DATA[y]}); }
  } else if(tx==='cat'&&ty==='num'){
    var g1=_uniq(DATA[x]).slice(0,30); layout.showlegend=false;
    for(k=0;k<g1.length;k++){var yy=[];for(i=0;i<DATA[x].length;i++){if(DATA[x][i]===g1[k])yy.push(DATA[y][i]);}
      traces.push({type:'box',name:String(g1[k]),y:yy});}
  } else if(tx==='num'&&ty==='cat'){
    var g2=_uniq(DATA[y]).slice(0,30); layout.showlegend=false;
    for(k=0;k<g2.length;k++){var xx=[];for(i=0;i<DATA[y].length;i++){if(DATA[y][i]===g2[k])xx.push(DATA[x][i]);}
      traces.push({type:'box',name:String(g2[k]),x:xx,orientation:'h'});}
  } else {
    var xs=_uniq(DATA[x]).slice(0,30), ys=_uniq(DATA[y]).slice(0,10); layout.barmode='stack'; layout.showlegend=true;
    for(k=0;k<ys.length;k++){var cnt=xs.map(function(xv){var m=0;for(i=0;i<DATA[x].length;i++){if(DATA[x][i]===xv&&DATA[y][i]===ys[k])m++;}return m;});
      traces.push({type:'bar',name:String(ys[k]),x:xs.map(String),y:cnt});}
    layout.yaxis.title='count';
  }
  Plotly.newPlot('bi-plot', traces, layout, {responsive:true, displayModeBar:false});
}
window.addEventListener('load', function(){ renderSection('overview'); });
"""
    js = (js.replace("__SPECS__", specs_js).replace("__SECMAP__", secmap_js)
          .replace("__DATA__", data_js).replace("__COLTYPES__", coltypes_js))

    nav = (
        "<a class='nav-item active' onclick=\"showSection('overview',this)\"><span class='ico'>▦</span><span class='txt'>Overview</span></a>"
        "<a class='nav-item' onclick=\"showSection('univariate',this)\"><span class='ico'>▮</span><span class='txt'>Univariate</span></a>"
        "<a class='nav-item' onclick=\"showSection('bivariate',this)\"><span class='ico'>◫</span><span class='txt'>Bivariate</span></a>"
        "<a class='nav-item' onclick=\"showSection('multivariate',this)\"><span class='ico'>▩</span><span class='txt'>Multivariate</span></a>")

    html_content = (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1.0'>"
        "<title>EDA Dashboard</title>"
        "<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>"
        "<style>" + css + "</style></head><body>"
        "<div class='app' id='app'>"
        "<aside class='sidebar' id='sidebar'>"
        "<div class='brand'><span class='logo'>◧</span><span class='brand-name'>EDA&nbsp;Studio</span></div>"
        "<nav>" + nav + "</nav></aside>"
        "<div class='main'>"
        "<header class='topbar' style='" + topbar_bg + "'>"
        "<button class='toggle' onclick='toggleSidebar()' title='Toggle sidebar'>☰</button>"
        "<div><div class='title'>Exploratory Data Analysis</div>"
        "<div class='subtitle'>" + f"{df.height:,} rows &middot; {df.width:,} columns" + "</div></div>"
        "</header>"
        "<main class='content'>"
        "<section id='overview' class='section active'>" + overview_body + "</section>"
        "<section id='univariate' class='section'><h2>Univariate analysis</h2>" + uni_body + "</section>"
        "<section id='bivariate' class='section'><h2>Bivariate analysis</h2>" + bi_body + "</section>"
        "<section id='multivariate' class='section'><h2>Multivariate analysis</h2>" + multi_body + "</section>"
        "</main></div></div>"
        "<script>" + js + "</script></body></html>")

    save_dashboard(html_content)



def fix_json_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Detect and fix JSON-type columns in the Polars DataFrame."""
    print_section_header("Checking and fixing json types of columns")

    print("Detecting and fixing json types of columns if there are any")
    new_columns = []

    for col in df.columns:
        if df[col].dtype == pl.Utf8:  # Ensure column is a string type
            try:
                # Check if at least one non-null row is valid JSON
                sample_value = df[col].drop_nulls().filter(
                    df[col].drop_nulls().str.starts_with("{") & df[col].drop_nulls().str.ends_with("}")
                ).head(1)

                if len(sample_value) > 0:
                    # Convert the column into a struct by parsing JSON
                    df = df.with_columns(
                        pl.col(col).map_elements(lambda x: json.loads(x) if x else None).alias(col)
                    )

                    # Expand struct into separate columns
                    expanded_cols = df[col].struct.unnest().rename({k: f"{col}_{k}" for k in df[col].struct.fields})

                    # Drop original JSON column and merge expanded data
                    df = df.drop(col).hstack(expanded_cols)
                    print(f"✅ Fixed JSON column: {col}")

            except Exception as e:
                print(f"⚠️ Error processing column {col}: {e}")

    return df



def detect_and_mask_pii_polars(df: pl.DataFrame, sample_size: int = 10) -> pl.DataFrame:
    """
    Detects and masks PII in a Polars DataFrame.

    Args:
        df (pl.DataFrame): Input Polars DataFrame.
        sample_size (int): Number of rows to sample for PII detection.

    Returns:
        pl.DataFrame: DataFrame with masked PII columns added.
    """
    print_section_header("Checking for any PII types in columns and masking them")

    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()
    pii_columns: Dict[str, List[str]] = {}

    # Step 1: Detect PII columns using sampling
    for col in df.columns:
        col_data = df[col].drop_nulls().cast(str)
        if len(col_data) == 0:
            continue
        sample = col_data.sample(min(sample_size, len(col_data)), seed=42)
        detected_types = set()

        for value in sample.to_list():
            results = analyzer.analyze(text=value, language='en')
            for r in results:
                detected_types.add(r.entity_type)

        if detected_types:
            pii_columns[col] = list(detected_types)

    # Print detected columns info or message if none found
    if pii_columns:
        print("Detected PII types in columns:")
        for col, types in pii_columns.items():
            print(f" - Column '{col}': {types}")
    else:
        print("No PII columns detected in the dataset.")

    # Step 2: Mask PII in detected columns
    df_dict = df.to_dict(as_series=False)  # Convert to dict for easy row-wise updates

    for col in pii_columns:
        masked_values = []
        for text in df_dict[col]:
            text = str(text) if text is not None else ""
            results = analyzer.analyze(text=text, language='en')
            if results:
                masked = anonymizer.anonymize(text=text, analyzer_results=results).text
                masked_values.append(masked)
            else:
                masked_values.append(text)

        df_dict[f"{col}_masked"] = masked_values

    return pl.DataFrame(df_dict)



def clean_data(df, background_image_path=None):
    """Main function to clean the data."""
    df = detect_column_types_and_process_text(df)
    df = handle_negative_values(df)
    df = replace_symbols_and_convert_to_float(df)
    df = fix_spelling_errors_in_columns(df)
    df = fix_spelling_errors_in_categorical(df)
    df = replace_symbols(df)

    df = handle_missing_values(df)
    df = handle_duplicates(df)
    check_cardinality(df)

    # df = remove_outliers(df)
    df = fix_skewness(df)
    df = check_multicollinearity(df)

    print_section_header("Enter target column (or press Enter to skip)")
    target_col = input("Enter the target column: ").strip()
    
    if target_col:
        if target_col in df.columns:
            df = check_and_handle_imbalance(df, target_col)
        else:
            print(f"⚠️ Warning: '{target_col}' not found in columns. Skipping imbalance handling.")
    else:
        print("ℹ️ Skipping target column–based imbalance check.")

    generate_dashboard(df, background_image_path=background_image_path)

    df = fix_json_columns(df)
    masked_df = detect_and_mask_pii_polars(df)
    df = save_cleaned_data(masked_df)
    return df
