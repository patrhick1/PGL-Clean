#file_manipulation.py

import os


def list_filepaths(directory_path):
    """
    Get all filepaths in a directory.

    Args:
        directory_path (str): The path to the directory.

    Returns:
        list: A list of full file paths in the directory, or an empty list if an error occurs.
    """
    try:
        # Normalize the directory path to handle backslashes properly
        directory_path = os.path.normpath(directory_path)
        return [os.path.join(directory_path, file) for file in os.listdir(directory_path)]
    except FileNotFoundError:
        print(f"Error: Directory '{directory_path}' not found.")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []

def get_file_extension(file_path):
    """
    Get the file extension from a given file path.

    Args:
        file_path (str): The path to the file.

    Returns:
        str: The file extension including the dot (e.g., '.txt', '.docx').
    """
    _, extension = os.path.splitext(file_path)
    return extension

def read_txt_file(file_path):
    """
    Read content from a text file.

    Args:
        file_path (str): The path to the text file.

    Returns:
        str: The content of the file if successful, an error message otherwise.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except UnicodeDecodeError:
        # If UTF-8 fails, try another common encoding
        with open(file_path, 'r', encoding='iso-8859-1') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        return 'File not found.'
    
