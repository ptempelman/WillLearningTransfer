import os


def delete_xml_files(directory):
    # Check if the directory exists
    if not os.path.exists(directory):
        print("Directory does not exist.")
        return

    # Loop through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".xml"):
            file_path = os.path.join(directory, filename)
            try:
                os.remove(file_path)
                print(f"Deleted {file_path}")
            except Exception as e:
                print(f"Error while deleting file {file_path}. Reason: {e}")


if __name__ == "__main__":
    # Usage
    directory_path = ".data/fruit/Orange"  # Replace with your directory path
    delete_xml_files(directory_path)
