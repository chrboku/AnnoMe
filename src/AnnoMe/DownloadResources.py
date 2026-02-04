from .Filters import download_MSMS_libraries, download_MS2DeepScore_model


def main():
    print("This script downloads necessary MS/MS libraries and models for AnnoMe.")
    print("Please note that some of these resources are rather large and in total are approximately 11 gigabytes in size.")
    print("Make sure you have sufficient disk space and a stable internet connection before proceeding.")
    print("Furthermore, please be aware that downloading these resources may take a considerable amount of time depending on your internet speed.")
    print("Please do not proceed if you are on a metered connection.")
    print("Do you wish to continue? (y/n): ", end="")
    choice = input().strip().lower()
    if choice != "y":
        print("Download aborted by user.")
    else:
        print("Starting download of resources to folder './resources'...")
        download_MSMS_libraries()
        download_MS2DeepScore_model()
