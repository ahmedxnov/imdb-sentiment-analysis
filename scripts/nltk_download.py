#!/usr/bin/env python3

"""
Script to download required NLTK data for the IMDb sentiment analysis project.
Run this once after installing the project dependencies.
"""
import nltk
import sys

def download_nltk_data():
    datasets = ['punkt', 'stopwords']
    print("Downloading required NLTK data...")
    
    for dataset in datasets:
        try:
            print(f"Downloading {dataset}...")
            nltk.download(dataset, quiet=True)
            print(f"{dataset} downloaded successfully")
        except Exception as e:
            print(f"Failed to download {dataset}: {e}")
            return False
    
    print("\nAll NLTK data downloaded successfully!")
    return True

if __name__ == "__main__":
    success = download_nltk_data()
    if not success:
        sys.exit(1)