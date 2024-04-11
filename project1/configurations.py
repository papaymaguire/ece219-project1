from dagster import Config

class Project1Config(Config):
    dataset_csv_path: str = "data/dataset.csv"