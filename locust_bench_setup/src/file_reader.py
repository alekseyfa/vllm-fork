import pandas as pd
import pyarrow.parquet as pq

class FileReader:
    def read_file(self, file_path: str) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement the read_file method")

    def write_file(self, df: pd.DataFrame, file_path: str) -> None:
        raise NotImplementedError("Subclasses must implement the write_file method")

class ParquetReader(FileReader):
    def read_file(self, file_path: str) -> pd.DataFrame:
        table = pq.read_table(file_path)
        return table.to_pandas()

    def write_file(self, df: pd.DataFrame, file_path: str) -> None:
        df.to_parquet(file_path)

class CSVReader(FileReader):
    def read_file(self, file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path)

    def write_file(self, df: pd.DataFrame, file_path: str) -> None:
        df.to_csv(file_path, index=False)

class FileProcessorFactory:
    @staticmethod
    def create_strategy(file_extension: str) -> FileReader:
        if file_extension == ".parquet":
            return ParquetReader()
        elif file_extension == ".csv":
            return CSVReader()
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")