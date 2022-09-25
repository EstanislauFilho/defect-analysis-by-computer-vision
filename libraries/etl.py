import pandas as pd

def __load_dataset(self, dataset_path: str) -> "pd.core.frame.DataFrame":
    """ Leitura e conversão do arquivo csv para dataframe

    Args:
        dataset_path (str): Caminho do arquivo csv que contém os dados
        do dataset que serão utilizados para a análise.

    Returns:
        pd.core.frame.DataFrame: DataFrame com os dados do dataset
    """
    return pd.read_csv(dataset_path)