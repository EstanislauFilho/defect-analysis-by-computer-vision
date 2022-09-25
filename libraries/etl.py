import pandas as pd

def load_dataset(dataset_path: str) -> "pd.core.frame.DataFrame":
    """ Leitura e conversão de arquivo csv para dataframe.

    Args:
        dataset_path (str): Caminho do arquivo csv que contém os dados
        do dataset que serão utilizados para a análise.

    Returns:
        pd.core.frame.DataFrame: DataFrame com os dados do dataset.
    """
    return pd.read_csv(dataset_path)