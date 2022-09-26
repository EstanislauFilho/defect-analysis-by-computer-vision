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


def count_of_defect_types_per_image(dataset: "pd.core.frame.DataFrame") \
                                    -> "pd.core.frame.DataFrame":
    """ Função responsável por analisar o dataset de placas de aço com defeito 
    e contar a quantidade de defeitos presente em cada imagem.

    Args:
        dataset (pd.core.frame.DataFrame): Dataset a ser analisado.
    Returns:
        dataset (pd.core.frame.DataFrame): Dataset com nova coluna indicando a
        quantidade defeitos encontrados para cada imagem.
    """
    # adiciona um nova coluna no final do dataframe
    dataset['mask'] = dataset['ClassId'].map(lambda x: 1)
    
    # faz o agrupamento das imagens repetidas somando o atributo mask,
    # sendo possível assim contar a quantidade defeitos por imagem.
    return dataset.groupby(['ImageId'])['mask'].sum()