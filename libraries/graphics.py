""" Script para apresentar e salvar gráficos gerados no projeto
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from settings import setup


def relationship_between_defects_and_no_defects(dataset: "pd.core.frame.DataFrame",
                                                **kwargs):
    """ Função responsável por gerar uma visualização gráfica que permite
    compreender a relação entre imagens de placas de aço com defeito e placas
    de aço sem defeito.

    Args:
        dataset (pd.core.frame.DataFrame): Dataset a ser analisado.
    """
    plt.figure(figsize=(10,10))
    sns.barplot(x = dataset['label'].value_counts().index, y = dataset['label'].value_counts())
    if kwargs.get("display") is True:
        plt.ylabel('Número de imagens')
        plt.xlabel('0 - Não defeito   1 - Defeito')
        plt.title('Relação entre Defeitos e Não defeitos');
        plt.show()
    if kwargs.get("save_img") is True:
        plt.savefig(setup.IMAGES_PATH+"/graphics/relationship_between_defects_and_no_defects.png", dpi=300)
    plt.close('all')

def relationship_between_types_of_defects(dataset: "pd.core.frame.DataFrame",
                                          **kwargs):
    """ Função responsável por gerar uma visualização gráfica que permite
    compreender a relação entre os tipos de defeitos.

    Args:
        dataset (pd.core.frame.DataFrame): Dataset a ser analisado.
    """
    plt.figure(figsize=(10,10))
    sns.countplot(x = dataset['ClassId'])
    if kwargs.get("display") is True:
        plt.ylabel('Número de imagens por defeito')
        plt.xlabel('Class ID')
        plt.title('Número de imagens por classe de defeito');
        plt.show()
    if kwargs.get("save_img") is True:
        plt.savefig(setup.IMAGES_PATH+"/graphics/relationship_between_types_of_defects.png", dpi=300)
    plt.close('all')