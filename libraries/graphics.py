""" Script para apresentar e salvar gráficos gerados no projeto
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from settings import setup


def relationship_between_defects_and_no_defects(dataframe: "pd.core.frame.DataFrame",
                                                **kwargs):
    """ Função responsável por gerar uma visualização gráfica que permite
    compreender a relação entre imagens de placas de aço com defeito e placas
    de aço sem defeito.

    Args:
        dataframe (pd.core.frame.DataFrame): DataFrame a ser analisado
        attribute (str): Atributo que terá sua variável contada em relação ao grupo
        relation (str): Variável que será utilizada para comparar as incidência de um
        atributo sobre outro.
        display (bool): True caso deseja-se apresentar a imagem. False caso contrário
        save_img (bool): True caso deseja-se salvar a imagem. False caso contrário
    """
    plt.figure(figsize=(10,10))
    sns.barplot(x = dataframe['label'].value_counts().index, y = dataframe['label'].value_counts())
    if kwargs.get("display") is True:
        plt.ylabel('Número de imagens')
        plt.xlabel('0 - Não defeito   1 - Defeito')
        plt.title('Defeitos e não defeitos');
        plt.show()
    if kwargs.get("save_img") is True:
        plt.savefig(setup.IMAGES_PATH+"/graphics/relationship_between_defects_and_no_defects.png", dpi=300)
    plt.close('all')