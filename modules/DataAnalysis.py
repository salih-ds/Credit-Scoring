## ФУНКЦИИ ДЛЯ ПРЕДВАРИТЕЛЬНОГО АНАЛИЗАН ДАННЫХ (EDA) ##

import matplotlib.pyplot as plt
import seaborn as sns

# вывести информацию о полях и размерности датасета
def primary_info_about_data(data):
    print(data.info())
    print('-'*50)
    print('Size: ', data.shape)
    print('-'*50)

# построить боксплоты для столбцов датафрейма
def boxplot_create(data, col):
    fig, axes = plt.subplots(figsize = (14, 4))
    sns.boxplot(x='default', y=col, data=data[data['sample']==1],ax=axes)
    axes.set_title('Boxplot for ' + col)
    plt.show()

