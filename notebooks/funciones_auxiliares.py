import matplotlib.pyplot as plt
import seaborn as sns
import warnings


def plot_feature(df, col_name, isContinuous, target):
    """
    Visualize a variable with and without faceting on the loan status.
    - df dataframe
    - col_name is the variable name in the dataframe
    - full_name is the full variable name
    - continuous is True if the variable is continuous, False otherwise
    """
    f, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (12,3), dpi = 90)
    
    count_null = df[col_name].isnull().sum()
    
    if isContinuous:
        sns.histplot(df.loc[df[col_name].notnull(), col_name], kde = False, ax = ax1)
        
    else:
        sns.countplot(x = df[col_name], order = sorted(df[col_name].unique()), color = '#5975A4', saturation = 1, ax = ax1)
    ax1.set_xlabel(col_name)
    ax1.set_ylabel('Count')
    ax1.set_title(col_name + ' Numero de nulos: ' + str(count_null))
    plt.xticks(rotation = 90)


    if isContinuous:
        sns.boxplot(x = col_name, y = target, data = df, ax = ax2)
        ax2.set_ylabel('')
        ax2.set_title(col_name + ' by ' + target)
    else:
        data = df.groupby(col_name)[target].value_counts(normalize = True).to_frame('proportion').reset_index() 
        data.columns = [col_name, target, 'proportion']
        sns.barplot(x = col_name, y = 'proportion', hue = target, data = data, saturation = 1, ax = ax2)
        ax2.set_ylabel(target + ' fraction')
        ax2.set_title(target)
        plt.xticks(rotation = 90)
    ax2.set_xlabel(col_name)
    
    plt.tight_layout()


import pandas as pd
import numpy as np

def dame_variables_categoricas(dataset = None):
    '''
    ----------------------------------------------------------------------------------------------------------
    Función dame_variables_categoricas:
    ----------------------------------------------------------------------------------------------------------
        -Descripción: Función que recibe un dataset y devuelve una lista con los nombres de las 
        variables categóricas
        -Inputs: 
            -- dataset: Pandas dataframe que contiene los datos
        -Return:
            -- lista_variables_categoricas: lista con los nombres de las variables categóricas del
            dataset de entrada
    '''
    if dataset is None:
        print(u'\nFaltan argumentos por pasar a la función')
        return 1
    
    lista_variables_categoricas = []
    other = []
    
    for i in dataset.columns:
        if (dataset[i].dtype != float) & (dataset[i].dtype != int):
            unicos = int(len(np.unique(dataset[i].dropna(axis = 0, how = "all"))))
            if i in ["fraud_bool", "payment_type", "employment_status", "housing_status", "source", "device_os"]:
                lista_variables_categoricas.append(i)
            else:
                other.append(i)

    return lista_variables_categoricas, other

    for col in dataset.columns:
        if pd.api.types.is_categorical_dtype(dataset[col]) or pd.api.types.is_object_dtype(dataset[col]):
            unicos = dataset[col].nunique(dropna=True)
            if unicos < 100:
                lista_variables_categoricas.append(col)
            else:
                other.append(col)
        else:
            other.append(col)

    return lista_variables_categoricas, other


def get_corr_matrix(dataset = None, metodo='pearson', size_figure=[10,8]):
    import seaborn as sns
    # Para obtener la correlación de Spearman, sólo cambiar el metodo por 'spearman'

    if dataset is None:
        print(u'\nHace falta pasar argumentos a la función')
        return 1
    sns.set(style="white")
    # Compute the correlation matrix
    corr = dataset.corr(method=metodo) 
    # Set self-correlation to zero to avoid distraction
    for i in range(corr.shape[0]):
        corr.iloc[i, i] = 0
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=size_figure)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, center=0,
                square=True, linewidths=.5,  cmap ='viridis' ) #cbar_kws={"shrink": .5}
    plt.show()
    
    return 0

def get_deviation_of_mean_perc(pd_loan, list_var_continuous, target, multiplier):
    pd_final = pd.DataFrame()

    # Iterar sobre cada variable en list_var_continuous
    for i in list_var_continuous:
        # Calcular la media y la desviación estándar de la variable actual
        series_mean = pd_loan[i].mean()
        series_std = pd_loan[i].std()

        # Determinar los límites del intervalo de confianza utilizando el multiplicador
        std_amp = multiplier * series_std
        left = series_mean - std_amp
        right = series_mean + std_amp

        # Verificar si hay valores atípicos para la variable actual
        if pd_loan[i][(pd_loan[i] < left) | (pd_loan[i] > right)].size > 0:
            # Crear un DataFrame temporal con el recuento de valores target para los valores atípicos
            temp_df = pd_loan[target][(pd_loan[i] < left) | (pd_loan[i] > right)]\
                      .value_counts(normalize=True).reset_index()
            temp_df.columns = ['target_value', 'percentage']  # Renombrar las columnas para claridad

            # Crear pd_concat_percent con la información deseada
            # Aquí se incluyen la variable, la suma de valores atípicos y los datos de target
            pd_concat_percent = pd.DataFrame({
                'variable': [i],
                'sum_outlier_values': [pd_loan[i][(pd_loan[i] < left) | (pd_loan[i] > right)].size],
                'target_value': [temp_df.iloc[0, 0]],
                'percentage': [temp_df.iloc[0, 1]]
            })

            # Agregar pd_concat_percent a pd_final
            pd_final = pd.concat([pd_final, pd_concat_percent], axis=0).reset_index(drop=True)

    # Imprimir un mensaje si no se encontraron valores atípicos en ninguna variable
    if pd_final.empty:
        print('No existen variables con valores atípicos')

    return pd_final



import pandas as pd

def get_percent_null_values_target(df, list_var_continuous, target):
    df_final = pd.DataFrame()
    for i in list_var_continuous:
        if i in ["prev_address_months_count", "current_address_months_count", "bank_months_count",
                 "session_length_in_minutes", "device_distinct_emails_8w", "intended_balcon_amount"]:
            # Obtener los valores de 'target' donde 'i' es nulo
            s = df[target][df[i].isnull()].value_counts(normalize=True)
            if not s.empty:
                # Si hay valores, resetear el índice y transponer el resultado
                df_concat_percent = pd.DataFrame(s.reset_index()).T
                # Asegurarse de que hay dos columnas antes de renombrarlas
                if df_concat_percent.shape[1] == 2:
                    df_concat_percent.columns = ["no_fraud", "fraud"]
                    df_concat_percent = df_concat_percent.drop(df_concat_percent.index[0])
                    df_concat_percent['variable'] = i
                    df_concat_percent['sum_null_values'] = df[i].isnull().sum()
                    df_concat_percent['porcentaje_sum_null_values'] = df[i].isnull().sum() / df.shape[0]
                    df_final = pd.concat([df_final, df_concat_percent], axis=0).reset_index(drop=True)
                else:
                    print(f'Error: se esperaban 2 columnas para {i} pero se encontraron {df_concat_percent.shape[1]}')

    if df_final.empty or df_final["sum_null_values"].sum() == 0:
        return print('No existen variables con valores nulos')
        
    return df_final





import scipy.stats as ss

def cramers_v(confusion_matrix):
    """ 
    calculate Cramers V statistic for categorical-categorical association.
    uses correction from Bergsma and Wicher,
    Journal of the Korean Statistical Society 42 (2013): 323-328
    
    confusion_matrix: table created with pd.crosstab()
    """
    chi2, p, dof, expected = ss.chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum() # Total number of observations
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

# Usage example with a pandas crosstab:
# confusion_matrix = pd.crosstab(data_input_train["your_categorical_var1"], data_input_train["your_categorical_var2"])
# print(cramers_v(confusion_matrix))
