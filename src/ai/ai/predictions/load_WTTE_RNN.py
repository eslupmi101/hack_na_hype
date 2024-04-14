from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Activation
from keras.layers import Masking
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras import backend as k
from tqdm import tqdm
from sklearn.preprocessing import normalize, StandardScaler, LabelEncoder
from scipy.stats import weibull_min
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import keras


# Инициализация стандартизатора
#scaler = StandardScaler()

# Инициализация LabelEncoder
label_encoder = LabelEncoder()


"""
     Дискретное логарифмическое правдоподобие для функции риска Вейбулла на цензурированных данных о выживании
     y_true — тензор (выборки, 2), содержащий время до события (y) и индикатор события (u).
     ab_pred — тензор (выборки, 2), содержащий предсказанные параметры Вейбулла альфа (а) и бета (б)
     Математические сведения см. https://ragulpr.github.io/assets/draft_master_thesis_martinsson_egil_wtte_rnn_2016.pdf (стр. 35).
"""
def weibull_loglik_discrete(y_true, ab_pred, name=None):
    y_ = y_true[:, 0]
    u_ = y_true[:, 1]
    a_ = ab_pred[:, 0]
    b_ = ab_pred[:, 1]

    hazard0 = k.pow((y_ + 1e-35) / a_, b_)
    hazard1 = k.pow((y_ + 1) / a_, b_)

    return -1 * k.mean(u_ * k.log(k.exp(hazard1 - hazard0) - 1.0) - hazard1)

"""
     Не используется для этой модели, но включен в комплект на случай, если кому-то понадобится.
     Математические сведения см. https://ragulpr.github.io/assets/draft_master_thesis_martinsson_egil_wtte_rnn_2016.pdf (стр. 35).
"""
def weibull_loglik_continuous(y_true, ab_pred, name=None):
    y_ = y_true[:, 0]
    u_ = y_true[:, 1]
    a_ = ab_pred[:, 0]
    b_ = ab_pred[:, 1]

    ya = (y_ + 1e-35) / a_
    return -1 * k.mean(u_ * (k.log(b_) + b_ * k.log(ya)) - k.pow(ya, b_))

"""
    Пользовательская функция активации Keras, выводит альфа-нейрон с использованием возведения в степень и бета-версию с помощью softplus
"""
def activate(ab):
    a = k.exp(ab[:, 0])
    b = k.softplus(ab[:, 1])

    a = k.reshape(a, (k.shape(a)[0], 1))
    b = k.reshape(b, (k.shape(b)[0], 1))

    return k.concatenate((a, b), axis=1)

def fun_mean_std(train):

  ids=train['client_id']
  train=train.drop('client_id', axis=1)

  # Вычисление средних значений и стандартных отклонений для каждого столбца и применение к данным
  #normalized_data = scaler.fit_transform(train)
  train=train.fillna(0)
  normalized_data = normalize(train.values, axis=0)
  #normalized_data
  # Создание нового DataFrame с нормализованными данными
  df_normalized = pd.DataFrame(normalized_data, columns=train.columns)

  df_normalized['User_ID']=ids

  return df_normalized

def fun_with_LabelEncoder(df_normalized):
  # Применение LabelEncoder к столбцу 'id'
  df_normalized['User_ID'] = label_encoder.fit_transform(df_normalized['User_ID'])

  df_normalized['Time_Step'] = df_normalized.groupby('User_ID').cumcount() + 1

  # Сгруппировать данные по user_id и вычислить максимальное значение временной метки в каждой группе
  max_timestamps = df_normalized.groupby('User_ID')['Time_Step'].max()

  # Объединить максимальные значения с исходным DataFrame
  df_normalized = df_normalized.merge(max_timestamps.rename('max_timestamp'), on='User_ID')

  # Вычесть максимальное значение временной метки из каждой временной метки в группе
  df_normalized['time_diff'] = df_normalized['max_timestamp'] - df_normalized['Time_Step']

  # Удаляем вспомогательный столбец
  df_normalized=df_normalized.drop(['max_timestamp'], axis=1)

  df_normalized['Time_Step'] = df_normalized['time_diff']
  df_normalized = df_normalized.drop(['time_diff'], axis=1)

  return df_normalized

#ускоренная версия функции build_data_v2
def build_data_v3(df, max_time, is_test, mask_value=-99):

    grouped = df.groupby('User_ID')

    out_y = []
    out_x = []

    for user_id, group in tqdm(grouped):
        max_engine_time = group['Time_Step'].max()

        if is_test:
            start = max_engine_time - 1
        else:
            start = 0

        this_x = []

        for j in range(start, max_engine_time):
            engine_x = group.iloc[:j+1].drop(['User_ID', 'Time_Step'], axis=1).to_numpy()

            out_y.append([max_engine_time - j, 1])

            xtemp = np.zeros((max_time, engine_x.shape[1]))
            xtemp[max_time-min(j, max_time-1)-1:max_time, :] = engine_x[max(0, j-max_time+1):j+1, :]
            this_x.append(xtemp)

        out_x.extend(this_x)

    return np.array(out_x), np.array(out_y)

#преобразует датафреймы в формат для обучения
def get_dfs(mini_test_df, test_y, max_time, mask_value):

  test_x = build_data_v3(mini_test_df, max_time, True, mask_value)[0]

  train_u = np.zeros((test_y.shape[0], 1), dtype=np.float32)
  train_u += 1
  test_y = np.append(np.reshape(test_y, (test_y.shape[0], 1)), train_u, axis=1)
  return (test_x, test_y)

def post_prediction_reconstruct(x_pred, y, y_cenzor=0):
  y[:,0] = y[:,0]+y_cenzor
  x_pred = np.resize(x_pred, (y.shape[0], 2))
  result = np.concatenate((y, x_pred), axis=1)

  # TTE, Event Indicator, Alpha, Beta
  results_df = pd.DataFrame(result, columns=['T', 'E', 'alpha', 'beta'])
  results_df['unit_number'] = np.arange(1, results_df.shape[0]+1)
  return results_df

def weibull_pdf(alpha, beta, t):
    return weibull_min.pdf(t, beta, scale=alpha)

def weibull_median(alpha, beta):
    return weibull_min.median(beta, scale=alpha)

def weibull_mean(alpha, beta):
    return weibull_min.mean(beta, scale=alpha)

def weibull_mode(alpha, beta):
    assert np.all(beta > 1)
    return alpha * ((beta-1)/beta)**(1/beta)

# Вычисляем вероятность попадания в интервал [a, b]

def get_prop(id, test_results_df,a ,b):
    row=test_results_df[test_results_df['unit_number']==id]

    #for i, row in enumerate(results_df.iterrows()):
    alpha=row['alpha'].values[0]
    beta = row['beta'].values[0]
    T = row['T'].values[0]
    weibull_dist = weibull_min(beta, scale=alpha)
    probability = weibull_dist.cdf(b) - weibull_dist.cdf(a)

    return probability

    #print("Вероятность попадания в интервал от {} до {}: {:.4f}".format(a, b, probability))

def prop_dicts(test_results_df, a, b):
  prop_dict={}
  for user_i in range(1,len(test_results_df)+1):
    prop_dict[user_i]=get_prop(user_i, test_results_df,a ,b)

  return prop_dict

def get_metric_AUC(best_threshold, prop_dict):

  y_pred_AUC = np.array(list(prop_dict.values()))
  taken_ids=list(prop_dict.keys())
  y_pred_AUC = y_pred_AUC > best_threshold

  return y_pred_AUC, taken_ids

def first_train_data(train, drop_list):
  train=train.drop(drop_list, axis=1)
  old_train=train
  return train, old_train

def prep_data(file_dist: str):
    train = pd.read_csv(file_dist)

    train = train.sort_values(by='quarter')

    # Group by
    client_lives = train.groupby('client_id')['churn'].sum()

    # Reset
    client_lives = pd.DataFrame(client_lives).reset_index()
    client_lives = client_lives.rename(
        columns={'churn': "sum_leave"}
    )

    train = pd.merge(train, client_lives, on='client_id', how='left')

    # Cut data
    d_train = train[train['sum_leave'] == 1]

    # Сгруппировать данные по user_id и вычислить максимальное значение временной метки в каждой группе
    max_timestamps = d_train.groupby('client_id')['clnt_cprtn_time_d'].max()

    # Объединить максимальные значения с исходным DataFrame
    d_train = d_train.merge(max_timestamps.rename('max_timestamp'), on='client_id')

    # Вычесть максимальное значение временной метки из каждой временной метки в группе
    d_train['time_diff'] = d_train['max_timestamp'] - d_train['clnt_cprtn_time_d']
    d_train = d_train.drop(['max_timestamp'], axis=1)

    d_train = d_train.drop(
        ['npo_account_id',
         'region',
         'lst_pmnt_date_per_qrtr',
         'frst_pmnt_date',
         'year'],
        axis=1
    )

    pp = d_train.groupby(['client_id', 'quarter']).mean()

    #pp.to_csv('prepare_data.csv')

    return pp


def main_for_model(filepath_for_model, test_x, test_y):
  model = keras.models.load_model(filepath = filepath_for_model, custom_objects = {"weibull_loglik_discrete": weibull_loglik_discrete, "activate": activate})

  def get_alpha_beta(x, y, y_cenzor=0):
    predict = model.predict(x)
    results_df = post_prediction_reconstruct(predict, y, y_cenzor)
    results_df.loc[results_df['beta'] <= 1, 'beta'] = 1.1
    return results_df

  test_results_df = get_alpha_beta(test_x, test_y)
  return test_results_df

def submisions(output, taken_ids, y_pred_AUC):

  output['customer_id']=label_encoder.inverse_transform(np.array(taken_ids)-1)
  output['label']=y_pred_AUC

  label_encoder.inverse_transform([2])

  return output

def WTTE_RNN(filepath_for_data ,drop_list, filepath_for_model, max_time, mask_value, a, b, best_threshold):
  train = prep_data(filepath_for_data)
  train = train.reset_index(['client_id', 'quarter'])
  train, old_train = first_train_data(train, drop_list)

  df_normalized = fun_mean_std(train)
  df_normalized = fun_with_LabelEncoder(df_normalized)

  test_y = df_normalized.groupby('User_ID')['Time_Step'].max()+1
  test_x, test_y = get_dfs(df_normalized, test_y, max_time, mask_value)

  #test_x.shape
  model = keras.models.load_model(filepath = filepath_for_model, custom_objects = {"weibull_loglik_discrete": weibull_loglik_discrete, "activate": activate})

  test_results_df = main_for_model(filepath_for_model, test_x, test_y)
  prop_dict = prop_dicts(test_results_df, a, b)
  y_pred_AUC, taken_ids = get_metric_AUC(best_threshold, prop_dict)

  output = pd.DataFrame([])
  output=submisions(output, taken_ids, y_pred_AUC)

  return output


def get_predictions():
    # Инициализация стандартизатора
    # scaler = StandardScaler()

    # Инициализация LabelEncoder
    # label_encoder = LabelEncoder()

    filepath_for_data = "./mediafiles/train.csv"

    filepath_for_model = "./mediafiles/model.keras"

    max_time = 5  # проходимся окном, длинной max_time для каждого пользователя. (Работает как буффер)
    mask_value = -99  # значение, которым будут заполняться неполные подпоследовательности (буффера)

    a, b = 0, 1
    best_threshold = 0.4

    drop_list = [
        'quarter', 'slctn_nmbr', 'pmnts_type', 'incm_per_year', 'mgd_accum_period',
        'mgd_payment_period', 'phone_number', 'email', 'lk',
        'assignee_ops', 'postal_code', 'fact_addrss', 'churn', 'sum_leave', 'time_diff'
    ]

    output = WTTE_RNN(
        filepath_for_data,
        drop_list,
        filepath_for_model,
        max_time,
        mask_value,
        a,
        b,
        best_threshold
    )
    output.to_csv('./mediafiles/result.csv')
