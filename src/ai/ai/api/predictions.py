import os

from ..predictions.load_WTTE_RNN import WTTE_RNN
from .schemas import Prediction


async def get_predictions(start_year: int, number_q: int) -> list[dict] | None:
    # Инициализация стандартизатора
    # scaler = StandardScaler()

    # Инициализация LabelEncoder
    # label_encoder = LabelEncoder()

    filepath_for_data = "./mediafiles/train.csv"

    not_docker_file_path = '../view/mediafiles/train.csv'

    if not os.path.exists(filepath_for_data):
        filepath_for_data = not_docker_file_path

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

    path = './mediafiles/result.csv'

    output.to_csv(path)

    predictions = []
    for _, row in output.head(500).iterrows():
        prediction = Prediction(customer_id=row['customer_id'], label=row['label'], percentage=0)
        predictions.append(prediction)

    return predictions
