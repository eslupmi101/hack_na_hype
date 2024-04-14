import json

from django.conf import settings
import requests

from .dataclass import Prediction
from .exceptions import PredictionsParseError


def get_prediction_data(start_year: int = 1993, number_q: int = 120) -> list[Prediction] | None:
    parsed_data = []

    try:
        response = requests.get(
            f'{settings.AI_API_URL}/api/v1/predictions/',
            data=json.dumps({'start_year': start_year, 'number_q': number_q})
        )

        if response.status_code == 200:
            parsed_data = response.json()
        else:
            PredictionsParseError(f'Error parse {response.status_code}')

    except Exception as e:
        raise PredictionsParseError(f'Error parse {e}')

    prediction_data = []
    for prediction in parsed_data['data']:
        prediction_data.append(
            Prediction(
                **prediction
            )
        )

    return prediction_data
