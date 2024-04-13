

async def get_predictions(start_year: int, number_q: int) -> list[dict] | None:
    result = []

    for i in range(100):
        result.append({
            'id': i,
            'mark': True,
            'percentage': 5
        })
    result.append({
        'id': 10,
        'mark': True,
        'percentage': 6
    })
    return result
