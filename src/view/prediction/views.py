from django.core.paginator import Paginator
from django.shortcuts import render
import logging

from .dataclass import Prediction
from .exceptions import PredictionsParseError
from .forms import UploadFileForm
from .services import get_prediction_data

logger = logging.getLogger(__name__)


def index(request):
    if request.method == 'POST':
        # Parse data
        form = UploadFileForm(
            request.POST,
            request.FILES or None
        )

        if form.is_valid():
            context = {
                'form': form
            }

            # Save csv to mediafiles dir
            try:
                csv_file = request.FILES['csv_file']
                with open('mediafiles/train.csv', 'wb') as destination:
                    for chunk in csv_file.chunks():
                        destination.write(chunk)
            except Exception:
                logger.critical('Error saving training data')

            # Get predictions
            data = None
            try:
                data = get_prediction_data(
                    start_year=form.cleaned_data['start_year'],
                    number_q=form.cleaned_data['number_q']
                )

                request.session['result_data'] = [obj.as_dict() for obj in data]
            except PredictionsParseError as e:
                logger.critical('Error parsing predictions training data %s', e)
            except Exception as e:
                logger.critical('Unexpected error whith parsing predictions training data %s', e)

            if data:
                paginator = Paginator(data, 30)
                page_number = request.GET.get('page')
                page_obj = paginator.get_page(page_number)
                context['page_obj'] = page_obj
            else:
                context['error_message'] = "Ошибка получения результата"

            return render(request, 'index.html', context)

    data = request.session.get('result_data', [])
    data = [Prediction(**obj) for obj in data]

    # Sort and find by id
    sort_param = request.GET.get('sort')
    if sort_param:
        if sort_param == 'asc':
            data = sorted(data, key=lambda x: x.percentage)
        elif sort_param == 'desc':
            data = sorted(data, key=lambda x: x.percentage, reverse=True)

    find_id_param = request.GET.get('find_id')
    if find_id_param and find_id_param.isdigit():
        data = [obj for obj in data if obj.id == int(find_id_param)]

    # Pagination
    paginator = Paginator(data, 30)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    form = UploadFileForm()
    context = {
        'page_obj': page_obj,
        'form': form
    }
    return render(request, 'index.html', context)
