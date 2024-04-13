from django import forms


class UploadFileForm(forms.Form):
    csv_file = forms.FileField(
        label='Загрузите train.csv',
        widget=forms.FileInput(attrs={'class': 'form-control-file'})
    )
    start_year = forms.IntegerField(
        label='Первый год',
        help_text='Результат будет с этого года',
        initial=1993,
        min_value=1993,
        max_value=2022
    )
    number_q = forms.IntegerField(
        label='Количество кварталов',
        help_text='На эти кварталы произведется прогноз, начиная с первого года',
        initial=120,
        min_value=1,
    )

    def clean_csv_file(self):
        csv_file = self.cleaned_data['csv_file']
        if not csv_file.name.endswith('.csv'):
            raise forms.ValidationError('Only CSV files are allowed.')
