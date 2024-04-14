from django import forms


class UploadFileForm(forms.Form):
    csv_file = forms.FileField(
        label='Загрузите train.csv',
        widget=forms.FileInput(attrs={'class': 'form-control-file'})
    )

    def clean_csv_file(self):
        csv_file = self.cleaned_data['csv_file']
        if not csv_file.name.endswith('.csv'):
            raise forms.ValidationError('Only CSV files are allowed.')
