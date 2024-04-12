from django.shortcuts import render


def index(request):
    template = 'index.html'
    context = {
        'user': request.user,
        # 'reviews_list': reviews_list
    }

    return render(request, template, context)
