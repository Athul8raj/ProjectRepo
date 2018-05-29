from django.shortcuts import render, redirect
from Todos.forms import RegisterForm
from .models import Tasks
from .filters import UserFilter


def tasks(request):
    todos = Tasks.objects.all()
    context = {
        'todos': todos
    }

    return render(request, 'tasks.html', context)


def register(request):
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('/todo')
    else:
        form = RegisterForm()

        args = {'form': form}
        return render(request, 'register.html', args)


def login(request):
    return render(request, 'login.html')


def add(request):
    if request.method == 'POST':
        title = request.POST['title']
        desc = request.POST['desc']
        status = request.POST['status']

        todo = Tasks(title=title, desc=desc, status=status)
        todo.save()
        return redirect('/todo')

    else:
        return render(request, 'add_tasks.html')


def details(request, id):
    todo = Tasks.objects.get(id=id)

    context = {
        'todo': todo
    }
    return render(request, 'detail.html', context)


def edit(request, username):
    if request.method == 'POST':
        title = request.POST['title']
        desc = request.POST['desc']
        status = request.POST['status']
        todo = Tasks.objects.get(created_by=username)

        if todo:
            todo = Tasks(title=title, desc=desc, status=status)
            todo.save()

            return redirect('/todo')

    else:
        return render(request, 'add_tasks.html')


def delete(request, username):
    todo = Tasks.objects.get(created_by=username)
    if todo:
        todo.delete()

        return redirect('/todo')


def search(request, status):
    if status == 'Done':
        user_list = Tasks.objects.all()
        user_filter = UserFilter(request.GET, queryset=user_list)
        return render(request, 'search/tasks.html', {'filter': user_filter})
    else:
        return render(request, 'tasks.html')
