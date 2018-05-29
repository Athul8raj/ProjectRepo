# ProjectRepo


# Django framework for a Todo App

This app is made on top of Django2.0 framework

I have made use of pipenv module to take care of both virtual environment and easy installation.Just clone the pipfile and pipfile.lock after installing pipenv module and do pipenv install.You can activate the virtual env by typing pipenv shell

This Task/Todo app lets the user register and login into the app to see all the tasks that listed.Once the user clicks on the tasks it will navigate him/her to a detail view of the task where thye can edit/delete the task if it creted by them.It records the timestamp at which the editing has been done and who did it.They can als filter the completed task.

I have made use of Postegresql database to store the user details and for storing the tasks creeted by the user.
I made use of multiselect field module to make two choices for Done button.
I have used Foreign key of user to get the created_by and modified_by values from db.
I installed the django-filter module to filter out the completed tasks
I unit tested the app using a mixture of pytest,mixer and pytest-cov for getting test coverage


