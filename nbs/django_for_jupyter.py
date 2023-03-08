import os, sys
# PWD = os.getenv('PWD')
PWD = os.path.dirname(os.getcwd())
PROJ_MISSING_MSG = """Set an enviroment variable:\n
`DJANGO_PROJECT=project`\n
or call:\n
`init_django(project)`
"""

def init_django(project_name=None):
    os.chdir(PWD)
    project_name = project_name or os.environ.get('DJANGO_PROJECT') or None
    if project_name == None:
        raise Exception(PROJ_MISSING_MSG) #  sys.path.insert(0, PWD)
#     sys.path.insert(0, os.getenv('PWD'))
    sys.path.insert(0, PWD)
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', f'{project_name}.settings')
    os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"
    import django
    django.setup()