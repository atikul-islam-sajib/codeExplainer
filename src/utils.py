import yaml
import os
import shutil
import joblib


def dump(value=None, filename=None):
    if (value is not None) and (filename is not None):
        joblib.dump(value=value, filename=filename)

    else:
        print("Error: value or filename is None".capitalize())


def load(filename=None):
    if filename is not None:
        return joblib.load(filename=filename)

    else:
        print("Filename cannot be none".capitalize())


def config():
    with open("./config.yml", "r") as file:
        return yaml.safe_load(file)


class CustomException(Exception):
    def __init__(self, message=None):
        if message is not None:
            self.message = message

        else:
            self.message = "This is the Custom Exception".title()


def clean_folder():
    code_path = config()["path"]["CODE_PATH"]
    if os.path.exists(code_path):
        shutil.rmtree(code_path)
    else:
        pass
