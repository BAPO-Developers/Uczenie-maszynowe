import pandas as pd


class PrepareDataSets(object):
    n_classes = 2
    n_instances = []
    n_atributes = []
    file_name = ""
    show_details = False

    def __init__(self, show_details=False):
        self.show_details = show_details

    def add_file(self, data_set_url):
        data = pd.read_csv(data_set_url)

        self.n_instances, self.n_atributes = data.shape
        firstpos = data_set_url.rfind("/")
        lastpos = data_set_url.rfind(".")
        self.file_name = data_set_url[firstpos+1:lastpos]

        if self.show_details:
            print('File name:', self.file_name)
            print("Number of columns ", self.n_atributes)
            print("Number of rows ", self.n_instances)
        return data
