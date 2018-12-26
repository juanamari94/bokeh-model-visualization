import pickle


class ModelLoader:

    def __init__(self, file_path):
        self.file_path = file_path

    def read_serialized_data(self):
        with open(self.file_path, "rb") as handler:
            return pickle.load(handler)

    def assemble_serialized_data(self):
        data = self.read_serialized_data()
        return data
