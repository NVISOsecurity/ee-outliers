from helpers.analyzer import Analyzer


class TestStubAnalyzer(Analyzer):

    def __init__(self, model_name, config_section):
        super(TestStubAnalyzer, self).__init__("analyzer", model_name, config_section)

    def evaluate_model(self):
        pass
