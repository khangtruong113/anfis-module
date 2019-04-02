import json


class Reports:
    @staticmethod
    def origin_anfis(name: str,
                     rule_number: int,
                     window_size: int,
                     mse: float,
                     dataset: str,
                     path: str):
        """
        :param name:
        :param rule_number:
        :param window_size:
        :param mse:
        :param dataset:
        :param path:
        :return:
        """
        template = {"name": name,
                    "dataset": dataset,
                    "rule_number": rule_number,
                    "window_size": window_size,
                    "mse": mse,
                    }
        with open(path, 'w') as fp:
            json.dump(template, fp)
