import pandas as pd


def create_test(path="../data/medmcqa/test.json"):
    test_set = pd.read_json(path)
    test_set['mod_question'] = test_set.apply(
        lambda row: f"For the given question, choose the correct answer from the options_list.\n\n"
                    f"question: {row['question']}\n\n"
                    f"options_list: [option_a:{row['opa']}\noption_b:{row['opb']}\n,option_c:{row['opc']}\noption_d:{row['opd']}]\n",
        axis=1
    )
    test_set['query'] = test_set.apply(lambda row : f"question: {row['question']}\n\n"
                                                    f"option_a:{row['opa']}\n"
                                                    f"option_b:{row['opb']}\n"
                                                    f"option_c:{row['opc']}\n"
                                                    f"option_d:{row['opd']}\n",
                                       axis=1)
    test_set.to_csv('mini_test.csv', index=False)


create_test()
