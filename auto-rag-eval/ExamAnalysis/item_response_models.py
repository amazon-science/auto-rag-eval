import glob
import json
from dataclasses import dataclass

# from scipy.stats import norm, lognorm, beta
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from ExamAnalysis.bloom_taxonomy_model import categorize_question
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error


@dataclass
class ExamSetting:
    llm: str
    retrieval: str
    icl: int
    name: str
    path_pattern: str  # Assuming base path is a constant attribute of the class

    def find_file_path(self) -> str:
        """
        Find the file path using the class attributes.
        """
        # Search for files matching the pattern
        matching_files = glob.glob(self.path_pattern)

        # Return the first matching file or None
        if matching_files is None or matching_files == []:
            raise ValueError(f"Incorrect path pattern {self.path_pattern}")

        return matching_files[0]

    @property
    def exists(self) -> bool:

        # Search for files matching the pattern
        matching_files = glob.glob(self.path_pattern)

        return matching_files is not None and matching_files != []

    @property
    def data_path(self) -> str:
        """
        Property to get the data path.
        """
        return self.find_file_path()


class BaseItemResponseModel:

    def __init__(self,
                 students: List[ExamSetting],
                 irt_model_type: int):

        self.students = [stud for stud in students if stud.exists]
        print(
            f'Total of {len(self.students)} students considered out of {len(students)}')
        self.irt_model_type = irt_model_type
        assert self.irt_model_type in [1, 2, 3], "Specify correct IRT model, among 1PL, 2PL or 3PL"
        self.data = np.array([[elem['acc'] for elem in self.load_data(exam_setting.data_path)]
                              for exam_setting in self.students])
        self.num_students = self.data.shape[0]
        self.num_items = self.data.shape[1]

        self.llm_map = {llm: index
                        for index, llm in enumerate(set([student.llm for student in self.students]))}
        self.retrieval_map = {llm: index
                              for index, llm in enumerate(set([student.retrieval for student in self.students]))}
        self.num_icl = 3
        self.num_llm = len(self.llm_map)
        self.num_retrieval = len(self.retrieval_map)
        self.num_theta_params = len(self.llm_map) + len(self.retrieval_map) + self.num_icl

    def load_data(self, data_path):
        with open(data_path, 'r') as f:
            data = [json.loads(line) for line in f]

        return data

    def irt_1pl(self, theta: np.array, b: float) -> float:
        return 1 / (1 + np.exp(-(theta - b)))

    def irt_2pl(self, theta: np.array, a: float, b: float) -> float:
        return 1 / (1 + np.exp(-a * (theta - b)))

    def irt_3pl(self, theta: np.array, a: float, b: float, c: float) -> float:
        return c + ((1 - c) / (1 + np.exp(-a * (theta - b))))

    def irt_model(self, theta: np.array, a: float, b: float, c: float) -> float:

        if self.irt_model_type == 1:

            return self.irt_1pl(theta, b)

        elif self.irt_model_type == 2:

            return self.irt_2pl(theta, a, b)

        else:

            return self.irt_3pl(theta, a, b, c)

    def plot(self,
             estimator: Dict[str, np.array],
             exam_model: str,
             save_path: str = None,
             font_size: int = 18) -> None:

        # Set global font size
        plt.rcParams.update({'font.size': font_size})

        model_list = [student.name for student in self.students]

        # Assume these are the estimated parameters for 3 items
        a = estimator['discrimination']  # Discrimination parameters
        b = estimator['difficulty']  # Difficulty parameters
        c = estimator['guessing']  # Guessing parameters

        # Create an array of theta values for plotting
        theta_values = np.linspace(-3, 3, 300)

        # Create a 2x2 grid of subplots
        fig, axs = plt.subplots(2, 2, figsize=(24, 16))

        # Plot Item Characteristic Curves (ICC)
        # plt.figure(figsize=(12, 8))
        for i in range(len(a)):
            p = self.irt_model(theta=theta_values, a=a[i], b=b[i], c=c[i])
            axs[0, 0].plot(theta_values, p, label=f'Item {i+1}')

        # Define a list of colors
        colors = ['red', 'green', 'blue', 'purple', 'orange']

        # Add markers on the x-axis for the estimated theta values
        for k, theta in enumerate(estimator['theta']):
            color = colors[k % len(colors)]
            axs[0, 0].scatter(theta, 0, marker='x', color=color)
            # axs[0, 0].text(theta, 0.02,
            #               model_list[k],
            #               ha='center',
            #               rotation=45,
            #               color=color)  # Adjust the second parameter to position the text above the marker

        axs[0, 0].set_title(f'Question Characteristic Curves - {exam_model} Exam - {self.irt_model_type}PL Model')
        axs[0, 0].set_xlabel('Theta (Ability)')
        axs[0, 0].set_ylabel('Probability of Correct Answer')
        # plt.legend()
        axs[0, 0].grid(True)

        # Plot Item Information Curves
        # plt.figure(figsize=(12, 8))
        for i in range(len(a)):
            p = self.irt_model(theta=theta_values, a=a[i], b=b[i], c=c[i])
            information = a[i]**2 * p * (1 - p)
            axs[0, 1].plot(theta_values, information, label=f'Item {i+1}')

        # Define a list of colors
        colors = ['red', 'green', 'blue', 'purple', 'orange']
        # Add markers on the x-axis for the estimated theta values
        for k, theta in enumerate(estimator['theta']):
            color = colors[k % len(colors)]
            axs[0, 1].scatter(theta, 0, marker='x', color=color)
            # axs[0, 1].text(theta, 0.02,
            #               model_list[k],
            #               ha='center',
            #               rotation=45,
            #               color=color)  # Adjust the second parameter to position the text above the marker

        axs[0, 1].set_title(f'Question Information Curves - {exam_model} Exam - {self.irt_model_type}PL Model')
        axs[0, 1].set_xlabel('Theta (Ability)')
        axs[0, 1].set_ylabel('Fisher Information')
        # plt.legend()
        axs[0, 1].grid(True)
        # plt.show()

        questions = [elem['doc']['question']
                     for elem in self.load_data(self.students[0].data_path)]
        # Plot Test Information Curve
        # plt.figure(figsize=(12, 8))

        test_information = np.zeros_like(theta_values)
        for i in range(len(a)):
            p = self.irt_model(theta=theta_values, a=a[i], b=b[i], c=c[i])
            information = a[i]**2 * p * (1 - p)
            test_information += information  # Sum up information from all items
        axs[1, 0].plot(theta_values, test_information / len(a), label=f'Average [{len(a)}]')

        for question_mark in ["Which", "What", "How", "When", "Why", "Where"]:
            test_information = np.zeros_like(theta_values)
            n_items = 0
            for i in range(len(a)):
                if question_mark.lower() in questions[i].lower():
                    n_items += 1
                    p = self.irt_model(theta=theta_values, a=a[i], b=b[i], c=c[i])
                    information = a[i]**2 * p * (1 - p)
                    test_information += information  # Sum up information from all items
            if n_items > 0:
                axs[1, 0].plot(theta_values, test_information / n_items, label=f'{question_mark} [{n_items}]')

        # Define a list of colors
        colors = ['red', 'green', 'blue', 'purple', 'orange']

        # Add markers on the x-axis for the estimated theta values
        for k, theta in enumerate(estimator['theta']):
            color = colors[k % len(colors)]
            axs[1, 0].scatter(theta, 0, marker='x', color=color)
            # axs[1, 0].text(theta,
            #               0.02,
            #               self.students[k].name,
            #               ha='center',
            #               rotation=45,
            #              color=color)  # Adjust the second parameter to position the text above the marker

        axs[1, 0].set_title(f'Exam Information Curve  - {exam_model} Exam - {self.irt_model_type}PL Model')
        axs[1, 0].set_xlabel('Theta (Ability)')
        axs[1, 0].set_ylabel('Fisher Information')
        axs[1, 0].legend()
        axs[1, 0].grid(True)
        # plt.show()

        questions_taxonomy = [categorize_question(elem['doc']['question'])
                              for elem in self.load_data(self.students[0].data_path)]
        # Plot Test Information Curve
        # plt.figure(figsize=(12, 8))

        test_information = np.zeros_like(theta_values)
        for i in range(len(a)):
            p = self.irt_model(theta=theta_values, a=a[i], b=b[i], c=c[i])
            information = a[i]**2 * p * (1 - p)
            test_information += information  # Sum up information from all items
        axs[1, 1].plot(theta_values, test_information / len(a), label=f'Average [{len(a)}]')

        for taxonomy in ['Remembering', 'Understanding', 'Applying', 'Analyzing', 'Evaluating', 'Creating', 'Uncategorized']:
            test_information = np.zeros_like(theta_values)
            n_items = 0
            for i in range(len(a)):
                if taxonomy in questions_taxonomy[i]:
                    n_items += 1
                    p = self.irt_model(theta=theta_values, a=a[i], b=b[i], c=c[i])
                    information = a[i]**2 * p * (1 - p)
                    test_information += information  # Sum up information from all items
            if n_items > 0:
                axs[1, 1].plot(theta_values, test_information / n_items, label=f'{taxonomy} [{n_items}]')

        # Define a list of colors
        colors = ['red', 'green', 'blue', 'purple', 'orange']

        # Add markers on the x-axis for the estimated theta values
        for k, theta in enumerate(estimator['theta']):
            color = colors[k % len(colors)]
            axs[1, 1].scatter(theta, 0, marker='x', color=color)
            # axs[1, 1].text(theta,
            #               0.02,
            #               self.students[k].name,
            #               ha='center',
            #               rotation=45,
            #               color=color)  # Adjust the second parameter to position the text above the marker

        axs[1, 1].set_title(f'Exam Information Curve  - {exam_model} Exam - {self.irt_model_type}PL Model')
        axs[1, 1].set_xlabel('Theta (Ability)')
        axs[1, 1].set_ylabel('Fisher Information')
        axs[1, 1].legend()
        axs[1, 1].grid(True)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()


class ItemResponseModel(BaseItemResponseModel):

    def __init__(self,
                 students: List[ExamSetting],
                 irt_model_type: int):

        super().__init__(students=students,
                         irt_model_type=irt_model_type)

    # Define the negative log-likelihood function for the 3PL model
    def neg_log_likelihood(self, params: np.array) -> float:
        a = params[:self.num_items]
        b = params[self.num_items:2 * self.num_items]
        c = params[2 * self.num_items:3 * self.num_items]
        theta = params[3 * self.num_items:]

        likelihood = 0
        for i in range(self.num_items):
            p = self.irt_model(theta=theta, a=a[i], b=b[i], c=c[i])
            likelihood += np.sum(self.data[:, i] * np.log(p) + (1 - self.data[:, i]) * np.log(1 - p))

        # Add a param penalty
        # l2_penalty = lambda_l2 * np.sum(params ** 2)
        # return -likelihood + l2_penalty

        # Add Gaussian priors for a, b, and c
        # prior_a = np.sum(-0.5 * ((a - prior_means['a']) ** 2) / prior_vars['a'])
        # prior_b = np.sum(-0.5 * ((b - prior_means['b']) ** 2) / prior_vars['b'])
        # prior_c = np.sum(-0.5 * ((c - prior_means['c']) ** 2) / prior_vars['c'])

        # Adding prior terms
        # Discrimination parameter a: log-normal prior
        # log_normal_prior_a = np.sum(lognorm.logpdf(a, 1))

        # Difficulty parameter b: normal prior
        # normal_prior_b = np.sum(norm.logpdf(b, 0, 1))

        # Guessing parameter c: beta prior
        # beta_prior_c = np.sum(beta.logpdf(c, 2, 5))

        return -likelihood

    def fit(self) -> Dict[str, np.array]:

        # Initial guesses for a, b, c and theta
        initial_guess = np.concatenate([
            np.ones(self.num_items),  # Initial guesses for a [discrimination]
            np.zeros(self.num_items),  # Initial guesses for b [difficulty]
            np.full(self.num_items, 0.25),  # Initial guesses for c [guessing]
            np.zeros(self.num_students)  # Initial guesses for theta
        ])

        # b should be initialized at log((1-c)/(E[P]-c)

        params_bounds = [
            [(0.5, 1.5) for _ in range(self.num_items)],
            [(0.01, 1) for _ in range(self.num_items)],
            [(0.2, 0.4) for _ in range(self.num_items)],
            [(-3, 3) for _ in range(self.num_students)]
        ]

        # Run optimization
        result = minimize(
            self.neg_log_likelihood,
            initial_guess,
            method='L-BFGS-B',
            bounds=[elem for bounds in params_bounds for elem in bounds]
        )

        return {
            'discrimination': result.x[:self.num_items],
            'difficulty': result.x[self.num_items:2 * self.num_items],
            'guessing': result.x[2 * self.num_items:3 * self.num_items],
            'theta': result.x[3 * self.num_items:],
        }

    def compute_stats(self, estimator: Dict[str, np.array]):

        # Calculate the RMSE for each item
        rmse_val = [np.sqrt(mean_squared_error(self.data[:, i],
                                               self.irt_model(a=estimator['discrimination'][i],
                                                              b=estimator['difficulty'][i],
                                                              c=estimator['guessing'][i],
                                                              theta=estimator['theta']))) for i in range(len(estimator['discrimination']))]
        rmse_val_moy = [np.sqrt(mean_squared_error(self.data[:, i],
                                                   self.data.mean(axis=1))) for i in range(len(estimator['discrimination']))]

        def get_mean_std(array: np.array) -> Dict[str, float]:
            return {'mean': np.mean(array), 'std': np.std(array)}

        stats = {
            "Mean Exam accuracy": {'mean': 100 * self.data.mean(), 'std': 100 * self.data.mean(axis=1).std()},
            "Estimators":
                {
                "Discrimination (a)": get_mean_std(estimator['discrimination']),
                "Difficulty (b)": get_mean_std(estimator['difficulty']),
                "Guessing (c)": get_mean_std(estimator['guessing']),
                "Theta": get_mean_std(estimator['theta']),
            },
            'All Thetas': {stud.name: f"{estimator['theta'][i]:.02f} (Acc: {self.data.mean(axis=1)[i]:.02f})"
                           for i, stud in enumerate(self.students)},
            'RMSE':
                {
                'IRT Pred': get_mean_std(rmse_val),
                'Mean Pred Baseline': get_mean_std(rmse_val_moy),
            }
        }

        return stats


class HierarchicalItemResponseModel(BaseItemResponseModel):

    def __init__(self,
                 students: List[ExamSetting],
                 irt_model_type: int):

        super().__init__(students=students,
                         irt_model_type=irt_model_type)

    def compute_theta(self, theta_params: np.array) -> np.array:

        llm_params = theta_params[:self.num_llm]
        retrieval_params = theta_params[self.num_llm:self.num_llm + self.num_retrieval]
        icl_params = theta_params[self.num_llm + self.num_retrieval:]

        return np.array([(llm_params[self.llm_map[model.llm]]
                          + retrieval_params[self.retrieval_map[model.retrieval]]
                          + icl_params[model.icl])
                         for model in self.students])

    # Define the negative log-likelihood function for the 3PL model
    def hierarchical_neg_log_likelihood(self, params: np.array) -> float:
        a = params[:self.num_items]
        b = params[self.num_items:2 * self.num_items]
        c = params[2 * self.num_items:3 * self.num_items]
        theta = self.compute_theta(theta_params=params[3 * self.num_items:])

        likelihood = 0
        for i in range(self.num_items):
            p = self.irt_model(theta=theta, a=a[i], b=b[i], c=c[i])
            likelihood += np.sum(self.data[:, i] * np.log(p) + (1 - self.data[:, i]) * np.log(1 - p))

        # Add a param penalty
        # l2_penalty = lambda_l2 * np.sum(params ** 2)
        # return -likelihood + l2_penalty

        # Add Gaussian priors for a, b, and c
        # prior_a = np.sum(-0.5 * ((a - prior_means['a']) ** 2) / prior_vars['a'])
        # prior_b = np.sum(-0.5 * ((b - prior_means['b']) ** 2) / prior_vars['b'])
        # prior_c = np.sum(-0.5 * ((c - prior_means['c']) ** 2) / prior_vars['c'])

        return -likelihood

    def fit(self) -> Dict[str, np.array]:

        # Initial guesses for a, b, c and theta
        initial_guess = np.concatenate([
            np.ones(self.num_items),  # Initial guesses for a [discrimination]
            np.zeros(self.num_items),  # Initial guesses for b [difficulty]
            np.full(self.num_items, 0.25),  # Initial guesses for c [guessing]
            np.zeros(self.num_theta_params)  # Initial guesses for theta
        ])

        # b should be initialized at log((1-c)/(E[P]-c)
        # For hierarchical model, theta = theta_{LLM} + theta_{IR} + theta{ICL}
        # thus it's bounded in [-9, 9] and not [-3, 3]

        params_bounds = [
            [(0.5, 1.5) for _ in range(self.num_items)],  # Bounds for a [discrimination]
            [(0.01, 1) for _ in range(self.num_items)],  # Bounds for b [difficulty]
            [(0.2, 0.4) for _ in range(self.num_items)],  # Bounds  for c [guessing]
            [(-3, 3) for _ in range(self.num_theta_params)]  # Bounds for theta
        ]

        # Run optimization
        result = minimize(
            self.hierarchical_neg_log_likelihood,
            initial_guess,
            method='L-BFGS-B',
            bounds=[elem for bounds in params_bounds for elem in bounds]
        )

        return {
            'discrimination': result.x[:self.num_items],
            'difficulty': result.x[self.num_items:2 * self.num_items],
            'guessing': result.x[2 * self.num_items:3 * self.num_items],
            'theta_params': result.x[3 * self.num_items:],
            'theta': self.compute_theta(result.x[3 * self.num_items:])
        }

    def compute_stats(self, estimator: Dict[str, np.array]):

        # Hierachical Model Params
        llm_params = estimator['theta_params'][:self.num_llm]
        retrieval_params = estimator['theta_params'][self.num_llm:self.num_llm + self.num_retrieval]
        icl_params = estimator['theta_params'][self.num_llm + self.num_retrieval:]

        # Calculate the RMSE for each item
        rmse_val = [np.sqrt(mean_squared_error(self.data[:, i],
                                               self.irt_model(a=estimator['discrimination'][i],
                                                              b=estimator['difficulty'][i],
                                                              c=estimator['guessing'][i],
                                                              theta=estimator['theta']))) for i in range(len(estimator['discrimination']))]
        rmse_val_moy = [np.sqrt(mean_squared_error(self.data[:, i],
                                                   self.data.mean(axis=1))) for i in range(len(estimator['discrimination']))]

        def get_mean_std(array: np.array) -> Dict[str, float]:
            return {'mean': np.mean(array), 'std': np.std(array)}

        stats = {
            "Mean Exam accuracy": {'mean': 100 * self.data.mean(), 'std': 100 * self.data.mean(axis=1).std()},
            "Estimators":
                {
                "Discrimination (a)": get_mean_std(estimator['discrimination']),
                "Difficulty (b)": get_mean_std(estimator['difficulty']),
                "Guessing (c)": get_mean_std(estimator['guessing']),
                "Theta": get_mean_std(estimator['theta']),
            },
            'Theta': {
                'LLM': {k: f"{llm_params[i]:.02f} [+ {llm_params[i]-llm_params[0]:.02f}]"
                        for k, i in self.llm_map.items()},
                'Retrieval': {k: f"{retrieval_params[i]:.02f} [+ {retrieval_params[i]-retrieval_params[0]:.02f}]"
                              for k, i in self.retrieval_map.items()},
                'ICL': {f"ICL@{k}": f"{icl_params[k]:.02f} [+ {icl_params[k]-icl_params[0]:.02f}]"
                        for k in range(self.num_icl)},
            },
            'All Thetas': {stud.name: f"{estimator['theta'][i]:.02f} (Acc: {self.data.mean(axis=1)[i]:.02f})"
                           for i, stud in enumerate(self.students)},
            'RMSE':
                {
                'IRT Pred': get_mean_std(rmse_val),
                'Mean Pred Baseline': get_mean_std(rmse_val_moy),
            }
        }

        return stats
