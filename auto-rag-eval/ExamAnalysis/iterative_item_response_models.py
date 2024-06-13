# from scipy.stats import norm, lognorm, beta
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from ExamAnalysis.item_response_models import BaseItemResponseModel, ExamSetting
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error


class IterativeHierarchicalItemResponseModel(BaseItemResponseModel):

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

    def _fit(self, 
             initial_guess: np.array, 
             params_bounds: List)  -> Dict[str, np.array]:

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
    
    def _get_params_bounds(self) -> List:
        
        return [
            [(0.5, 1.5) for _ in range(self.num_items)],  # Bounds for a [discrimination]
            [(0.01, 1) for _ in range(self.num_items)],  # Bounds for b [difficulty]
            [(0.2, 0.4) for _ in range(self.num_items)],  # Bounds  for c [guessing]
            [(-3, 3) for _ in range(self.num_theta_params)]  # Bounds for theta
        ]
    
    def fit(self, 
            n_steps: int = 2,
            drop_ratio: float = 0.1) -> Dict[str, np.array]:
        
        estimator_dict = {}

        # Initial guesses for a, b, c and theta
        initial_guess = np.concatenate([
            np.ones(self.num_items),  # Initial guesses for a [discrimination]
            np.zeros(self.num_items),  # Initial guesses for b [difficulty]
            np.full(self.num_items, 0.25),  # Initial guesses for c [guessing]
            np.zeros(self.num_theta_params)  # Initial guesses for theta
        ])

        params = self._fit(initial_guess=initial_guess, 
                           params_bounds=self._get_params_bounds()) 
        
        estimator_dict[0] = params

        for step in range(1, n_steps):

            # Low-discrimation filtering, remove low self.drop_ratio % of questions
            percentile_value = np.percentile(params['discrimination'],
                                             drop_ratio)

            # Find the index of the closest value to this percentile in the array and filter it
            indices_to_remove = [k 
                                for k,v in enumerate(params['discrimination']) 
                                if v <= percentile_value]
            self.num_items -= len(indices_to_remove)


            # Round 1 guesses for a, b, c and theta
            updated_guess = np.concatenate([
                np.delete(params['discrimination'], indices_to_remove),
                np.delete(params['difficulty'], indices_to_remove),
                np.delete(params['guessing'], indices_to_remove),
                params['theta_params']
            ])

            params = self._fit(initial_guess=updated_guess, 
                               params_bounds=self._get_params_bounds())
            
            estimator_dict[step] = params

        return estimator_dict
        

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
    
    def plot_iterative_informativeness(self,
                                       estimator_dict: Dict[str, Dict[str, np.array]],
                                       exam_model: str,
                                       save_path: str = None,
                                       font_size: int = 18) -> None:
        
        # Set global font size
        plt.rcParams.update({'font.size': font_size})

        # Create an array of theta values for plotting
        theta_values = np.linspace(-3, 3, 300)

        # Create a 2x2 grid of subplots
        fig, ax = plt.subplots(figsize=(12, 8))

        # colors = ['red', 'green', 'blue', 'purple', 'orange']

        for step, estimator in estimator_dict.items():

            # Assume these are the estimated parameters for 3 items
            a = estimator['discrimination']  # Discrimination parameters
            b = estimator['difficulty']  # Difficulty parameters
            c = estimator['guessing']  # Guessing parameters

            test_information = np.zeros_like(theta_values)
            for i in range(len(a)):
                p = self.irt_model(theta=theta_values, a=a[i], b=b[i], c=c[i])
                information = a[i]**2 * p * (1 - p)
                test_information += information  # Sum up information from all items
            ax.plot(theta_values, test_information / len(a), label=f'Step {step}')

            # # Add markers on the x-axis for the estimated theta values
            # for k, theta in enumerate(estimator['theta']):
            #     color = colors[k % len(colors)]
            #     ax.scatter(theta, 0, marker='x', color=color)

        ax.set_title(f'Exam Information Curve  - {exam_model} Exam - {self.irt_model_type}PL Model')
        ax.set_xlabel('Theta (Ability)')
        ax.set_ylabel('Fisher Information')
        ax.legend()
        ax.grid(True)
        ax.grid(True)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
