# Exam Analysis

This folder contains several function and notebook utilies for the analysis of the generated exam.
In particular:

* **Item Response Theory Models**
  * `item_response_models.py` contains the classes for the base IRT model `BaseItemResponseModel` and for the `HierarchicalItemResponseModel`
  * `iterative_item_response_models.py` contains the class for the `IterativeHierarchicalItemResponseModel` described in section 6 of the paper.
  * `generate_irt_plots` allows to generate the IRT graphs and analysis results for your task of interest, using the previous classes.
* **Bloom's Taxonomy**
  * `bloom_taxonomy_model.py`: Automated classification of a question into Bloom's taxonomy criteria.
  * `taxonomy_analysis.ipynb`: Notebook to apply Bloom's taxonomy model to a given exam and study results.
* **General Utilities**
  * `compute_exam_radar_plot.ipynb` is a utility notebook to generate radar plot per categories of the exam performance.