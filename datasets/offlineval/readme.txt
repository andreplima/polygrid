Note:
-------------------------------------------------------------------------------------
Since the healthcare datasets used in the paper/thesis cannot be distributed with this software, we included the results of an offline evaluation using a couple of public multiclass datasets, so the user can tinker with the Polygrid/CLI and compare results.
The latter were selected by the number of attributes and, less strictly, the attributes representing some sort of physical measurement related to an implicit/explicit latent variable:
- from the UCI repo: iris (150 instances), wine (178 instances), cancer (Breast Cancer Winconsin - Diagnostic, 569 instances)
- Palmer's Penguins dataset (333 instances)
- from the KDIS repo: foodtruck (407 instances), water (1054 instances)
- from the Paderborn repo: iris@pb (150 instances), wine@pb (178 instances), vowel@pb (528 instances)

Content in ./multiclass
-------------------------------------------------------------------------------------
distribute_<timestamp>_<new or resume>.log: log file of the offline evaluation engine
display.log: log file of the process that analyses the results of the offline evaluation
multiclass_evaluation.csv: detailed data about the overall performance evaluation
multiclass_evaluation_rank.csv: detailed data about the analysis of the performance data
multiclass_results.xlsx: an Excel worksheet with the previous csv files imported + legend
multiclass_evaluation.pkl: the results tensor, saved as a pickled file.

Content in ./healthcare
-------------------------------------------------------------------------------------
The files follow the same structure described above for ./multiclass
There are several distribute*.log files because the process was resumed after disruptions in the Celery-based computational infrastructure we employed, without prejudice to the results.
