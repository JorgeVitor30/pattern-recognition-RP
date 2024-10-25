RP
==============================

Implementing ML Algorithms

Project Organization
------------

```
RP/
├── LICENSE     
├── README.md                  
├── Makefile                     # Makefile with commands like `make data` or `make train`                   
├── configs                      # Config files (models and training hyperparameters)
│   └── model1.yaml              
│
├── data                         
│   ├── processed                # The final, canonical data sets for modeling.
│   └── raw                      # The original, immutable data dump.
│
├── docs                         # Project documentation.
│
├── models                       # Trained and serialized models.
│
├── notebooks                    # Jupyter notebooks.
││
├── reports                      # Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures                  # Generated graphics and figures to be used in reporting.
│
├── requirements.txt             # The requirements file for reproducing the analysis environment.
├── src                          # Source code for use in this project.
│    ├── _init__.py              # Makes src a Python module.
│    │
│    ├── data                     # Data engineering scripts.      
│    │
│    ├── models                   # ML model engineering (a folder for each model).
│    │   └── model1      
│    │       ├── model.py                
│    │
│    └── visualization        # Scripts to create exploratory and results oriented visualizations.│
└── main.py
```


--------
<p><small>Project based on the <a target="_blank" href="https://github.com/Chim-SO/cookiecutter-mlops/">cookiecutter MLOps project template</a>
that is originally based on <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. 
#cookiecuttermlops #cookiecutterdatascience</small></p>
