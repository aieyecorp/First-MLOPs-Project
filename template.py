import logging
import os
from pathlib import Path

list_of_files=[
        ".github/workflows/.gitkeep",
        "src/__init__.py",
        "src/components/__init__.py",
        "src/components/data_ingestion.py",
        "src/components/model_trainer.py",
        "src/components/model_evalution.py",
        "src/components/data_ingestion.py",
        "src/pipeline/__init__.py",
        "src/pipeline/training_pipeline.py",
        "src/pipeline/prediction_pipeline.py",
        "src/utils/utils.py",
        "src/logger/logging.py",
        "src/exception"
        "tests/unit/__init__.py",
        "tests/integration/__init__.py",
        "init_setup.sh",
        "requirements.txt",
        "requirements_dev.txt",
        "setup.cfg",
        "setup.py",
        "pyproject.toml",
        "tox.ini",
        "experiment/experiments.ipynb"
        ]

#TODO Create files
for filepath in list_of_files:
    file_path=Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"creating directory : {filedir} for file: {filename}")

    if not os.path.exists(filepath) or os.path.getsize(filepath)==0:
        with open(filepath, "w") as f:
            pass
