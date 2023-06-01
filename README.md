Assessment of Pre-trained Observational Large Longitudinal models in OHDSI (APOLLO)
===================================================================================

[![Build Status](https://github.com/OHDSI/Apollo/workflows/Build-and-test/badge.svg)](https://github.com/OHDSI/Apollo/actions?query=workflow%3ABuild-and-test)

## Introduction
This Python package is for building and evaluating large general pre-trained models on data in the OMOP Common Data Model (CDM) format. The models are fitted on the structured data (concepts) in the CDM, not any natural language. We aim to evaluate these models on various tasks, such as patient-level prediction (either zero-shot or fine-tuned).

# Overview
This package assumes the [GeneralPretrainedModelTools](https://github.com/OHDSI/GeneralPretrainedModelTools) R package has been executed to retrieve (a sample of) the CDM data to local Parquet files. After this, a 'cdm_processor' must be run to convert the data to sequence data suitable for a large language model. TODO: how to go from here. 

## Getting Started

### Pre-requisite
The project is built in python 3.10, and project dependency needs to be installed 

Create a new Python virtual environment
```console
python -m venv venv;
source venv/bin/activate;
```

Install the packages in requirements.txt
```console
pip install -r requirements.txt
```

### Procesing CDM data for CEHR-BERT

1. Edit cehr-bert.ini to point to folders on the local file system.

2. Run:

    ```python
	python cdm_processing/cehr_bert_cdm_processor.py cehr_bert.ini
	```

## License

Apollo is licensed under Apache License 2.0.

## Development status

Under development. Do not use.
