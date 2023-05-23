Assessment of Pre-trained Observational Large Language-models in OHDSI (APOLLO)
===============================================================================

## Introduction
This Python package is for building and evaluating large general pre-trained models on data in the OMOP Common Data Model (CDM) format. The models are fitted on the structured data (concepts) in the CDM, not any natural language. We aim to evaluate these models on various tasks, such as patient-level prediction (either zero-shot or fine-tuned).

# Overview
This package assumes the [GeneralPretrainedModelTools](https://github.com/OHDSI/GeneralPretrainedModelTools) R package has been executed to retrieve (a sample of) the CDM data to local Parquet files. After this, a 'cdm_processor' must be run to convert the data to sequence data suitable for a large language model. TODO: how to go from here. 

## Python version

Currently developing against Pytohn 3.10

# License

Apollo is licensed under Apache License 2.0.

## Development status

Under development. Do not use.
