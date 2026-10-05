# Industrial Predictive Maintenance MLOps Platform

An end-to-end MLOps platform for industrial predictive maintenance, designed to demonstrate Machine Learning Engineering, cloud, automation, CI/CD, and MLOps practices using Python, Databricks, Azure Machine Learning, MLflow, Feature Store, Azure Data Factory, and GitHub Actions.

## Project Overview

This project implements a predictive maintenance platform that receives machine measurements and estimates the probability of machine failure.

The system is designed around an end-to-end MLOps workflow rather than focusing only on model training.

The main objective is to demonstrate how a machine learning solution can be developed, tested, versioned, deployed, and orchestrated using cloud and MLOps technologies.

The project is specifically designed as a portfolio project aligned with the technical requirements and responsibilities described in a Machine Learning Engineer position at NTT DATA Europe & Latam.

---

## Objectives

The project aims to demonstrate the following capabilities:

- Develop reusable Python scripts and modules.
- Process and validate machine data using Databricks.
- Store processed data using Delta Lake.
- Create reusable machine learning features using a Feature Store.
- Train and evaluate a classification model for predictive maintenance.
- Track experiments and models using MLflow.
- Build a reproducible machine learning pipeline using Azure Machine Learning.
- Register and version machine learning models.
- Deploy a Batch Endpoint for batch inference.
- Orchestrate the end-to-end workflow using Azure Data Factory.
- Implement CI/CD pipelines using GitHub Actions.
- Use reusable GitHub Actions workflows.
- Manage Azure authentication through protected GitHub Secrets.
- Implement automated tests for code, data, and model behavior.
- Maintain reproducible and auditable machine learning workflows.

---

## Problem Statement

Unexpected machine failures can generate downtime, maintenance costs, and operational disruptions.

Predictive maintenance attempts to identify machines that are more likely to fail so that maintenance activities can be prioritized before a failure occurs.

This project uses machine measurements such as:

- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear

The machine learning system estimates the probability of failure from these measurements.

The prediction output will contain information such as:

- Failure probability
- Failure alert
- Prediction timestamp
- Input batch information

---

## Dataset

The project uses the **AI4I 2020 Predictive Maintenance Dataset**.

The dataset contains 10,000 synthetic observations representing industrial machine measurements.

The dataset includes measurements related to:

- Machine type
- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear

The target variable represents machine failure.

### Important Dataset Limitation

The AI4I 2020 dataset is synthetic and does not represent a real sequential stream of machine measurements.

Therefore, this project will simulate the arrival of new machine measurements through batch processing.

The purpose of the dataset is to demonstrate the technical MLOps architecture and workflow.

It should not be interpreted as evidence that the resulting model represents production performance on real industrial machines.

---

## MLOps Architecture

The target architecture is:

```text
                    AI4I 2020 Dataset
                            |
                            v
                 Python + Databricks
                            |
                            v
                       Delta Lake
                            |
                            v
                     Feature Store
                            |
                            v
                 Azure ML Pipeline
                    /           \
                   /             \
            Data Preparation   Model Training
                   \             /
                    \           /
                     Evaluation
                          |
                          v
                        MLflow
                          |
                          v
                    Model Registry
                          |
                          v
                    Batch Endpoint
                          |
                          v
                 Azure Data Factory
                          |
                          v
                    Predictions