# Medical Time Series Transformer Model

## Introduction
This repository contains the code and resources for a machine learning model based on the Transformer architecture, designed for the analysis and prediction of medical time series data. The primary focus of this model is to work with lab values, utilizing the MIMIC-IV database, a publicly available dataset of electronic health records, as a case study.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Table of Contents

- [Introduction](#introduction)
- [Setup](#setup)
- [Usage](#usage)
- [Acknowledgments](#acknowledgments)
- [Contributing](#contributing)
- [License](#license)
- [Citing](#citing)

## Setup

#### 1. Clone this repository:

Copy code

```{bash}
git clone https://github.com/skylershapiro/Labrador_time_encoding.git
cd Labrador_time_encoding
```

#### 1.  Install dependencies:

In this case we used python 3.9.12.

Copy code

```{bash}
pip install -r requirements.txt
```

## Usage

### 1. Get the MIMIC-IV dataset.

1. Request access to the [MIMIC IV dataset](https://physionet.org/content/mimiciv/2.2/) in [PhysioNet](https://physionet.org/).

Documentation for MIMIC-IV's can be found [here](https://mimic.mit.edu/).

2. Integrate BigQuery with the project.

Since we'll use the python API to access big quety data, you'll need and account and a project in GCP. Follow the next steps:

* Create a [Google Cloud Platform Account](https://console.cloud.google.com/bigquery). 
* Enable the [BigQuery API](https://console.cloud.google.com/apis/api/bigquery.googleapis.com)
* Create a [Service Account](https://console.cloud.google.com/iam-admin/serviceaccounts)
* Create an API key in JSON format.
* Create a .env file with the command `nano .env` or `touch .env` for Mac and Linux users or `echo. >  .env` for Windows.
* Update your .env file with your ***JSON keys*** path.

```sh
KEYS_FILE = "Path/to/GoogleCloud_keys.json"
```

3. Get the lab values from MIMIC-IV dataset.

Run the Jupyter notebook in the src/get_data.ipynb or src/get_data.ip


### License
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
This project is licensed under the MIT License.


### Acknowledgments
MIMIC-IV Database


### Questions and Issues
If you have any questions or encounter issues with this project, please open a GitHub issue.

### Contributions
Contributions to improve and extend this project are welcome. Create a pull request or discuss potential contributions in the issues section.

### Citing
If you use this project in your research, please cite it using: 
*TODO*.
