# PankRAG
Modified from nano-graphrag

![overall workflow](./methodology_00.png)

# Contents

- Usage
  - Installation
  - Configuration
  - Running

## Usage

### Installation

```bash
conda create -n PankRAG python=3.10 -y  
conda activate PankRAG  
pip install -r requirements.txt
`````
### Configuration

Before running, please adjust or configure the corresponding settings in run.py and any other program files including the model name, API key, etc.

For your convenience, the datasets folder contains the organized dataset files, including the corpus and questions files.
### Running

```bash
python run.py
`````
