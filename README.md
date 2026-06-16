# SGFT Project

## Setup

clone the repository:
```bash
git clone https://github.com/karad-tanmay/sgft-demo.git
cd sgft-project
```

create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

install the required dependencies:
```bash
pip install -r requirements.txt
```
create a .env file in root directory, refer to env_sample.txt

## Usage

create the data/raw dir: (if not already present)
```bash
mkdir -p data/raw
```

execute the data downloading script: (original dataset already being pushed to repo, so this step can be skipped)
```bash
python3 -m src.data.download_data
```

execute the data processing script: (processed data already being pushed to repo, so this step can be skipped)
```bash
python3 -m src.data.generate_sg
```