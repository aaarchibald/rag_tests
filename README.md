# Setup

The project requires *Python 3* and [uv](https://docs.astral.sh/uv/) for package / depedency management. 

### Install uv package manager: 
```
pip install uv
```

### Check the installation and show commands:
```
uv
```

### Update the project's environment
```
uv sync
```

### Install all libraries from a requirements file:
```
uv add -r requirements.txt
```

### Add new dependency
```
uv add
```


### Set Up and activate a Virtual Environment
```
uv venv
source .venv/bin/activate
```