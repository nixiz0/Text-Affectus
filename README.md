## Emotion AI Recognition App

Building a AI Agent that recognize emotions on sentences.


## Installation

=> You will need **Python 3.11.7** & **CUDA 11.8** (for training on your GPU the models)

=> Create a .env with ```python -m venv .env```

=> Activate your env

=> Run this command ```pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118```

=> Run ```pip install -r requirements.txt```


# Utilization

When you have built your models, the French and the English model (run the notebooks) and you have their weights then you just have to do this command to be able to integrate with the streamlit interface provided for : ```streamlit run interface.py```


## Author

- [@nixiz0](https://github.com/nixiz0)