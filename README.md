# LiteraryAI CLI


Консольний аналізатор художніх англомовних текстів.


## Установка


```bash
python -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
# за бажанням (краще якість, повільніше):
python -m spacy download en_core_web_trf