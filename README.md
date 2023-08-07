# ChatGPT with your own code repository

Proof-of-concept project for parsing your existing code base and asking ChatGPT about it. 

### Requirements

- Python 3.x
- OpenAI API token

### Installation

`pip install -r requirements.txt`

Copy `.env.example` to `.env` and put your OpenAI API key at `OPENAI_API_KEY` variable.

### Indexing Repository

- Clone your repository into `data/src` dir, i.e. `data/src/playwright`.
- Run indexing with `qa.py playwright 1`

It should take couple of minutes depending on the repo size.

### Running QA Chat

After indexing a repo start a chat with:

`./qa.py playwright`

Happy chatting!

### Chat Settings

To increase chat accuracy increase the number of matched documents to ChatGPT in `config.ini` file, i.e.:

```ini
[retriever]
k = 10
```

Keep in mind this will generate more usage (dollars) on your OpenAI plan.
