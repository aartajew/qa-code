#!/usr/bin/env python3

from dotenv import load_dotenv

import configparser
import os
import sys

from colored import Fore, Style
from langchain.document_loaders import TextLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import DeepLake

load_dotenv()
config = configparser.ConfigParser()
config.read('config.ini')


def main(repo, reindex):
    if reindex:
        reindex_repo(repo)
    else:
        chat(repo)


def get_embeddings():
    return OpenAIEmbeddings(disallowed_special=())


def get_repo_path(repo):
    return f'./data/src/{repo}'


def get_db_path(repo):
    return f'./data/embeddings/{repo}'


def get_qa_chain(repo):
    model = ChatOpenAI(model_name=config['chat']['model'])
    qa = ConversationalRetrievalChain.from_llm(model, retriever=get_retriever(repo))
    return qa


def chat(repo):
    if not os.path.exists(get_db_path(repo)):
        print(f'{Fore.red}No index found at {get_db_path(repo)}, did you forget to run indexing first?{Style.reset}')
        sys.exit(1)

    qa = get_qa_chain(repo)
    cfg = config['chat']
    chat_history = []
    print(f'{Fore.white}Welcome to the chat! Ask you question or type quit, q or exit to finish.{Style.reset}')
    while True:
        query = input(f'{Fore.green}Prompt: {Style.reset}')
        if query in ['quit', 'q', 'exit']:
            sys.exit()
        if len(query) > cfg.getint('min_prompt_len'):
            result = qa({'question': query, 'chat_history': chat_history})
            chat_history.append((query, result['answer']))
            print(f"{Fore.yellow}{result['answer']}{Style.reset}")
        else:
            print(f"{Fore.red}Provide at least {cfg.getint('min_prompt_len')} characters{Style.reset}")


def get_retriever(repo):
    db = DeepLake(
        dataset_path=get_db_path(repo),
        read_only=True,
        embedding_function=get_embeddings(),
        verbose=True
    )
    cfg = config['retriever']
    result = db.as_retriever()
    result.search_kwargs['distance_metric'] = cfg.get('distance_metric')
    result.search_kwargs['fetch_k'] = cfg.getint('fetch_k')
    result.search_kwargs['maximal_marginal_relevance'] = cfg.getboolean('maximal_marginal_relevance')
    result.search_kwargs['k'] = cfg.getint('k')
    return result


def reindex_repo(repo):
    if not os.path.exists(get_repo_path(repo)):
        print(f'{Fore.red}No repo found at {get_repo_path(repo)}{Style.reset}')
        sys.exit(1)

    docs = []
    for dir_path, dir_names, filenames in os.walk(get_repo_path(repo)):
        for file in filenames:
            try:
                loader = TextLoader(os.path.join(dir_path, file), encoding='utf-8')
                docs.extend(loader.load_and_split())
            except Exception as e:
                pass
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)
    DeepLake.from_documents(texts, get_embeddings(), dataset_path=get_db_path(repo))


if __name__ == '__main__':
    if len(sys.argv) > 1:
        reindex_only = sys.argv[2].lower() in ('1', 'true') if len(sys.argv) > 2 else False
        main(repo=sys.argv[1], reindex=reindex_only)
    else:
        print(f'{Fore.red}Usage: ./qa.py <repo_name> [reindex_only: 1|true]{Style.reset}')
