from django.shortcuts import render, redirect
from django.urls import reverse
from urllib.parse import urlencode
from .views import *
import numpy as np 
import pandas as pd
import spacy
import string
from operator import itemgetter
import gensim
from gensim.similarities import MatrixSimilarity
import operator
import re
from spacy.lang.en.stop_words import STOP_WORDS
from wordcloud import WordCloud
from gensim import corpora
import os
from operator import itemgetter
import json
# import url parse

spacy_nlp = spacy.load('en_core_web_sm')

df=pd.read_csv('model/static/research_publication_tabular_format_details.csv')
login_df=pd.read_csv('model/static/login.csv')
#create list of punctuations and stopwords
punctuations = string.punctuation
stop_words = spacy.lang.en.stop_words.STOP_WORDS

#function for data cleaning and processing
#This can be further enhanced by adding / removing reg-exps as desired.

def spacy_tokenizer(sentence):
 
    #remove distracting single quotes
    sentence = re.sub('\'','',sentence)

    #remove digits adnd words containing digits
    sentence = re.sub('\w*\d\w*','',sentence)

    #replace extra spaces with single space
    sentence = re.sub(' +',' ',sentence)

    #remove unwanted lines starting from special charcters
    sentence = re.sub(r'\n: \'\'.*','',sentence)
    sentence = re.sub(r'\n!.*','',sentence)
    sentence = re.sub(r'^:\'\'.*','',sentence)
    
    #remove non-breaking new line characters
    sentence = re.sub(r'\n',' ',sentence)
    
    #remove punctunations
    sentence = re.sub(r'[^\w\s]',' ',sentence)
      
    #creating token object
    tokens = spacy_nlp(sentence)
    
    #lower, strip and lemmatize
    tokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in tokens]
    
    #remove stopwords, and exclude words less than 2 characters
    tokens = [word for word in tokens if word not in stop_words and word not in punctuations and len(word) > 2]
    
    #return tokens
    return tokens

def search(request):
    query=request.GET.get('search')
    if query is None:
        return render(request, 'index.html')
    
    df['Title of Article_tokenized'] = df['Title of Article'].map(lambda x: spacy_tokenizer(x))
    df_article = df['Title of Article_tokenized']
    df_article[0:5]
    series = pd.Series(np.concatenate(df_article)).value_counts()[:100]
    wordcloud = WordCloud(background_color='white').generate_from_frequencies(series)
    dictionary = corpora.Dictionary(df_article)

    stoplist = set('hello and if this can would should could tell ask stop come go')
    stop_ids = [dictionary.token2id[stopword] for stopword in stoplist if stopword in dictionary.token2id]
    dictionary.filter_tokens(stop_ids)
    dict_tokens = [[[dictionary[key], dictionary.token2id[dictionary[key]]] for key, value in dictionary.items() if key <= 50]]
    corpus = [dictionary.doc2bow(desc) for desc in df_article]
    word_frequencies = [[(dictionary[id], frequency) for id, frequency in line] for line in corpus[0:3]]
    #Load the indexed corpus
    article_tfidf_model = gensim.models.TfidfModel(corpus, id2word=dictionary)
    article_lsi_model = gensim.models.LsiModel(article_tfidf_model[corpus], id2word=dictionary, num_topics=300)
    article_tfidf_corpus = gensim.corpora.MmCorpus('article_tfidf_model_mm')
    article_lsi_corpus = gensim.corpora.MmCorpus('article_lsi_model_mm')
    gensim.corpora.MmCorpus.serialize('article_tfidf_model_mm', article_tfidf_model[corpus])
    gensim.corpora.MmCorpus.serialize('article_lsi_model_mm',article_lsi_model[article_tfidf_model[corpus]])
    article_index = MatrixSimilarity(article_lsi_corpus, num_features = article_lsi_corpus.num_terms)
    
    query_bow = dictionary.doc2bow(spacy_tokenizer(query))
    query_tfidf = article_tfidf_model[query_bow]
    query_lsi = article_lsi_model[query_tfidf]

    article_index.num_best = 10

    article_list = article_index[query_lsi]

    article_list.sort(key=itemgetter(1), reverse=True)
    faculty_detail = []

    for j, movie in enumerate(article_list):

        faculty_detail.append (
            {
                'Relevance': round((movie[1] * 100),2),
                'Author Name': df['Author Name'][movie[0]],
                'Title of Article': df['Title of Article'][movie[0]]
            }

        )
        if j == (article_index.num_best-1):
            break

    a=pd.DataFrame(faculty_detail, columns=['Relevance','Author Name','Title of Article'])
    print(a)
    AuthorName = a['Author Name'].values.tolist()
    TitleofArticle = a['Title of Article'].values.tolist()
    list_zip = zip(AuthorName, TitleofArticle)

    zipped_list = list(list_zip)

    context={'a':a,'author':AuthorName,'title':TitleofArticle,'zip':zipped_list}
    return render(request,"index.html",context)

def login(request):
    # df=pd.read_csv('model/static/login.csv')
    if request.method == 'POST':
        name = request.POST['author-name']
        password = request.POST['author-pass']
        # if name or password is None:
        #     print("name: {}\npassword: {} ".format(name,password))
        #     return redirect('login')
        author = login_df.loc[login_df['Author Name'].str.lower()==name.lower()]
        if not author.empty:
            if (author['Password'] == password).values[0]:
                # print("logged in")
                base_url = reverse('authorPage')  # 1 /products/
                query_string =  urlencode({'author': name})  # 2 category=42
                url = '{}?{}'.format(base_url, query_string)  # 3 /products/?category=42
                return redirect(url)  # 4
            else:
                return render(request,"login.html",{'error':True,'msg':'Incorrect password.'})
        else:
            return render(request,"login.html",{'error':True,'msg':'Incorrect username.'})
    else:
        return render(request,"login.html",{'error':False})

def authorPage(request):
    author_name=request.GET.get("author")
    authorDf = df.loc[df['Author Name'].str.lower()==author_name.lower()]
    author_data_list = authorDf.values.tolist()
    # print(author_data_list)
    r = json.dumps(author_data_list)
    custom_list = {
        "title_of_article":authorDf["Title of Article"].tolist(),
        "date_of_pub":authorDf["Date of Publication"].tolist()
    }
    print(json.dumps(custom_list))
    loaded_r = json.loads(r)
    return render(request,"author-page.html",{
        'author':author_name,
        'email':authorDf['Email'].iloc[0],
        'author_data':loaded_r,
        'author_list': json.dumps(custom_list)
    })



