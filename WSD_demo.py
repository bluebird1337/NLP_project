#function
import torch
from transformers import pipeline
import nltk
from lemminflect import getAllInflections, getAllLemmas
from nltk.corpus import wordnet as wn
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import semcor
from IPython.display import clear_output
import warnings
warnings.filterwarnings("ignore")

# load data
unmasker = pipeline('fill-mask', model='bert-base-uncased', top_k=10)
clear_output()

# Preprocess the sentences
def preprocess(text):
    """
    input: a string
    output: a list
    - transform to lower case
    - remove the punctuation
    - seperate the words by blank
    """
    text = text.lower()
    punc = '!()-[]{};:"\,<">./?@#$%^&*_~'
    for p in punc: 
        text = text.replace(p, "")
    return text



def check_word_exist(st, base_word):
    """"
    若st 中有base_word的任何變形，回傳True
    """
    tokens = st.split(' ')
    vairation = getAllInflections(base_word)
    var_list = set()
    for types in vairation:
        for item in vairation[types]:
            var_list.add(item)
        
    for item in var_list:
        if item in tokens:
            return True
    return False

def put_mask(sentense, base_word):
    """
    把 [MASK] 放到第一個出現的 `base_word`各種變形
    """
    tokens = sentense.split(' ')
    vairation = getAllInflections(base_word)
    var_list = set()
    for types in vairation:
        for item in vairation[types]:
            var_list.add(item)
            
    rep_tokens = []
    mask = 0 # Only put mask on the first appeared base word
    for token in tokens:
        add = 0
        for item in var_list:
            if token == item and mask== 0:
                rep_tokens.append("[MASK]")
                add = 1
                mask += 1
        if add == 0:
            rep_tokens.append(token)

    res_sent = " ".join(rep_tokens)
    return res_sent, var_list

def get_candidates(sentense, base_word):
    """
    所有`base_word`的變形都不會納入candidates
    """
    sentense, var_list = put_mask(sentense, base_word)
    candidate = unmasker(sentense)
    result = {}
    for i in range(len(candidate)):
        same = 0
        for item in var_list:
            if candidate[i]['token_str'] == item:
                same = 1
        if same == 0:
            result[candidate[i]['token_str']] = candidate[i]['score']
    return result

# 檢查是否是AKL字
with open("data/noun.txt", 'r', encoding="utf-8") as f:
    noun = f.read().strip().split(', ')
with open("data/adj.txt", 'r', encoding="utf-8") as f:
    adj = f.read().strip().split(', ')
with open("data/adv.txt", 'r', encoding="utf-8") as f:
    adv = f.read().strip().split(', ')
with open("data/verb.txt", 'r', encoding="utf-8") as f:
    verb = f.read().strip().split(', ')
with open("data/others.txt", 'r', encoding="utf-8") as f:
    others = f.read().strip().split(', ')

AKL_merge = noun + adj + adv + verb + others
def check_akl(word):
    if word in AKL_merge:
        return True
    return False

#得到相似度分數(use wup_similarity)
def get_similarity_score(base_word, syn_word):
    """
    return mean similarity score of this two words
    compare all meaning
    """
    base_sets = wn.synsets(base_word)
    syn_sets = wn.synsets(syn_word)
    n = len(base_sets)
    m = len(syn_sets)
    score = 0
    for i in range(n):
        for j in range(m):
            try:
                score += base_sets[i].wup_similarity(syn_sets[j])
            except:
                pass
    try:
        score = score/ (n*m)
    except:
        score = score / 1
    return score

# 把分太細的POS 縮小分類
verb = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
adj = ['JJ', 'JJR', 'JJS']
adv = ['RB', 'RBR', 'RBS']
noun = ['NN', 'NNS', 'NNP', 'NNPS']

# 拿到句子中的詞性
def get_POS(sentense, target_word):
    """
    回傳 `target_word` 在 `sentense`中的詞性
    詞性種類: https://www.guru99.com/pos-tagging-chunking-nltk.html
    """
    # print("sentense: ", sentense)
    # print("target_word: ", target_word)
    tokens = nltk.word_tokenize(sentense)
    tag = nltk.pos_tag(tokens)
    # all variation 
    vairation = getAllInflections(target_word)
    var_list = set()
    for types in vairation:
        for item in vairation[types]:
            var_list.add(item)
    flag = 0
    mini_pos = ""
    # print(var_list)
    for tu in tag:
        for var in var_list:
            if tu[0] == var:
                mini_pos = tu[1]
                flag = 1
                break
        if flag == 1: # found
            break
    # print("mini_pos: ", mini_pos)
    pos = []
    if mini_pos in verb:
        pos.append("verb")
    if mini_pos in adj:
        pos.append("adj")
    if mini_pos in adv:
        pos.append("adv")
    if mini_pos in noun:
        pos.append("noun")
#     print(pos)
#     print('-'*10)
    return pos 

def same(cand_pos ,base_pos):
    for i in cand_pos:
        for j in base_pos:
            if i==j:
                return True
    return False        
    
def calculate_weight_ver2(cand, sentense, base_word):
    """
    input 1: the possible words dictionary
    input 2: the sentense used
    input 3: base word
    """
    # print(cand)
    data_items = cand.items()
    data_list = list(data_items)
    cand_df = pd.DataFrame(data_list, columns=['Words', 'Score'])
    
    # AKL part
    c_len = len(cand_df)
    for i in range(c_len):
        if check_akl(cand_df['Words'][i]):
            cand_df['Score'][i] = cand_df['Score'][i] *1.25
#             print("in AKL")
            
    # POS-tagging part
    base_pos = get_POS(sentense, base_word) #取得base的詞性
    vairation = getAllInflections(base_word)
    var_list = set()
    for types in vairation:
        for item in vairation[types]:
            var_list.add(item)
    # print("base_pos", base_pos)
    for i in range(c_len): 
        sen_tokens = sentense.split()
        for var in var_list:
            if var in sen_tokens:
                sent_temp = sentense.replace(var, cand_df['Words'][i])
                break
        cand_pos = get_POS(sent_temp, cand_df['Words'][i]) #取得candidate的詞性
        # print("cand", cand_df['Words'][i])
        # print("cand_pos", cand_pos)
        # print("sent_temp: ", sent_temp)
        if same(cand_pos ,base_pos):
            cand_df['Score'][i] = cand_df['Score'][i] *1.5
            # print("Same type")
#         else:
#             print('Not Same type')
    # Wordnet Similarity
    for i in range(c_len):
        cand_df['Score'][i] += get_similarity_score(base_word, cand_df['Words'][i])
    
    cand_df = cand_df.sort_values(by=['Score'], ascending=False).reset_index(drop=True)
    return cand_df

# 找兩個字最近的字義
def find_sense_of_two_words(base_word, syn_word):
    base_word = wn.synsets(base_word) #可增加詞性 base_word = wn.synsets(base_word, pos=wn.VERB)  [VERB, NOUN, ADJ, ADV]
    syn_word = wn.synsets(syn_word) #可增加詞性 syn_word = wn.synsets(syn_word, pos=wn.VERB)  [VERB, NOUN, ADJ, ADV]
#     print("-"*10)
#     print("find_sense_of_two_words")
#     print("base_word", base_word)
#     print("syn_word", syn_word)
#     print("-"*10)
    wup_similarity=[]
    wup_similarity_dict={}
    for i in base_word:
        for j in syn_word:
            if wn.wup_similarity(i, j) != None:
                wup_similarity.append(wn.wup_similarity(i, j))
                wup_similarity_dict[wn.wup_similarity(i, j)]=[i,j]
    # print("wup_similarity", wup_similarity)
    # print("wup_similarity_dict", wup_similarity_dict)  
    
    #找出相似度最大的值與sense    
    similarity = max(wup_similarity)
    #sense編號 
    sense= wup_similarity_dict[max(wup_similarity)][0]
    #字義
    definition = wup_similarity_dict[max(wup_similarity)][0].definition()
    
#     sense1 = wup_similarity_dict[max(wup_similarity)][0].definition()
#     sense2 = wup_similarity_dict[max(wup_similarity)][1].definition()
#     print("sense1: ", wup_similarity_dict[max(wup_similarity)][0], sense1)
#     print("sense2: " ,  wup_similarity_dict[max(wup_similarity)][1], sense2)

    return similarity, sense, definition  #propose和need相似度, propose和need相似度最接近的sense編號, 字義 


def summary(sentence, base_word):
    """
    輸入: sentence, target word
    輸出: target word/ 例句/ 在此例句中找到最近的詞比對出來的字義 
    """
    sentence = preprocess(sentence)
    cand = get_candidates(sentence, base_word) # 找出可能的答案 
    # print("candidate", cand)
    result_df = calculate_weight_ver2(cand, sentence, base_word) # 加權
    r_len = len(result_df['Words'])
    for i in range(r_len):
        if(len(wn.synsets(result_df['Words'][i])))!= 0:
            syn_final_word = result_df['Words'][i] # 拿第一名的結果
            break
    # print("base_word: ", base_word, "syn_final_word", syn_final_word)
    similarity, sense, definition = find_sense_of_two_words(base_word, syn_final_word) # 找最近的字義
    
    similar = sense.lemma_names()
    filter = []
    for tmp in similar:
        if tmp!=base_word:
            filter.append(tmp)
    result = '、'.join(filter)
    
    # 印出結果
    print(f"""
    Target Word：{base_word}
    例句：{sentence}
    --------------------
    在此例句中 "{base_word}" 的字義是：{definition} 
    
    以下列出同義字：{result}
    """)
    return sense, definition, result

#streamlit app    

import streamlit as st

st.title('AKL Word Sense Disambiguation')
#在網頁呈現 h1 大小的文字
st.header('針對論文與期刊中AKL文字之字義解歧')

st.write('論文資料來源: https://github.com/allenai/s2orc')

#在網頁呈現 h2 大小的文字
st.write('自然語言處理 組員：施、王、徐')
st.write('------------------------------------------------------------')

#在網頁呈現文字

#title = st.text_input('輸入論文檔案、文字', 'Factor Analysis for Game Software Using Structural Equation Modeling with Hierarchical Latent Dirichlet Allocation in User’s Review Comments')
#st.write('論文題目', title)

#sidebar word,paper
select_word = st.sidebar.selectbox(
    "AKL word list",
    ("propose", "need", "absolute", "physical", "clear", "function", "degree", "character",
     "loss", "interest", "class", "kind", "act", "seek", "operation")
)

select_paper = st.sidebar.selectbox(
    "Paper list",
    ("Paper1", "Paper2", "Paper3", "Paper4", "Paper5","Paper6","Paper7","Paper8","Paper9","Paper10","Paper11","Paper12","Paper13","Paper14")
)


#顯示出對應的paper內容
with open('essay_0.txt', 'r', encoding='utf-8') as f:
     essay = f.readlines()
paper_list=["Paper1", "Paper2", "Paper3", "Paper4", "Paper5","Paper6","Paper7","Paper8","Paper9","Paper10","Paper11","Paper12","Paper13","Paper14"]
for i,paper in enumerate(paper_list):
     if select_paper == paper :
          st.write(paper)
          st.write(essay[i])
               

#輸入要測試的句子
sent_iunput = st.text_input('擷取的句子：')


#選擇AKL字詞
AKL_word_list=["propose", "need", "absolute", "physical", "clear", "function", "degree", "character",
     "loss", "interest", "class", "kind", "act", "seek", "operation"]
for word in AKL_word_list:
     if select_word == word:
          base_word=word 

#按鈕跑出選擇的AKL
# if st.button('Find Sense'):
#      st.write(base_word)

st.write('------------------------------------------------------------')

if st.button('Find Sense'):
     answer, definition, result = summary(sent_iunput, base_word)
     st.write('相關屬性:', answer)
     st.write('字義：', definition)
     st.write('同義詞：', result)

#針對特定字詞標記顏色
# import streamlit as st
# from annotated_text import annotated_text

#
# a = ['this', 'is' ,'a' ,'propose']
# for index,j in enumerate(a):
#     if "propose" in j :
#           a[index] = annotated_text("propose"," ", "#8ef")

#word = annotated_text(i for i in a)
#b = annotated_text(str(i) for i in a)





# 標顏色
# for i,paper in enumerate(paper_list):
#      essay_word = essay[i].strip().split()
#      for index,j in enumerate(essay_word):
#           if "propose" in j :
#                essay_word[index] = ("propose"," ", "#8ef")


# a = annotated_text(
#     "This ",
#     ("is","1", "#8ef"),
#     " some ",
#     ("annotated", "2", "#faa"),
#     ("text", "3", "#afa"),
#     " for those of ",
#     ("you", "4", "#fea"),
#     " who ",
#     ("like", "5", "#8ef"),
#     " this sort of ",
#     ("thing", "6", "#afa"),
# )


# definition