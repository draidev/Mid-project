import pandas as pd
import numpy as np
import re
from konlpy.tag import Okt
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import MinMaxScaler
import dill as pickle
from PIL import Image
from urllib import request
from io import BytesIO
import streamlit as st

st.set_page_config(layout="wide")
st.title("강아지 사료 검색")

@st.cache(allow_output_mutation=True)
def load_data():
    data_df = pd.read_excel('./data_df_full.xlsx')
    tfidf_vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
#     data_df_func = pd.read_excel('./data_df_func.xlsx')
#     data_df_target= pd.read_excel('./data_df_target.xlsx')
#     data_df_ingredient= pd.read_excel('./data_df_ingredient.xlsx')
#     data_df_cat= pd.read_excel('./data_df_cat.xlsx')
#     data_df_grade= pd.read_excel('./data_df_grade.xlsx')
    
#     return data_df,tfidf_vectorizer,data_df_func,data_df_target,data_df_ingredient,data_df_cat,data_df_grade
    return data_df, tfidf_vectorizer

data_df,tfidf_vectorizer = load_data()
# data_df,tfidf_vectorizer,data_df_func,data_df_target,data_df_ingredient,data_df_cat,data_df_grade = load_data()

# 복수값을 허용하는 열에 대해 one-hot-encoding
@st.cache(allow_output_mutation=True)
def feat_to_vec(data_df, col):
    items = set(",".join(data_df[col]).replace(" ","").split(','))
    data_df_col = pd.DataFrame(columns = list(items))

    for i in range(len(data_df[col])):
        new_tuple = dict(zip(list(items), [0]*len(list(items)))) 
        for item_have in data_df[col][i].replace(" ","").split(','):
            for item_candidate in list(items):
                if item_have == item_candidate:
                    new_tuple[item_candidate]=1
        data_df_col = data_df_col.append(new_tuple, ignore_index=True)
        
    return data_df_col

data_df_func = feat_to_vec(data_df,'기능')
data_df_target = feat_to_vec(data_df,'급여대상')
data_df_ingredient = feat_to_vec(data_df,'주원료')
data_df_cat = feat_to_vec(data_df,'분류')
data_df_grade =  feat_to_vec(data_df,'등급')
for i in range(len(data_df_target)):
    if data_df_target['전연령'][i]==1:
        data_df_target['시니어'][i]=1
        data_df_target['퍼피'][i]=1
        data_df_target['어덜트'][i]=1           
        
data_df_target = data_df_target.drop('전연령', axis=1)

@st.cache(allow_output_mutation=True)
def make_cleaned_corpus(str):
    tokenizer = Okt()
    del_list = ['kg'] 
    raw_pos_tagged = tokenizer.pos(str, norm=True, stem=True)
    word_cleaned = []
    for word in raw_pos_tagged:
        if not word[1] in ["Josa", "Eomi", "Punctuation", "Foreign", "Number"]: # Foreign == ”, “ 와 같이 제외되어야할 항목들
            if word[0] not in del_list: # 한 글자로 이뤄진 단어들을 제외 & 원치 않는 단어들을 제외
                word_cleaned.append(word[0])
    result = " ".join(word_cleaned)
    return result

@st.cache(allow_output_mutation=True)
def make_tfidf_matrix(df):
    cleaned_name = []
    for i in range(len(df)):
        word_cleaned = make_cleaned_corpus(df.iloc[i])
        cleaned_name.append(word_cleaned)
    tfidf_matrix = tfidf_vectorizer.transform(cleaned_name).todense()
    
    return tfidf_matrix

tfidf_matrix = make_tfidf_matrix(data_df['이름'])

@st.cache(allow_output_mutation=True)
def get_score_on_col(query, data_df_col):
    cleaned_query = make_cleaned_corpus(query)
    querys = cleaned_query.split(' ')
    items = list(data_df_col.columns)
    query_tuple = dict(zip(items, [0]*len(items)))
    
    for query in querys:
        for item in items:
            if query in item:
                query_tuple[item]=1 # 두번 입력해도 가중 X
    
    scores = []
    for i in range(len(data_df_col)):
        score = 0
        for item in items:
            if query_tuple[item]==1 & data_df_col[item][i]==1:
                score +=1
        scores.append(score)
        
        scaler = MinMaxScaler()
        result = scaler.fit_transform(np.array(scores).reshape(len(scores),1))
        result = result.reshape(len(result))
    return result

@st.cache(allow_output_mutation=True)
def dogfood_search(query):
    cleaned_query = make_cleaned_corpus(query)
    query_tfidf = tfidf_vectorizer.transform([cleaned_query]).todense()

    name_cosine_sim = cosine_similarity(query_tfidf, tfidf_matrix)[0]
    func_scores = get_score_on_col(query, data_df_func)
    target_scores = get_score_on_col(query, data_df_target)
    ingredient_scores = get_score_on_col(query, data_df_ingredient)
    cat_scores = get_score_on_col(query, data_df_cat)
    grade_scores = get_score_on_col(query, data_df_grade)


    a = 3 # 이름 가중치
    b = 1 # 기능 가중치
    c = 1 # 급여대상 가중치
    d = 1 # 주원료 가중치
    e = 1 # 분류 가중치
    f = 1 # 등급 가중치
    total_score =(a*name_cosine_sim+b*func_scores+c*target_scores+d*ingredient_scores+e*cat_scores+f*grade_scores)/(a+b+c+d+e+f)
    result = pd.concat([data_df, pd.DataFrame(total_score, columns = ['score'])], axis = 1)
    result = result.sort_values('score',ascending=False)
    
    return result


def search(result, start_idx, end_idx):
    sub_df = result.iloc[start_idx:end_idx]
    for i in range(len(sub_df)):
        con = st.container()
        name = sub_df['이름'].iloc[i]
        link = sub_df['링크'].iloc[i]
        con.subheader(f'[{name}]({link})')
        col1, col2 = con.columns([1,3])
        with col1:
            url = sub_df['이미지'].iloc[i]
            res = request.urlopen(url).read()
            image = Image.open(BytesIO(res))
            st.image(image,use_column_width='always')
        with col2:
            st.write('가격: '+str(sub_df['가격'].iloc[i])+"원")
            st.write('중량: '+str(sub_df['중량'].iloc[i]/1000)+"kg")
            st.write('등급: ',sub_df['등급'].iloc[i])
            st.write('급여대상: ',sub_df['급여대상'].iloc[i])
            st.write('기능: ',sub_df['기능'].iloc[i])
            st.write('주원료: ',sub_df['주원료'].iloc[i])
            st.caption("리뷰건수: "+str(sub_df['리뷰건수'].iloc[i]))
    return

def filtering(result, target_selected_options, func_selected_options, ing_selected_options, price_selected_options, sort_by):
    for i in range(len(target_selected_options)):
        result = result[result['급여대상'].str.contains(target_selected_options[i])]
    for i in range(len(func_selected_options)):
        result = result[result['기능'].str.contains(func_selected_options[i])]
    for i in range(len(ing_selected_options)):
        result = result[result['주원료'].str.contains(ing_selected_options[i])]
    for i in range(len(grade_selected_options)):
        result = result[result['등급'].str.contains(grade_selected_options[i])]
    result = result[(result['가격']>=price_selected_options[0]) & (result['가격']<=price_selected_options[1])]
    if sort_by == "관련도순":
        result = result.sort_values('score', ascending = False)[:100]
    elif sort_by == "가격낮은순":
        result = result.sort_values('score', ascending = False)[:100].sort_values('가격', ascending = True)
    elif sort_by == "가격높은순":
        result = result.sort_values('score', ascending = False)[:100].sort_values('가격', ascending = False)
    else:
        result = result.sort_values('score', ascending = False)[:100].sort_values('리뷰건수', ascending = False)
    return result

# =================== Side Bar ======================
logo = Image.open('./img/likelion.png')
st.sidebar.image(logo, width = 50)
st.sidebar.header("Team 육져따리!")
st.sidebar.subheader("필터 적용")

target_selected_options = st.sidebar.multiselect("급여대상:",
    list(data_df_target.columns))
func_selected_options = st.sidebar.multiselect("기능:",
    list(data_df_func.columns))
ing_selected_options = st.sidebar.multiselect("주원료:",
    list(data_df_ingredient.columns))
grade_selected_options = st.sidebar.multiselect("등급:",
    list(data_df_grade.columns))

price_selected_options = st.sidebar.slider('가격:', 0, 500000, (0, 500000),step=1000)

sort_by = st.sidebar.radio("정렬기준:", ("관련도순", "가격낮은순","가격높은순", "리뷰건수순"))





# ==================== Search Box =========================
col1, col2 = st.columns(2)
with col1:
    query = st.text_input(label="검색어를 입력해주세요", key='query', value="")
    search_button = st.button("검색")
con = st.container()
con.caption("Result")

# ==================== When Search button Clicked ==========================
if 'page_number' not in st.session_state:
    st.session_state.page_number = 0
    
N = 5
prev, _ ,nxt = st.columns([1, 10, 1])
nxt_button = nxt.button("Next")
prev_button = prev.button("Previous")

if search_button:
    result = dogfood_search(query)
    result = filtering(result,target_selected_options, func_selected_options, ing_selected_options, price_selected_options, sort_by)
    st.session_state.page_number = 0
    if 'last_page' not in st.session_state:
        st.session_state.last_page = len(result) // N
    start_idx = st.session_state.page_number * N 
    end_idx = (1 + st.session_state.page_number) * N 
    search(result,start_idx,end_idx)
        
    
if nxt_button:
    result = dogfood_search(query)
    result = filtering(result,target_selected_options, func_selected_options, ing_selected_options, price_selected_options, sort_by)

    st.session_state.last_page = len(result) // N
    if st.session_state.page_number + 1 > st.session_state.last_page:
        st.session_state.page_number = 0
    else:
        st.session_state.page_number += 1
    start_idx = st.session_state.page_number * N 
    end_idx = (1 + st.session_state.page_number) * N
    search(result,start_idx,end_idx)

if prev_button:
    result = dogfood_search(query)
    result = filtering(result,target_selected_options, func_selected_options, ing_selected_options, price_selected_options, sort_by)

    st.session_state.last_page = len(result) // N
    if st.session_state.page_number - 1 < 0:
        st.session_state.page_number = st.session_state.last_page
    else:
        st.session_state.page_number -= 1

    start_idx = st.session_state.page_number * N 
    end_idx = (1 + st.session_state.page_number) * N 
    search(result,start_idx,end_idx)


