import streamlit as st
import torch
import pandas as pd
import torch

from torch.utils.data import TensorDataset, DataLoader


# SETTINGS
st.set_page_config(layout='wide')
PATH = 'data/data2/'
device = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache
def load_model():
    model = torch.load('./models/FM_model.pt', map_location=device)
    return model

def age_map(x: int) -> int:
    x = int(x)
    if x < 20:
        return 1
    elif x >= 20 and x < 30:
        return 2
    elif x >= 30 and x < 40:
        return 3
    elif x >= 40 and x < 50:
        return 4
    elif x >= 50 and x < 60:
        return 5
    else:
        return 6

def process_fm_data(users, books, ratings1, ratings2):
    users['location_country'] = users['location'].apply(lambda x: x.split(',')[2])
    users = users.drop(['location'], axis=1)

    ratings = pd.concat([ratings1, ratings2]).reset_index(drop=True)

    # 인덱싱 처리된 데이터 조인
    context_df = ratings.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'language', 'book_author']], on='isbn', how='left')
    test_df = ratings2.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'language', 'book_author']], on='isbn', how='left')

    # 인덱싱 처리
    loc_country2idx = {v:k for k,v in enumerate(context_df['location_country'].unique())}
    test_df['location_country'] = test_df['location_country'].map(loc_country2idx)
    test_df['age'] = test_df['age'].fillna(int(test_df['age'].mean()))
    test_df['age'] = test_df['age'].apply(age_map)

    # book 파트 인덱싱
    category2idx = {v:k for k,v in enumerate(context_df['category'].unique())}
    language2idx = {v:k for k,v in enumerate(context_df['language'].unique())}
    author2idx = {v:k for k,v in enumerate(context_df['book_author'].unique())}

    test_df['category'] = test_df['category'].map(category2idx)
    test_df['language'] = test_df['language'].map(language2idx)
    test_df['book_author'] = test_df['book_author'].map(author2idx)

    return test_df

def fm_data_load():

    ######################## DATA LOAD
    users = pd.read_csv(PATH + 'users.csv')
    books = pd.read_csv(PATH+ 'books.csv')
    train = pd.read_csv(PATH + 'train_ratings.csv')
    test = pd.read_csv(PATH + 'streamlit_test_ratings.csv')
    sub = pd.read_csv(PATH + 'streamlit_sample_submission.csv')

    ids = pd.concat([train['user_id'], sub['user_id']]).unique()
    isbns = pd.concat([train['isbn'], sub['isbn']]).unique()

    idx2user = {idx:id for idx, id in enumerate(ids)}
    idx2isbn = {idx:isbn for idx, isbn in enumerate(isbns)}

    user2idx = {id:idx for idx, id in idx2user.items()}
    isbn2idx = {isbn:idx for idx, isbn in idx2isbn.items()}

    train['user_id'] = train['user_id'].map(user2idx)
    sub['user_id'] = sub['user_id'].map(user2idx)
    test['user_id'] = test['user_id'].map(user2idx)
    users['user_id'] = users['user_id'].map(user2idx)

    train['isbn'] = train['isbn'].map(isbn2idx)
    sub['isbn'] = sub['isbn'].map(isbn2idx)
    test['isbn'] = test['isbn'].map(isbn2idx)
    books['isbn'] = books['isbn'].map(isbn2idx)

    context_test = process_fm_data(users, books, train, test)

    data = {
            'test':context_test.drop(['rating'], axis=1),
            }

    return data

def fm_data_loader(data):
    test_dataset = TensorDataset(torch.LongTensor(data['test'].values))
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return test_dataloader

def main():
    st.title("Book Rating Prediction Model")
    st.header("책을 사기 전에, 취향에 맞는 책일지 미리 알려드릴게요!")

    model = load_model()

    if 'id' not in st.session_state:
        st.session_state.id = ""
    if 'isbn' not in st.session_state:
        st.session_state.isbn = ""

    st.session_state.id = str(st.text_input('User ID를 입력하세요.\
    ex) 8, 278529, 168809, 67544 ...', key='user_id'))
    st.session_state.isbn = str(st.text_input('Book ISBN을 입력하세요.\
    ex) 1558532331, 080075672X, 1557489602, 2266046497 ...', key='book_isbn'))

    id_input = st.session_state['user_id']
    isbn_input = st.session_state['book_isbn']

    if st.button("평점 예측하기") and id_input and isbn_input:
        st.write(f"유저 {id_input}의 책 {isbn_input}에 대한 평점을 예측중이에요...")
        streamlit_test_df = pd.DataFrame({
            'user_id': [id_input],
            'isbn': [isbn_input],
            'rating': [0]
        })
        streamlit_test_df.to_csv('./data/data2/streamlit_test_ratings.csv', index=False)
        streamlit_test_df.to_csv('./data/data2/streamlit_sample_submission.csv', index=False)
        data = fm_data_load()
        data = fm_data_loader(data)
        
        with torch.no_grad():
            model.eval()
            for fields in data:
                fields = fields[0].to(device)
                pred = model(fields)
                pred = round(pred.item(), 2)

        st.write(f'예측 평점: {pred}')
        if pred < 4:
            st.write(f"책 {isbn_input}은 고객님 취향에 맞지 않을것 같아요. 다른 책을 찾아보시겠어요?")
        elif pred < 7:
            st.write(f"책 {isbn_input}은 고객님 취향에 맞을거같아요! 하지만 더 잘 맞는 책이 있을수도 있어요!")
        else:
            st.write(f"책 {isbn_input}은 고객님 취향 100% 저격! 지금 바로 구매하세요!")


if __name__ == "__main__":
    main()
    