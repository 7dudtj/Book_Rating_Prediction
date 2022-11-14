import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


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
    elif x >= 60 and x < 70 :
        return 6
    else:
        return 7

def wdn_data_load(args):

    ######################## DATA LOAD
    users = pd.read_csv(args.DATA_PATH + 'users.csv')
    books = pd.read_csv(args.DATA_PATH + 'books.csv')

    train = pd.read_csv(args.DATA_PATH + 'train_ratings.csv')
    test = pd.read_csv(args.DATA_PATH + 'test_ratings.csv')
    sub = pd.read_csv(args.DATA_PATH + 'sample_submission.csv')

    ids = pd.concat([train['user_id'], sub['user_id']]).unique()
    isbns = pd.concat([train['isbn'], sub['isbn']]).unique()

    context = pd.concat([train, test]).reset_index(drop=True)

    users['location_city'] = users['location'].apply(lambda x: x.split(',')[0])
    users['location_state'] = users['location'].apply(lambda x: x.split(',')[1])
    users['location_country'] = users['location'].apply(lambda x: x.split(',')[2])
    users = users.drop(['location'], axis=1)

    # 인덱싱 처리된 데이터 조인
    context_df = context.merge(users[['user_id', 'location_city', 'location_state', 'location_country', 'age']], on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author', 'year_of_publication']], on='isbn', how='left')
    train = train.merge(users[['user_id', 'location_city', 'location_state', 'location_country', 'age']], on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author', 'year_of_publication']], on='isbn', how='left')
    test = test.merge(users[['user_id', 'location_city', 'location_state', 'location_country', 'age']], on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author', 'year_of_publication']], on='isbn', how='left')

    idx2user = {idx:id for idx, id in enumerate(ids)}
    idx2isbn = {idx:isbn for idx, isbn in enumerate(isbns)}

    user2idx = {id:idx for idx, id in idx2user.items()}
    isbn2idx = {isbn:idx for idx, isbn in idx2isbn.items()}

    train['user_id'] = train['user_id'].map(user2idx)
    sub['user_id'] = sub['user_id'].map(user2idx)
    test['user_id'] = test['user_id'].map(user2idx)

    train['isbn'] = train['isbn'].map(isbn2idx)
    sub['isbn'] = sub['isbn'].map(isbn2idx)
    test['isbn'] = test['isbn'].map(isbn2idx)

    # 인덱싱 처리
    loc_city2idx = {v:k for k,v in enumerate(context_df['location_city'].unique())}
    loc_state2idx = {v:k for k,v in enumerate(context_df['location_state'].unique())}
    loc_country2idx = {v:k for k,v in enumerate(context_df['location_country'].unique())}

    train['location_city'] = train['location_city'].map(loc_city2idx)
    train['location_state'] = train['location_state'].map(loc_state2idx)
    train['location_country'] = train['location_country'].map(loc_country2idx)
    test['location_city'] = test['location_city'].map(loc_city2idx)
    test['location_state'] = test['location_state'].map(loc_state2idx)
    test['location_country'] = test['location_country'].map(loc_country2idx)
    
    train['age'] = train['age'].fillna(int(train['age'].mean()))
    test['age'] = test['age'].fillna(int(test['age'].mean()))

    category2idx = {v:k for k,v in enumerate(context_df['category'].unique())}
    publisher2idx = {v:k for k,v in enumerate(context_df['publisher'].unique())}
    language2idx = {v:k for k,v in enumerate(context_df['language'].unique())}
    author2idx = {v:k for k,v in enumerate(context_df['book_author'].unique())}
    year2idx = {v:k for k,v in enumerate(context_df['year_of_publication'].unique())}

    train['category'] = train['category'].map(category2idx)
    train['publisher'] = train['publisher'].map(publisher2idx)
    train['language'] = train['language'].map(language2idx)
    train['book_author'] = train['book_author'].map(author2idx)
    train['year_of_publication'] = train['year_of_publication'].map(year2idx)
    test['category'] = test['category'].map(category2idx)
    test['publisher'] = test['publisher'].map(publisher2idx)
    test['language'] = test['language'].map(language2idx)
    test['book_author'] = test['book_author'].map(author2idx)
    test['year_of_publication'] = test['year_of_publication'].map(year2idx)

    idx = {
        "loc_city2idx":loc_city2idx,
        "loc_state2idx":loc_state2idx,
        "loc_country2idx":loc_country2idx,
        "category2idx":category2idx,
        "publisher2idx":publisher2idx,
        "language2idx":language2idx,
        "author2idx":author2idx,
        "year2idx":year2idx
    }

    field_dims = np.array([
                        len(user2idx),
                        len(isbn2idx),
                        99,
                        len(idx['loc_city2idx']),
                        len(idx['loc_state2idx']),
                        len(idx['loc_country2idx']),
                        len(idx['category2idx']), 
                        len(idx['publisher2idx']), 
                        len(idx['language2idx']), 
                        len(idx['author2idx']),
                        len(idx['year2idx'])
                           ], dtype=np.uint32)

    data = {
            'train':train,
            'test':test.drop(['rating'], axis=1),
            'field_dims':field_dims,
            'users':users,
            'books':books,
            'sub':sub,
            'idx2user':idx2user,
            'idx2isbn':idx2isbn,
            'user2idx':user2idx,
            'isbn2idx':isbn2idx,
            }

    return data

def wdn_data_split(args, data):
    X_train, X_valid, y_train, y_valid = train_test_split(
                                                        data['train'].drop(['rating'], axis=1),
                                                        data['train']['rating'],
                                                        test_size=args.TEST_SIZE,
                                                        random_state=args.SEED,
                                                        shuffle=True
                                                        )
    data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
    return data

def wdn_data_loader(args, data):
    train_dataset = TensorDataset(torch.LongTensor(data['X_train'].values), torch.LongTensor(data['y_train'].values))
    valid_dataset = TensorDataset(torch.LongTensor(data['X_valid'].values), torch.LongTensor(data['y_valid'].values))
    test_dataset = TensorDataset(torch.LongTensor(data['test'].values))

    train_dataloader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=args.DATA_SHUFFLE)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.BATCH_SIZE, shuffle=args.DATA_SHUFFLE)
    test_dataloader = DataLoader(test_dataset, batch_size=args.BATCH_SIZE, shuffle=False)

    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader

    return data
