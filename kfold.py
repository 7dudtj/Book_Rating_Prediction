import time
import argparse
import pandas as pd
import numpy as np
import wandb

from sklearn.model_selection import KFold

from src import seed_everything

from src.data import fm_data_load, fm_data_loader
from src.data import ffm_data_load, ffm_data_loader
from src.data import ncf_data_load, ncf_data_loader
from src.data import wdn_data_load, wdn_data_loader

from src import FactorizationMachineModel, FieldAwareFactorizationMachineModel
from src import NeuralCollaborativeFiltering, WideAndDeepModel


def main(args):
    seed_everything(args.SEED)

    ######################## DATA LOAD
    print(f'--------------- {args.MODEL} Load Data ---------------')
    if args.MODEL in ('FM'):
        data = fm_data_load(args)
    elif args.MODEL in ('FFM'):
        data = ffm_data_load(args)
    elif args.MODEL in ('WDN'):
        data = wdn_data_load(args)
    elif args.MODEL in ('NCF'):
        data = ncf_data_load(args)
    else:
        pass

    ######################## SET KFOLD
    kf = KFold(n_splits=args.K_VALUE, shuffle=True, random_state=args.SEED)
    data_list = []
    predicts_list = []

    ######################## Train/Valid Split
    print(f'--------------- {args.MODEL} Train/Valid Split ---------------')
    if args.MODEL in ('FM', 'FFM', 'NCF', 'WDN'):
        X_data = np.array(data['train'].drop(['rating'], axis=1))
        y_data = np.array(data['train']['rating'])
        for train_idx, test_idx in kf.split(X_data):
            X_train, X_valid = X_data[train_idx], X_data[test_idx]
            y_train, y_valid = y_data[train_idx], y_data[test_idx]
            data['X_train'] = pd.DataFrame(X_train)
            data['X_valid'] = pd.DataFrame(X_valid)
            data['y_train'] = pd.DataFrame(y_train).squeeze()
            data['y_valid'] = pd.DataFrame(y_valid).squeeze()
            if args.MODEL in ('FM'):
                data = fm_data_loader(args, data)
            elif args.MODEL in ('FFM'):
                data = ffm_data_loader(args, data)
            elif args.MODEL in ('NCF'):
                data = ncf_data_loader(args, data)
            elif args.MODEL in ('WDN'):
                data = wdn_data_loader(args, data)
            else:
                pass
            data_list.append(data)
    else:
        pass

    ######################## MODEL, WANDB, TRAIN, INFERENCE
    for idx in range(args.K_VALUE):
        print(f'\n--------------- Iteration {idx} ---------------')

        ######################## MODEL
        print(f'--------------- INIT {args.MODEL} ---------------')
        if args.MODEL=='FM':
            model = FactorizationMachineModel(args, data_list[idx])
        elif args.MODEL=='FFM':
            model = FieldAwareFactorizationMachineModel(args, data_list[idx])
        elif args.MODEL=='NCF':
            model = NeuralCollaborativeFiltering(args, data_list[idx])
        elif args.MODEL=='WDN':
            model = WideAndDeepModel(args, data_list[idx])
        else:
            pass

        ######################## WANDB
        if args.WANDB:
            wandb.init(project="your-project", entity="your-entity")
            wandb.run.name =  args.MODEL
            wandb.config = {
                "learning_rate": args.LR ,
                "epochs": args.EPOCHS,
                "batch_size": args.BATCH_SIZE,
                "architecture": args.MODEL,
            }

        ######################## TRAIN
        print(f'--------------- {args.MODEL} TRAINING ---------------')
        model.train()
        
        if args.WANDB:
            wandb.finish()

        ######################## INFERENCE
        print(f'--------------- {args.MODEL} PREDICT ---------------')
        if args.MODEL in ('FM', 'FFM', 'NCF', 'WDN'):
            predicts = model.predict(data['test_dataloader'])
        elif args.MODEL=='CNN_FM':
            predicts  = model.predict(data['test_dataloader'])
        else:
            pass

        predicts_list.append(np.array(predicts))

    mean_predicts = sum(predicts_list) / len(predicts_list)
    
    ######################## SAVE PREDICT
    print(f'--------------- SAVE {args.MODEL} PREDICT ---------------')
    submission = pd.read_csv(args.DATA_PATH + 'sample_submission.csv')
    if args.MODEL in ('FM', 'FFM', 'NCF', 'WDN'):
        submission['rating'] = mean_predicts
    else:
        pass

    now = time.localtime()
    now_date = time.strftime('%Y%m%d', now)
    now_hour = time.strftime('%X', now)
    save_time = now_date + '_' + now_hour.replace(':', '')
    submission.to_csv('submit/{}_{}.csv'.format(save_time, args.MODEL), index=False)


if __name__ == "__main__":

    ######################## BASIC ENVIRONMENT SETUP
    parser = argparse.ArgumentParser(description='parser')
    arg = parser.add_argument

    ############### BASIC OPTION
    arg('--DATA_PATH', type=str, default='data/raw_data/', help='Data path를 설정할 수 있습니다.')
    arg('--MODEL', type=str, choices=['FM', 'FFM', 'NCF', 'WDN'],
                                help='학습 및 예측할 모델을 선택할 수 있습니다.')
    arg('--DATA_SHUFFLE', type=bool, default=True, help='데이터 셔플 여부를 조정할 수 있습니다.')
    arg('--TEST_SIZE', type=float, default=0.2, help='Train/Valid split 비율을 조정할 수 있습니다.')
    arg('--SEED', type=int, default=42, help='seed 값을 조정할 수 있습니다.')
    arg('--WANDB', type=bool, default=False, help='wandb 기록 여부를 선택할 수 있습니다.')
    arg('--K_VALUE', type=int, default=5, help='K-Fold의 K value 값을 조정할 수 있습니다.')

    ############### TRAINING OPTION
    arg('--BATCH_SIZE', type=int, default=1024, help='Batch size를 조정할 수 있습니다.')
    arg('--EPOCHS', type=int, default=10, help='Epoch 수를 조정할 수 있습니다.')
    arg('--LR', type=float, default=1e-3, help='Learning Rate를 조정할 수 있습니다.')
    arg('--WEIGHT_DECAY', type=float, default=1e-6, help='Adam optimizer에서 정규화에 사용하는 값을 조정할 수 있습니다.')

    ############### GPU
    arg('--DEVICE', type=str, default='cuda', choices=['cuda', 'cpu'], help='학습에 사용할 Device를 조정할 수 있습니다.')

    ############### FM
    arg('--FM_EMBED_DIM', type=int, default=16, help='FM에서 embedding시킬 차원을 조정할 수 있습니다.')

    ############### FFM
    arg('--FFM_EMBED_DIM', type=int, default=16, help='FFM에서 embedding시킬 차원을 조정할 수 있습니다.')

    ############### NCF
    arg('--NCF_EMBED_DIM', type=int, default=16, help='NCF에서 embedding시킬 차원을 조정할 수 있습니다.')
    arg('--NCF_MLP_DIMS', type=list, default=(16, 16), help='NCF에서 MLP Network의 차원을 조정할 수 있습니다.')
    arg('--NCF_DROPOUT', type=float, default=0.2, help='NCF에서 Dropout rate를 조정할 수 있습니다.')

    ############### WDN
    arg('--WDN_EMBED_DIM', type=int, default=16, help='WDN에서 embedding시킬 차원을 조정할 수 있습니다.')
    arg('--WDN_MLP_DIMS', type=list, default=(16, 16), help='WDN에서 MLP Network의 차원을 조정할 수 있습니다.')
    arg('--WDN_DROPOUT', type=float, default=0.2, help='WDN에서 Dropout rate를 조정할 수 있습니다.')
    
    args = parser.parse_args()
    main(args)
