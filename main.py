import argparse
import wandb

from src import seed_everything

from src.data import fm_data_load, fm_data_split, fm_data_loader
from src.data import ffm_data_load, ffm_data_split, ffm_data_loader
from src.data import ncf_data_load, ncf_data_split, ncf_data_loader
from src.data import wdn_data_load, wdn_data_split, wdn_data_loader

from src import FactorizationMachineModel, FieldAwareFactorizationMachineModel
from src import NeuralCollaborativeFiltering, WideAndDeepModel

def predicts_map(x: float) -> float:
    if x < 1:
        return 1.0
    elif x > 10:
        return 10.0
    else:
        return x
        
def main(args):
    seed_everything(args.SEED)

    ######################## SET WANDB
    if args.WANDB:
        wandb.init(project="your-project", entity="your-entity")
        wandb.run.name =  args.MODEL
        wandb.config = {
            "learning_rate": args.LR ,
            "epochs": args.EPOCHS,
            "batch_size": args.BATCH_SIZE,
            "architecture": args.MODEL,
        }

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

    ######################## Train/Valid Split
    print(f'--------------- {args.MODEL} Train/Valid Split ---------------')
    if args.MODEL in ('FM'):
        data = fm_data_split(args, data)
        data = fm_data_loader(args, data)

    elif args.MODEL in ('FFM'):
        data = ffm_data_split(args, data)
        data = ffm_data_loader(args, data)
    
    elif args.MODEL in ('WDN'):
        data = wdn_data_split(args, data)
        data = wdn_data_loader(args, data)

    elif args.MODEL in ('NCF'):
        data = ncf_data_split(args, data)
        data = ncf_data_loader(args, data)
    
    else:
        pass

    ######################## Model
    print(f'--------------- INIT {args.MODEL} ---------------')
    if args.MODEL=='FM':
        model = FactorizationMachineModel(args, data)
    elif args.MODEL=='FFM':
        model = FieldAwareFactorizationMachineModel(args, data)
    elif args.MODEL=='NCF':
        model = NeuralCollaborativeFiltering(args, data)
    elif args.MODEL=='WDN':
        model = WideAndDeepModel(args, data)
    else:
        pass

    ######################## TRAIN & INFERENCE & SAVE PREDICT
    print(f'--------------- {args.MODEL} TRAINING ---------------')
    model.train()
    
    if args.WANDB:
        wandb.finish()


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
