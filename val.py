import argparse
from trainCRNN import *


def model_val(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpuid
    config = yaml.load(open(opt.config,'r',encoding='utf-8'),Loader=yaml.FullLoader)
    config['train']['alphabet'] = loadkeyfile(config['train']['key_file'])
    config['train']['nclass'] = len(config['train']['alphabet']) + 1
    
    config['train']['local_rank'] = torch.device('cuda')
    if config['train']['algorithm']=='CRNN':
        model = crnnModel(config).cuda()
    elif config['train']['algorithm']=='SVTR':
        model = SVTRModel(config).cuda()
        
    
    model_dict = torch.load(config['infer']['model_file'])['state_dict']
    new_dict = {}
    for key in model.state_dict().keys():
        new_dict[key] = model_dict['module.'+key]
    model.load_state_dict(new_dict)
    
    criterion = CTCLoss().cuda()
    
    val_datasets = CreateDataset(config,'val')
    
    val_loaders = []
    for val_dataset in val_datasets:
        val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1,
        num_workers=2,
        collate_fn=alignCollate(config,'val'),
        shuffle=False,
        drop_last=False,
        pin_memory=True)
        val_loaders.append(val_loader)
    
    acc_dict,acc_list = Val(model,val_loaders,criterion,config)
    print(acc_dict)
    
    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', type=str, default=0, help='DDP parameter, do not modify')
    parser.add_argument('--config', type=str, default='', help='DDP parameter, do not modify')
    opt = parser.parse_args()
    model_val(opt)