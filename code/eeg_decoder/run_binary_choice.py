import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from NVAdataset import DATASET

import physiopro.network as pnet

from utilsd import setup_experiment
from utilsd.experiment import print_config
from utilsd.config import PythonConfig, configclass, RuntimeConfig, RegistryConfig
from utilsd.logging import print_log

from pathlib import Path
import os
import itertools
from tqdm import tqdm
from dataclasses import field
from typing import Optional, Union
import numpy as np
import yaml

@configclass
class TrainConfig(PythonConfig):
    data: RegistryConfig[DATASET]
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-3

def check_dir(root):
    # Check if the folder exists
    if not os.path.exists(root):
        # Create the folder
        os.makedirs(root)
        print(f"Folder '{root}' created.")
    else:
        print(f"Folder '{root}' already exists.")

        

class MMM_EEG(nn.Module):
    def __init__(
        self, 
        input_size: Optional[int] = None,
        output_size: Optional[int] = None, 
        pretrained_weight: Optional[Union[str, Path]] = None, 
        ch_num: int = 64,
        region: int = 5,
        depth: int = 6,
        encoder_dim: int = 16
    ):
        super(MMM_EEG, self).__init__()
        self.encoder = pnet.MMM_Encoder(
            in_chans=input_size,
            channel_num=ch_num + region,
            encoder_dim=encoder_dim,
            pe_type="1d",
            depth=depth,
        )
        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(self.encoder.embed_dim  * (region + 1)),
            nn.Linear(self.encoder.embed_dim * (region + 1), output_size)
        )
        self.region = region
        self.supernodes = nn.Parameter(torch.zeros(1,self.region, input_size))
        torch.nn.init.normal_(self.supernodes)
        
        if pretrained_weight is not None:
            state_dict = torch.load(pretrained_weight)
            # remove network. in state_dict keys:
            state_dict = {k.replace("network.", ""): v for k, v in state_dict.items()}
            state_dict["pos_embed"] = self.encoder.pos_embed
            self.encoder.load_state_dict(state_dict)
    
    def forward(self, x: torch.Tensor):
        supernodes = torch.repeat_interleave(self.supernodes, x.shape[0], 0)
        x = torch.cat([x, supernodes], dim=1)
        x = self.encoder(x, region=self.region)
        return self.decoder(x)

def main(config: TrainConfig):
    #
    subj_list = ["03"]
    pe_list = ['srpe3']
    depths = {'03': 4}
    regions = {'03': 10}
    embed_sizes = {'03': 8}
    lrs = {'03': 0.001}
    wds = {'03': 1e-05}
    folds = range(5)

    for id, fold, pe in itertools.product(subj_list, folds, pe_list):

        config.data.subject_id = id
        config.data.fold = fold
        config.data.pe_type = pe
        config.lr = lrs[id]
        config.weight_decay = wds[id]
        r = regions[id]
        d = depths[id]
        e = embed_sizes[id]

        check_dir(f"{config.runtime.output_dir.parts[0]}/{config.runtime.output_dir.parts[1]}/{pe}/{id}")
        config.runtime.output_dir = Path(f"{config.runtime.output_dir.parts[0]}/{config.runtime.output_dir.parts[1]}/{pe}/{id}/{fold}")

        setup_experiment(config.runtime)
        
        print_config(config)

        train_data = config.data.build(split="train")
        train_loader = DataLoader(
            train_data, batch_size=config.batch_size, shuffle=True
        )
        
        print_log(len(train_loader), __name__)
        test_data = config.data.build(split="test")
        test_loader = DataLoader(
            test_data, batch_size=config.batch_size, shuffle=False,
        )
        print_log(len(test_loader), __name__)
        

        model = MMM_EEG(input_size=config.data.banknum, output_size=train_data.num_classes, \
                      ch_num=np.shape(test_data.data)[-3], region=r,depth=d,encoder_dim=e)
    
        net = model.cuda()

        
        loss_func = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        # TensorBoard writer
        writer = SummaryWriter(log_dir=config.runtime.output_dir) 

        num_epoch = 10
        best_test_acc = 0
        for epoch in tqdm(range(num_epoch), desc="Epochs"):
            # TRAINING phase
            total_correct = 0
            total_loss = 0
            model.train()
            for batch_idx, (data, label) in enumerate(train_loader):
                data = data.squeeze(-2).cuda()
                label = label.cuda()
                optimizer.zero_grad()
                output = net(data)
                total_correct += (output.argmax(dim=1) == label).sum().item()
                loss = loss_func(output, label.long())
                loss.backward()
                total_loss += loss
                optimizer.step()

            train_acc = total_correct/len(train_data)
            # Log train loss to TensorBoard
            writer.add_scalar('Loss/train', total_loss, epoch)
            # Log train acc to TensorBoard
            writer.add_scalar('Acc/train', train_acc, epoch)

            total_correct = 0
            total_loss = 0
            model.eval()
            with torch.no_grad():
                for batch_idx, (data, label) in enumerate(test_loader):
                    data = data.squeeze(-2).cuda()
                    label = label.cuda()
                    output = net(data)
                    total_correct += (output.argmax(dim=1) == label).sum().item()
                    total_loss += loss_func(output, label.long()).item()
                
                test_acc = total_correct / len(test_data)
                
                # Log test loss to TensorBoard
                writer.add_scalar('Loss/test', total_loss, epoch)
                # Log test acc to TensorBoard
                writer.add_scalar('Acc/test', test_acc, epoch)

                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    # Save checkpoint for the best model
                    torch.save({
                        'epoch': epoch,
                        'best_acc':best_test_acc,
                        'net_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': total_loss,
                    }, f'{config.runtime.output_dir}/checkpoint_epoch_{epoch}.pt')

                print_log(f"test acc: {test_acc}, test loss: {total_loss / len(test_data)}", __name__)
                print_log(f"best test acc: {best_test_acc}", __name__)
    

if __name__ == "__main__":
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    _config = TrainConfig.fromcli()
    main(_config)
