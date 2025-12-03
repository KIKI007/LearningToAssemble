import torch.nn.utils.spectral_norm as spectral_norm
import torch.nn as nn
import torch
from torch.nn import init
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
import time
from torch.nn import Linear, Conv1d, BatchNorm1d, Conv2d, InstanceNorm3d, AdaptiveAvgPool1d, ModuleList
import math
import numpy as np
from types import SimpleNamespace
from torch.distributions import Categorical
from learn2assemble import update_default_settings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
floatType = torch.float32
intType = torch.int32

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Conv")!=-1:
        init.kaiming_uniform_(m.weight.data)
    elif classname.find("Linear")!=-1:
        init.kaiming_uniform_(m.weight.data)
    elif classname.find("BatchNorm")!=-1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

class Upsample(nn.Module):
    def __init__(self,inchannels, outchannels,factor=2.0):
        super(Upsample,self).__init__()
        self.conv = nn.Conv2d(inchannels,outchannels,kernel_size=3,stride=1,padding=1)
        self.factor = factor

    def forward(self,x):
        x = torch.nn.functional.interpolate(x, scale_factor=self.factor, mode="bicubic",align_corners=False)
        x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self,inchannels, outchannels,factor=2):
        super(Downsample,self).__init__()
        self.conv = nn.Conv2d(inchannels,outchannels,kernel_size=4,stride=factor,padding=1)

    def forward(self,x):
        x = self.conv(x)
        return x

class ResidualLayer(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels):
        super(ResidualLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, out_channels//4,kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels//4, out_channels//4,kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(out_channels//4, out_channels,kernel_size=3, padding=1, bias=False)
        self.ac = nn.LeakyReLU()

        if self.in_channels != self.out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels,kernel_size=3, padding=1, bias=False)

    def forward(self, x) :

        h = self.conv1(x)

        h = self.ac(h)

        h = self.conv2(h)

        h = self.ac(h)

        h = self.conv3(h)

        if self.in_channels != self.out_channels:
            x = self.shortcut(x)

        return x + h

class UNet2D(nn.Module):
    def __init__(self, out_channels, ch, grid, scale=3, num_blocks=1):
        super().__init__()

        self.out_channels = out_channels
        self.ch = ch
        self.grid = grid
        self.num_blocks = num_blocks
        self.ac = nn.LeakyReLU()

        self.embedding_contact = torch.nn.Embedding(64, self.ch // 2)
        self.embedding_part_state = torch.nn.Embedding(4, self.ch // 2)

        self.down = nn.ModuleList()
        block = nn.Module()
        module = [Downsample(self.ch, self.ch)]
        module += [ResidualLayer(self.ch,self.ch) for i in range(0,num_blocks)]

        block.block = nn.ModuleList(module)
        self.down.append(block)

        # Build Encoder
        for i in range(1,scale):
            block = nn.Module()
            module = [Downsample(self.ch,2 * self.ch)]
            module += [ResidualLayer(2 * self.ch, 2 * self.ch) for i in range(0,num_blocks)]
            block.block = nn.ModuleList(module)
            self.down.append(block)
            self.ch *= 2

        self.grid = self.grid // np.power(2, scale)

        self.mid = nn.Module()

        self.mid.block_1 = ResidualLayer(self.ch,self.ch)
        self.mid.block_2 = ResidualLayer(self.ch,self.ch)
        self.v_mlp = nn.Linear(in_features=self.ch * self.grid * self.grid, out_features=1)

        # Build Decoder
        self.up = nn.ModuleList()
        self.conv_in = nn.Conv2d(self.ch, self.ch, kernel_size=3,stride=1,padding=1)

        for i in range(scale,1,-1):
            block = nn.Module()
            module = [ResidualLayer(2*self.ch,self.ch)]
            module += [ResidualLayer(self.ch,self.ch) for i in range(0,num_blocks)]
            module += [Upsample(self.ch, self.ch//2)]
            block.block = nn.ModuleList(module)
            self.up.append(block)
            self.ch //= 2

        block = nn.Module()
        module = [ResidualLayer(2*self.ch,self.ch)]
        module += [ResidualLayer(self.ch,self.ch) for i in range(0,num_blocks)]
        module += [Upsample(self.ch, self.ch)]
        block.block = nn.ModuleList(module)

        self.up.append(block)

        self.conv_out = nn.Conv2d(self.ch,self.out_channels,kernel_size=3,stride=1,padding=1)


    def forward(self, input, part_masks):
        nbatch = input.shape[0]
        grid = input.shape[-1]

        contacts = input[:, 0, :, :].reshape(nbatch, -1).type(torch.int32)
        contacts = self.embedding_contact(contacts)
        h = contacts.shape[-1]
        contacts = contacts.reshape(nbatch, grid, grid, h)
        contacts = contacts.permute([0, 3, 1, 2])

        states = input[:, 1, :, :].reshape(nbatch, -1).type(torch.int32)
        states = self.embedding_part_state(states).reshape(nbatch, grid, grid, h)
        states = states.permute([0, 3, 1, 2])

        x = torch.concatenate([contacts, states], axis = 1)

        hs = []

        for i_level in range(len(self.down)):
            x = self.down[i_level].block[0](x)
            x = self.ac(x)
            for j_level in range(1,self.num_blocks+1):
                x = self.down[i_level].block[j_level](x)
            hs.append(x)

        x = self.mid.block_1(x)
        v = torch.tanh(self.v_mlp(x.reshape(x.shape[0], -1)).squeeze(1))

        x = self.mid.block_2(x)

        x = self.conv_in(x)
        x = self.ac(x)

        for i_level in range(len(self.up)):
            features = hs.pop()
            x = self.up[i_level].block[0](torch.cat((x,features),dim=1))
            for j_level in range(1,self.num_blocks+1):
                x = self.up[i_level].block[j_level](x)
            x = self.up[i_level].block[-1](x)
            x = self.ac(x)

        x = self.conv_out(x)
        x = x.reshape(nbatch, self.out_channels, -1)
        n_part = part_masks.shape[0]
        mask = part_masks.reshape(n_part, -1)
        ac = torch.einsum('bck, pk -> bcp', x, mask).reshape(nbatch, -1)
        ac = torch.softmax(ac, dim=-1)
        return ac, v


class UNetPolicy2D(nn.Module):
    def __init__(self, settings):
        super(UNetPolicy2D, self).__init__()
        policy_config = update_default_settings(settings,
                                                'policy',
                                                {
                                                    "unet_grid": 16,
                                                    "unet_hidden_dims": 16,
                                                })


        policy_config = SimpleNamespace(**policy_config)

        self.actor = UNet2D(2, policy_config.unet_hidden_dims, policy_config.unet_grid, 3, 1)
        self.actor.apply(weights_init_kaiming)
        self.mask_prob = 1E-9

    def act(self, state, part_masks, action_mask, deterministic=False):
        action_probs, state_val = self.actor(state, part_masks)
        action_probs = action_mask * action_probs + self.mask_prob
        dist = Categorical(action_probs)
        if deterministic:
            action = torch.argmax(dist.probs, dim=-1)
        else:
            action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, part_masks, action, mask):
        action_probs, state_values = self.actor(state, part_masks)
        action_probs = mask * action_probs + self.mask_prob
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprobs, state_values, dist_entropy


if __name__ == "__main__":
    from learn2assemble.assembly import load_assembly_from_files, compute_assembly_contacts
    from learn2assemble import ASSEMBLY_RESOURCE_DIR,RESOURCE_DIR, update_default_settings, default_settings
    from learn2assemble.voxel import create_voxel_masks, get_voxel_features_2d
    import numpy as np
    import time
    import os
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    parts = load_assembly_from_files(ASSEMBLY_RESOURCE_DIR + "/tetris-1")
    part_masks, contact_masks = create_voxel_masks(parts, 0.25)

    grid = 16

    filename = os.path.join(RESOURCE_DIR, "curriculum/tetris-1.pt")
    policy_dataset = torch.load(filename)
    part_states = policy_dataset['input']

    dataset = TensorDataset(policy_dataset['input'].to('cuda'), policy_dataset['output'].to('cuda').type(torch.float32))
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

    model = UNet2D(2, 16, grid, 3, 1).to('cuda')

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("num_data", len(dataset))

    for epoch in range(1000):
        for data in dataloader:
            start = time.perf_counter()
            part_states = data[0].cpu().numpy()

            nbatch = part_states.shape[0]
            voxel_feats = get_voxel_features_2d(part_states, part_masks, contact_masks, grid)

            n_part, nx, ny, nz = part_masks.shape
            action_masks = torch.ones((nbatch, n_part * 2), device=device, dtype=bool)

            pad_part_masks = torch.zeros((n_part, grid, grid), device='cuda', dtype=floatType)
            pad_part_masks[:, :nx, :nz] = part_masks.squeeze(2)

            a, v = model(voxel_feats, pad_part_masks)

            optimizer.zero_grad()
            loss = criterion(a, data[1])
            loss.backward()
            optimizer.step()
            print(loss.item())