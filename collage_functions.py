import torch
import torch.nn as nn
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import sys
import cv2
import clip
import torchvision

from copy import copy,deepcopy
from PIL import Image
from PIL import Image, ImageDraw, ImageFont
from matplotlib.lines import Line2D

#To install the following package from my (Jakob L. Christensen) github write:
#pip install git+https://github.com/JakobLC/jlc.git
import jlc

#To install open-ai's clip library do:
#pip install git+https://github.com/openai/CLIP.git
import clip

##############################
# PUT YOUR ROOT FOLDER BELOW #
##############################
ROOT = "C:/Users/Jakob/Desktop/DTU/Computational_Photography/Deliverables/"
sys.path.append(ROOT)

def draw_font_image(text,img_size=(224,224,3),fontsize=24,font_path="C:\\Windows\\Fonts\\courbd.ttf",letters_per_line=14):
    """ function used to generate text on images from font files

    Args:
        text (str): texts to draw.
        img_size (tuple, optional): Size of the generated image. Defaults to (224,224,3).
        fontsize (int, optional): Size of the font. Defaults to 24.
        font_path (str, optional): Path to the font file. Defaults to "C:\Windows\Fonts\courbd.ttf".
        letters_per_line (int, optional): _description_. Defaults to 14.

    Returns:
        img (np.array) : generated image
    """
    img = Image.fromarray(np.uint8(np.ones(img_size)*255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, fontsize)
    x, y = (10, 10)
    text2 = ""
    for i in range(np.ceil(len(text)/letters_per_line).astype(int)):
        text2 += text[i*letters_per_line:min((i+1)*letters_per_line,len(text))]+"\n"
    draw.text((x, y), text2, fill='black', font=font)
    return np.array(img)

class CollageTransformer(nn.Module):
    def __init__(self, 
                 allow_decoder_attention=False,
                 embedding_size=768,
                 dim_feedforward=512,
                 num_encoder_layers=6,
                 num_decoder_layers=2,
                 nhead=8,
                 output_size=1,
                 dropout=0.1,
                 use_masks = True,
                 device = None,
                 use_embedding_cnn = False,
                 pretrained_embedding_cnn = False,
                 freeze_embedding_cnn = False):
        """_summary_
        Args:
            allow_decoder_attention (bool, optional): Should candidate images be able to look at eachother. Defaults to False.
            embedding_size (int, optional): Size of the embedding. Defaults to 768 which is CLIPs size.
            dim_feedforward (int, optional): Dimension of the feedforward layer in the transformer. Defaults to 512.
            num_encoder_layers (int, optional): Number of transformer encoder layers. Defaults to 6.
            num_decoder_layers (int, optional): Number of transformer decoder layers. Defaults to 2.
            nhead (int, optional): Number of attention heads. Defaults to 8.
            output_size (int, optional): Size of the transformer output. Defaults to 1.
            dropout (float, optional): Transformer dropout. Defaults to 0.1.
            use_masks (bool, optional): Use masked attention. Defaults to True.
            device (torch.device, optional): model device. Defaults to None.
            use_embedding_cnn (bool, optional): should a CNN (ResNet50) be used to get embeddings instead of clip. Defaults to False.
            pretrained_embedding_cnn (bool, optional): should a embedding CNN (ResNet50) be pretrained. Defaults to False.
            freeze_embedding_cnn (bool, optional): should a embedding CNN (ResNet50) be frozen. Defaults to False.
        """
        super().__init__()
        self.device = "cuda:0" if device is None else device
        self.embedding_size = embedding_size
        self.allow_decoder_attention = allow_decoder_attention

        self.transformer = torch.nn.Transformer(d_model=self.embedding_size, 
                                                nhead=nhead, 
                                                num_encoder_layers=num_encoder_layers, 
                                                num_decoder_layers=num_decoder_layers, 
                                                dim_feedforward=dim_feedforward, 
                                                dropout=dropout, 
                                                batch_first=True)
        self.linear = torch.nn.Linear(in_features=embedding_size,out_features=output_size,bias=False)
        self.use_masks = use_masks

        self.use_embedding_cnn = use_embedding_cnn
        if self.use_embedding_cnn:
            preprocess = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            resnet = torchvision.models.resnet50(pretrained=pretrained_embedding_cnn)
            self.resnet = torch.nn.Sequential(preprocess,
                                              *list(resnet.children())[:-1])
            self.linear_emb = torch.nn.Linear(2048, embedding_size)
            if freeze_embedding_cnn:
                for param in self.resnet.parameters():
                    param.requires_grad = False
        
    def forward(self, col, cand, col_mask=None, cand_mask=None, batched=True):
        """ calculates candidate probabilities of belonging in the collage

        Args:
            col (torch.tensor): Collage image or embedding vectors
            cand (torch.tensor): Candidate image or embedding vectors
            col_mask (torch.tensor, optional): attention mask for the encoder. Defaults to None.
            cand_mask (torch.tensor, optional): attention mask for the decoder. Defaults to None.
            batched (bool, optional): is the input batched. Defaults to True.

        Returns:
            output: Probabilities for each candidate to belong to the collage
        """
        bsz = col.shape[0] if batched else 1
        n_col = col.shape[1]
        n_cand = cand.shape[1]

        if self.use_embedding_cnn:
            col_flat = torch.flatten(col,start_dim=0,end_dim=1)
            cand_flat = torch.flatten(cand,start_dim=0,end_dim=1)
            col = self.linear_emb(self.resnet(col_flat).squeeze()).view(bsz,n_col,self.embedding_size)
            cand = self.linear_emb(self.resnet(cand_flat).squeeze()).view(bsz,n_cand,self.embedding_size)
        eps = 1e-14
        
        if col_mask is None:
            col_padding_mask = col.std(-1)<eps
        if cand_mask is None:
            cand_padding_mask = cand.std(-1)<eps
            
        col_mask = torch.zeros(n_col,n_col,dtype=bool,device=self.device)
        colcand_mask = torch.zeros(n_cand,n_col,dtype=bool,device=self.device)
        if self.allow_decoder_attention:
            cand_mask = torch.zeros(n_cand,n_cand,dtype=bool,device=self.device)
        else:
            cand_mask = torch.eye(n_cand,device=self.device)==0
        if self.use_masks:
            x = self.transformer(src=col,
                             tgt=cand,
                             src_mask=col_mask,
                             tgt_mask=cand_mask,
                             memory_mask=colcand_mask,
                             src_key_padding_mask=col_padding_mask,
                             tgt_key_padding_mask=cand_padding_mask)
        else:
            x = self.transformer(src=col,
                             tgt=cand,
                             src_mask=col_mask,
                             tgt_mask=cand_mask,
                             memory_mask=colcand_mask)
        output = self.linear(x)
        if len(output.shape)>2:
            output = output.squeeze(2)
        return output
    
class loss_with_ignore_index():
    def __init__(self,
                 criterion = torch.nn.BCEWithLogitsLoss(),
                 ignore_index = "negatives"):
        """BCE loss and acccuracy wrapper which ignores any ground truth label less than 0 or a given value
        Args:
            criterion (optional): Which loss to use. Defaults to torch.nn.BCEWithLogitsLoss().
            ignore_index (Union[str,float,int], optional): which ground truth labels to ignore gradients on. Defaults to "negatives".
        """
        self.criterion = criterion
        self.ignore_index = ignore_index

    def acc(self,pred,label,thresh=0.5):
        """Compute classification accuracy but ignores self.ignore_index

        Args:
            pred (torch.tensor): Prediction probabilities
            label (torch.tensor): ground truth labels
            thresh (float in [0,1], optional): thresholding value. Defaults to 0.5.

        Returns:
            acc: mean accuracy
        """
        if self.ignore_index=="negatives":
            mask = label>=0
        else:
            mask = label!=self.ignore_index
        acc = ((pred[mask]>=thresh)==label[mask].bool()).float().mean().item()
        return acc

    def __call__(self,
                 pred,label):
        """computes loss

        Args:
            pred (torch.tensor): prediction probability
            label (torch.tensor): ground truth label

        Returns:
            loss: loss
        """
        if self.ignore_index=="negatives":
            mask = label>=0
        else:
            mask = label!=self.ignore_index
        return self.criterion(pred[mask],label[mask])

def train(model,loss_dict,train_loader,vali_loader,criterion,optimizer,
          n_iter=10000,
          val_every=1000,
          n_vali_batches=100,
          warmup=0.1,
          gradient_clipping=0):
    """Training function for training a CollageTransformer 

    Args:
        model (CollageTransformer): CollageTransformer network to train
        loss_dict (dict): variable to store losses in for later plotting and saving
        train_loader (torch dataloader): Test set dataloader
        vali_loader (torch dataloader): Validation set dataloader
        criterion (loss_with_ignore_index): Criterion for training
        optimizer (torch optimizer): Optimizer for training
        n_iter (int, optional): Number of training iterations. Defaults to 10000.
        val_every (int, optional): Number of training iterations between each validation. Defaults to 1000.
        n_vali_batches (int, optional): Number of iterations to compute validation over. Defaults to 100.
        warmup (float, optional): Ratio of total training iterations (n_iter) to do linear warmup over. Defaults to 0.1.
        gradient_clipping (int, optional): Uses gradient clipping if >0 with the given value as the gradient clipping. Defaults to 0.
    """
    if len(loss_dict["ite_t"])>0:
        n0 = loss_dict["ite_t"][-1]+1
    else:
        n0 = 0
    device = model.device
    lr0 = copy(optimizer.param_groups[0]['lr'])
    t = tqdm.tqdm(range(n0,n_iter+n0))
    for ite in t: 
        col, cand, labels, col_id, cand_id, info = next(iter(train_loader))
        col, cand, labels = (col.to(device),cand.to(device),labels.to(device))
        preds = model(col, cand)
        loss = criterion(preds,labels)
        acc_t = criterion.acc(preds,labels)
        optimizer.zero_grad()   
        loss.backward()

        if gradient_clipping>0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        optimizer.step()
        loss_t = loss.item()
        if ite%val_every==0:
            loss_v,acc_v = validate(model,vali_loader,criterion,n_vali_batches=n_vali_batches)
            loss_dict["loss_v"].append(loss_v)
            loss_dict["acc_v"].append(acc_v)
            loss_dict["ite_v"].append(ite)
        t.set_description(f"ite: {ite:5.0f}, acc_t: {acc_t:.4f}, acc_v: {acc_v:.4f}, loss_t: {loss_t:6.4f}, loss_v: {loss_v:6.4f}")
        loss_dict["loss_t"].append(loss_t)
        loss_dict["acc_t"].append(acc_t)
        loss_dict["ite_t"].append(ite)
        adjust_optim_lr(optimizer,ite,n_iter,warmup,lr0)

def adjust_optim_lr(optimizer,
                    ite,
                    n_iter,
                    warmup,
                    lr0):
    """adjust the learning rate of an optimizer for warmup

    Args:
        optimizer (torch optimizer): Optimizer to adjust learning rate of
        ite (int): current iteration
        n_iter (int): max number of iterations
        warmup (float): ratio of warmup compared to n_iter
        lr0 (float): Starting learning rate
    """
    ratio = ite/n_iter
    rel_ratio = ratio/warmup if warmup>0 else 1.0
    optimizer.param_groups[0]['lr'] = min(1,rel_ratio)*lr0

def validate(model,vali_loader,criterion,n_vali_batches):
    """Compute loss and accuracy over a dataset without gradients

    Args:
        model (CollageTransformer): model to use
        vali_loader (torch dataloader): Dataloader over which to compute loss and accuracy
        criterion (loss_with_ignore_index): loss function 
        n_vali_batches (int): Number of batches to do

    Returns:
        loss_v,acc_v: mean loss and accuracy
    """
    device = model.device
    loss_v = 0
    acc_v = 0
    with torch.no_grad():
        for i in range(n_vali_batches):
            col, cand, labels, col_id, cand_id, info = next(iter(vali_loader))
            col, cand, labels = (col.to(device),cand.to(device),labels.to(device))
            preds = model(col, cand)
            loss = criterion(preds, labels)
            
            acc = criterion.acc(preds, labels)
            loss_v += loss.item()
            acc_v += acc
    loss_v = loss_v/n_vali_batches
    acc_v = acc_v/n_vali_batches
    return loss_v, acc_v

def save_model(save_name,model,optimizer,loss_dict,save_root="models/"):
    """save a torch network

    Args:
        save_name (str): name of save file (with or without ".pt")
        model (torch.nn.module): network to save
        optimizer (torch optimizer): optimizer to save
        loss_dict (dict): dictionary with losses to save
        save_root (str, optional): folder in which to save the model. Defaults to "models/".
    """
    save_dict = {"model": model,
                 "optimizer": optimizer,
                 "loss_dict": loss_dict}
    if save_name[-3:]==".pt":
        torch.save(save_dict,save_root+save_name)
    else:
        torch.save(save_dict,save_root+save_name+".pt")

def load_model(save_name,save_root="models/",device=None):
    """Load a model which was saved with save_model

    Args:
        save_name (str): name of saved file to load
        save_root (str, optional): folder in which to load the model from. Defaults to "models/".
        device (torch.device, optional): device to put network on. Defaults to None.

    Returns:
        model: loaded model
        optimizer: loaded optimizer
        loss_dict: loaded loss dictionary
    """
    save_dict = torch.load(save_root+save_name)
    model = save_dict["model"]
    optimizer = save_dict["optimizer"]
    loss_dict = save_dict["loss_dict"]
    if device is not None:
        model.to(device)
    return model,optimizer,loss_dict

def plot_loss_dict(loss_dict,
                   figsize=(8,6),
                   sigma_loss_t=10,
                   sigma_acc_t=10,
                   log_loss=False,
                   ylim_loss=[None,None]):
    """Plot training graph from a loss dict

    Args:
        loss_dict (dict): loss dict to plot
        figsize (tuple, optional): figure size. Defaults to (8,6).
        sigma_loss_t (int, optional): standard deviation for the gaussian filtering for loss training graph. Defaults to 10.
        sigma_acc_t (int, optional): standard deviation for the gaussian filtering for acc training graph. Defaults to 10.
        log_loss (bool, optional): should the loss plot be in log. Defaults to False.
        ylim_loss (list, optional): y limit to use for the plot. Defaults to [None,None].
    """
    n_ite = max(loss_dict["ite_t"])
    plt.figure(figsize=figsize)
    g_loss_t = nd.gaussian_filter(loss_dict["loss_t"],sigma=sigma_loss_t)
    g_acc_t = nd.gaussian_filter(loss_dict["acc_t"],sigma=sigma_loss_t)

    plt.subplot(2,1,1)
    plt.plot(loss_dict["ite_t"],g_loss_t,label="train loss")
    plt.plot(loss_dict["ite_v"],loss_dict["loss_v"],".-",label="vali loss")
    plt.legend()
    plt.ylabel("Loss")
    plt.xlabel("Ite")
    plt.ylim(ylim_loss)
    if log_loss:
        plt.yscale("log")
    
    plt.subplot(2,1,2)
    plt.plot(loss_dict["ite_t"],g_acc_t,label="train acc")
    plt.plot(loss_dict["ite_v"],loss_dict["acc_v"],".-",label="vali acc")
    plt.legend()
    plt.ylabel("Accuracy")
    plt.xlabel("Ite")

class CollageMaker():
    def __init__(self,
                 dataset,
                 collage_sizes = [5],
                 candidate_sizes = [5],  
                 batch_size = 8,
                 positive_candidate_prob = 0.5):
        """ Class to create collages from a trained Collage model

        Args:
            dataset (CollageDataset): Dataset to use for collage creation.
            collage_sizes (list, optional): Size of collage batch, for visualizing. Samples a uniform element from the list. Defaults to [5].
            candidate_sizes (list, optional): Size of candidate batch, for visualizing. Samples a uniform element from the list.. Defaults to [5].
            batch_size (int, optional): Batch size when using the model. Defaults to 8.
            positive_candidate_prob (float, optional): Probability to sample positive examples from the dataset. Defaults to 0.5.
        """
        dataset2 = deepcopy(dataset)
        dataset2.collage_sizes = collage_sizes
        dataset2.candidate_sizes = candidate_sizes
        dataset2.positive_candidate_prob = positive_candidate_prob
        self.data_loader = torch.utils.data.DataLoader(dataset2, batch_size=batch_size,collate_fn=custom_collate_with_info)
        self.N = len(self.data_loader.dataset)
        self.batch_size = batch_size
        CLIP, preprocess = clip.load("ViT-L/14", device="cuda")  #RN50x64
        self.CLIP = CLIP
        self.preprocess = preprocess

    def get_cand_batch_ids(self,num,illegal_ids=[]):
        """Fast function to sample batch indices from

        Args:
            num (int): number of batch ids to sample
            illegal_ids (list, optional): indices which should not be sampled. Defaults to [].

        Returns:
            batch_ids: sampled batch ids
        """
        batch_ids = torch.randint(high=self.N,size=(2*num,)).tolist()
        if sum([(b not in illegal_ids) for b in batch_ids])>=num:
            batch_ids = [b for b in batch_ids if (b not in illegal_ids)]
        batch_ids = batch_ids[:num]

        return batch_ids

    def visualize_batch_predictions(self,
                                    model,
                                    return_im=False,
                                    imshow=True,
                                    reshape_size=(224,224),
                                    show_probs=True,
                                    padding=[1,1],
                                    figsize_per_pixel = 3/400):
        """Function which plots batch visualization for a given model

        Args:
            model (CollageTransformer): Model to use
            return_im (bool, optional): Should the visualization image be returned. Defaults to False.
            imshow (bool, optional): Should the visualization image be plt.imshow()'ed. Defaults to True.
            reshape_size (tuple, optional): Size to reshape images into for visualization. Defaults to (224,224).
            show_probs (bool, optional): Should predicted probabilities be overlaid on their images. Defaults to True.
            padding (list, optional): Should padding be used for the images (to seperate them visually). Defaults to [1,1].
            figsize_per_pixel (float, optional): How large a matplotlib figsize should plt.imshow() use PER PIXEL. Defaults to 3/400.

        Returns:
            montage_im: if return_im==True then it returns the visualization image.
        """
        criterion = loss_with_ignore_index(criterion=torch.nn.BCEWithLogitsLoss(reduction="none"))
        device = model.device
        model.eval()
        with torch.no_grad():
            col, cand, labels, col_id, cand_id, info = next(iter(self.data_loader))
            col, cand, labels = (col.to(device),cand.to(device),labels.to(device))
            
            preds = model(col,cand)
            probs = torch.sigmoid(preds)
            loss = criterion(preds, labels)
            
        batch_size = col.shape[0]
        n_col = col.shape[1]
        n_cand = cand.shape[1]
        
        acc_btm = [(labels[i]>=0).sum().int().item() for i in range(len(labels))]
        acc_top = [int(round(criterion.acc(preds[i],labels[i])*acc_btm[i])) for i in range(len(labels))]
        
        k = 0
        montage_im_list = []
        n_col_batch = []
        n_cand_batch = []
        for i in range(batch_size):
            col_clip_matrix_idx = col_id[i][col_id[i]>=0].flatten().tolist()
            cand_clip_matrix_idx = cand_id[i][cand_id[i]>=0].flatten().tolist()

            col_images = self.data_loader.dataset.get_images(col_clip_matrix_idx)
            cand_images = self.data_loader.dataset.get_images(cand_clip_matrix_idx)
            
            n_col_batch.append(len(col_images))
            n_cand_batch.append(len(cand_images))
            black_image = [0*col_images[0]]
            montage_im_list.append(col_images+black_image+cand_images)

        montage_im = jlc.functions.montage(montage_im_list,reshape_size=reshape_size,padding=padding,imshow=False,return_im=True)
        if reshape_size is None:
            reshape_size = montage_im_list[0][0].shape
        if imshow:
            plt.figure(figsize=(figsize_per_pixel*montage_im.shape[1],figsize_per_pixel*montage_im.shape[0]))
            plt.imshow(montage_im,cmap="gray")
            if show_probs:
                s0 = reshape_size[0]+padding[0]*2
                s1 = reshape_size[1]+padding[1]*2
                for i in range(batch_size):
                    j2 = n_col_batch[i]
                    s = f'acc={acc_top[i]}/{acc_btm[i]}\nloss={loss[i].mean():.3f}'
                    plt.text(s1*j2+reshape_size[0]*0.03,s0*i+reshape_size[0]*0.03,s,
                             color=[1,1,1],
                             fontweight="bold",
                             ha="left",
                             va="top")
                    for j in range(n_cand_batch[i]):
                        c = [0,1,0] if int(round(labels[i,j].item())) else [1,0,0]
                        j2 = j+n_col_batch[i]+1
                        s = f'p={probs[i,j]*100:.0f}%'
                        plt.text(s1*j2+reshape_size[0]*0.03,s0*i+reshape_size[0]*0.03,s,
                                 color=c,
                                 fontweight="bold",
                                 ha="left",
                                 va="top")

        if return_im:
            return montage_im
              
    def grow_collage(self,
                     model,
                     clip_text=None,
                     clip_image=None,
                     seed_clip_idx=None,
                     seed_group_type="random",
                     num_seeds=1,
                     selection_criteria="best",
                     p_threshold=0.9,
                     max_candidates_per_ite=64,
                     grow_per_ite=1,
                     stop_size=16,
                     imshow=True,
                     padding=[1,1],
                     return_im=False,
                     figsize_per_pixel = 3/400,
                     images_per_row = 8,
                     reshape_size = (224,224),
                     show_text=False,
                     fancy_collage=False,
                     return_col_ids=False,
                     n_fancy_rows=None,
                     fancy_in_random_order=False):
        """Collage growing function

        Args:
            model (CollageTransformer): Model to use when calcuting probabilities for growing the collage
            clip_text (Union[list,nonetype,str], optional): Text which is embedding with clip and used as seeds. Defaults to None.
            clip_image (Union[list,nonetype,torch.tensor], optional): Images which are embedded with clip and used as seeds. Defaults to None.
            seed_clip_idx (Union[list,nonetype,int], optional): Dataset indices to use as seeds. Defaults to None.
            seed_group_type (Union[int,"random"], optional): Which group type (from the dataset) should be sampled from (e.g. fonts, letters, prompts). Defaults to "random".
            num_seeds (int, optional): Number of seeds to use (for random seeds, has to align with the given input seeds). Defaults to 1.
            selection_criteria (["best","threshold"], optional): Select either the best seeds per iteration, or all seeds better than some threshold. Defaults to "best".
            p_threshold (float, optional): Threshold to use if selection_criteria=="threshold". Defaults to 0.9.
            max_candidates_per_ite (int, optional): How many candidates should be shown as to the model per growing iteration. Defaults to 64.
            grow_per_ite (int, optional): How many images should be accepted as new collage images per growing iteration. Defaults to 1.
            stop_size (int, optional): What size should the collage be stopped at. Defaults to 16.
            imshow (bool, optional): Should the collage be shown. Defaults to True.
            padding (list, optional): Should images have padding for the collage image. Defaults to [1,1].
            return_im (bool, optional): Should the collage image be returned. Defaults to False.
            figsize_per_pixel (float, optional): How large a matplotlib figsize should plt.imshow() use PER PIXEL. Defaults to 3/400.
            images_per_row (int, optional): How many images so be in each row for the collage image. Defaults to 8.
            reshape_size (tuple, optional): Size to reshape images into for visualization. Defaults to (224,224).
            show_text (bool, optional): Should text with info about images be overload on them. Defaults to False.
            fancy_collage (bool, optional): Should "fancy" collage image be generated (which renders most image arguments irrelevant, but plots an exact fit image). Defaults to False.
            return_col_ids (bool, optional): Return dataset indices from the generated collage. Defaults to False.
            n_fancy_rows (_type_, optional): Number of rows to place images in if fancy_collage==True. Defaults to None.
            fancy_in_random_order (bool, optional): should images be shuffled randomly in the visualization for fancy collages. Defaults to False.

        Returns:
            collage_im,collage_ids: collage image and collage indices (but only if they have been flagged to be returned in the input arguments)
        """
        device = model.device
        assert selection_criteria in ["best","threshold"]
        if isinstance(seed_group_type,int):
            assert seed_group_type<len(self.data_loader.dataset.collage_types)
            seed_group_type_idx = seed_group_type
        elif isinstance(seed_group_type,str):
            assert seed_group_type in self.data_loader.dataset.collage_types+["random"]
            if seed_group_type=="random":
                seed_group_type_idx = -1
            else:
                seed_group_type_idx = self.data_loader.dataset.collage_types.index(seed_group_type)
        

        if seed_clip_idx is not None:
            assert len(seed_clip_idx)==num_seeds
        else:
            if seed_group_type_idx<0:
                seed_clip_idx = torch.randint(high=self.N,size=(num_seeds,)).tolist()
            else:
                group_idx = self.data_loader.dataset.sample_group_idx(seed_group_type_idx)
                seed_clip_idx = np.random.choice(self.data_loader.dataset.groups[group_idx],size=num_seeds).list()
        use_clip = (clip_text is not None) or (clip_image is not None)

        len_extra = 0
        if self.data_loader.dataset.CNN:
            col_extra = torch.zeros(1,0,self.data_loader.dataset.cnn_reshape[2],
                                        self.data_loader.dataset.cnn_reshape[0],
                                        self.data_loader.dataset.cnn_reshape[1]).to(device)
        else:
            col_extra = torch.zeros(1,0,self.data_loader.dataset.clip_dim).to(device)

        if use_clip:
            images_extra = []
            with torch.no_grad():
                if (clip_text is not None):
                    assert self.data_loader.dataset.CNN==False
                    if not isinstance(clip_text,list):
                        clip_text = [clip_text]
                    len_extra += len(clip_text)
                    text = clip.tokenize(clip_text).to(device)
                    text_features = self.CLIP.encode_text(text).unsqueeze(0)
                    col_extra = torch.cat((col_extra,text_features),dim=1)
                    text_image = [draw_font_image(text=t) for t in clip_text] 
                    images_extra.extend(text_image)
                if (clip_image is not None):
                    if not isinstance(clip_image,list):
                            clip_image = [clip_image]
                            clip_image_extra = [np.asarray(c) for c in clip_image]
                            images_extra.extend(clip_image_extra)
                    if self.data_loader.dataset.CNN:
                        col_extra = (self.data_loader.dataset.get_images(image_idx=None,
                                            reshape_size=self.data_loader.dataset.cnn_reshape,
                                            return_torch=True,
                                            input_for_reshape=clip_image)/255).to(device).unsqueeze(0)
                    else:
                        len_extra += len(clip_image)
                        images = torch.stack([self.preprocess(im) for im in clip_image],dim=0)
                        images = images.to(device)
                        images_np = images.permute((0,2,3,1)).detach().cpu().numpy()
                        images_np = [images_np[im_i] for im_i in range(len(images_np))]
                        image_features = self.CLIP.encode_image(images).unsqueeze(0)
                        col_extra = torch.cat((col_extra,image_features),dim=1)

            num_seeds = len_extra
            col_ids = []
            if self.data_loader.dataset.normalize and not self.data_loader.dataset.CNN:
                col_extra = torch.nn.functional.normalize(col_extra,dim=2)
        else:
            images_extra = []
            col_ids = seed_clip_idx
        col_ite = [0 for _ in range(len(col_ids))]
        col_p_vals = [0 for _ in range(len(col_ids))]
        n_batch = np.ceil(max_candidates_per_ite/self.batch_size).astype(int)

        num_ites = np.ceil((stop_size-num_seeds)/grow_per_ite).astype(int)

        model.eval()
        with torch.no_grad():
            for ite in tqdm.tqdm(range(1,num_ites+1)):
                if self.data_loader.dataset.CNN:
                    col = (self.data_loader.dataset.get_images(image_idx=col_ids,
                                            reshape_size=self.data_loader.dataset.cnn_reshape,
                                            return_torch=True,
                                            input_for_reshape=None)/255).to(device).unsqueeze(0)
                else:
                    col = self.data_loader.dataset.clip_matrix[col_ids].to(device).unsqueeze(0).type(torch.float32)
                col = torch.cat((col_extra,col),axis=1)
                cand_ids = []
                p_vals = []
                stop_criteria = False
                for batch_i in range(n_batch):
                    batch_size2 = min(self.batch_size,max_candidates_per_ite-len(cand_ids))
                    assert batch_size2>0
                    batch_ids = self.get_cand_batch_ids(num=batch_size2,illegal_ids=col_ids)
                    if self.data_loader.dataset.CNN:
                        cand = (self.data_loader.dataset.get_images(image_idx=batch_ids,
                                            reshape_size=self.data_loader.dataset.cnn_reshape,
                                            return_torch=True,
                                            input_for_reshape=None)/255).to(device).unsqueeze(0)
                    else:
                        cand = self.data_loader.dataset.clip_matrix[batch_ids].to(device).unsqueeze(0).type(torch.float32)
                        
                    preds = model(col,cand)
                    probs = torch.sigmoid(preds)

                    cand_ids.extend(batch_ids)
                    p_vals.extend(probs.flatten().tolist())

                    if selection_criteria=="best":
                        pass
                    elif selection_criteria=="threshold":
                        stop_criteria = sum([p>=p_threshold for p in p_vals])>=grow_per_ite

                    if stop_criteria:
                        break
                order = np.argsort(p_vals)
                n_add = min(grow_per_ite,stop_size-len(col_ids)-len_extra)

                add_idx = order[-n_add:].tolist()

                p_vals_add = [p_vals[i] for i in add_idx]

                col_p_vals.extend(p_vals_add)
                col_ids.extend([cand_ids[i] for i in add_idx])
                col_ite.extend([ite for _ in range(len(add_idx))])

                if selection_criteria=="threshold":
                    if min(p_vals_add)<p_threshold:
                        print(f'warning: did not find {grow_per_ite} p-vals >={p_threshold:.4f} in\
                        the {max_candidates_per_ite} iteration samples')
                
                if len(col_ids)+len_extra>=stop_size:
                    break
        images = self.data_loader.dataset.get_images(col_ids)
        images = images_extra+images
        n_col = min(images_per_row,stop_size)
        n_row = np.ceil(stop_size/images_per_row).astype(int)
        if fancy_collage:
            n_align = np.ceil(stop_size**0.24).astype(int) if n_fancy_rows is None else n_fancy_rows
            montage_im = jlc.collage(images,n_alignment=n_align,imshow=False,return_image=True,
                                     random_order=fancy_in_random_order)
        else:
            montage_im = jlc.functions.montage(images,padding=padding,
                                           imshow=False,
                                           n_col=n_col,
                                           n_row=n_row,
                                           reshape_size=reshape_size,
                                           return_im=True)
        if imshow:
            plt.figure(figsize=(figsize_per_pixel*montage_im.shape[1],figsize_per_pixel*montage_im.shape[0]))
            plt.imshow(montage_im,cmap="gray")
            

            if show_text and not fancy_collage:
                if reshape_size is None:
                    reshape_size = images[0].shape
                s0 = reshape_size[0]+padding[0]*2
                s1 = reshape_size[1]+padding[1]*2
                row_i = 0
                col_i = 0
                for i in range(stop_size):
                    if i>=len_extra:
                        i_hat = i-len_extra
                        s = f'id={col_ids[i_hat]}\nite={col_ite[i_hat]}\np={col_p_vals[i_hat]*100:.0f}%'
                        plt.text(s1*row_i+reshape_size[1]*0.03,s0*col_i+reshape_size[0]*0.03,s,
                                color=[0,0,1],
                                fontsize=7,
                                fontweight="bold",
                                ha="left",
                                va="top")
                    row_i += 1
                    if row_i>=images_per_row:
                        row_i = 0
                        col_i += 1
            plt.show()
        if return_col_ids:
            if return_im:
                return montage_im,col_ids
            else:
                return col_ids
        else:
            if return_im:
                return montage_im

class CollageDataset(torch.utils.data.Dataset):
    def __init__(self,
                 collage_sizes = list(range(1,8)),
                 candidate_sizes = [8],
                 positive_candidate_prob = 0.5,
                 collage_type_prob = [0.25,0.25,0.5,0],
                 normalize = True,
                 padding_idx = -1,
                 clip_dim = 768,
                 small_group_strategy="concat",
                 min_group_length = None,
                 filter_positives = True,
                 split_start_and_stop = [0,1],
                 allow_dataset_mixing = False,
                 CNN = False,
                 cnn_reshape = [224,224,3]):
        """Dataset for training a CollageTransformer

        Args:
            collage_sizes (list, optional): array to sample the collage size from uniformly. Defaults to list(range(1,8)).
            candidate_sizes (list, optional): array to sample the candidate size from uniformly. Defaults to [8].
            positive_candidate_prob (float, optional): Probability of sampling positive candidates for the collage. Defaults to 0.5.
            collage_type_prob (list, optional): length 4 array or list with probabilities for different collage types in the order ["font","letter","same_prompt","1k"] for datasets [Dafont-free,Dafont-free,Simulacra,LAION-Aesthetics6.5+]. Defaults to [0.25,0.25,0.5,0].
            normalize (bool, optional): Should the clip vectors be L2-normalized. Defaults to True.
            padding_idx (int, optional): padding index used for the torch.tensor entries which are empty since the collage was smaller than the maximum size. Defaults to -1.
            clip_dim (int, optional): Dimension of clip vectors. Defaults to 768.
            small_group_strategy (["concat","ignore"], optional): How to deal with small groups (collages). "concat" combines small groups with other groups. "ignore" only uses single groups - however they can be very small which usually ends in the same images being sampled many times. Defaults to "concat".
            min_group_length (int, optional): Minimum size which groups should have to be included in the dataset. Defaults to None.
            filter_positives (bool, optional): Should positives which are already in the collage not be included in candidates. Defaults to True.
            split_start_and_stop (list, optional): dataset starting and stopping point in terms of ratio of indices to use for dataloader. Defaults to [0,1].
            allow_dataset_mixing (bool, optional): should the model show candidates from different datasets than the sampled used for the collage. Defaults to False.
            CNN (bool, optional): Is a CNN being used (instead of CLIP) to embed images to vectors. Defaults to False.
            cnn_reshape (list, optional): Reshape size for inputs of CNN if a CNN is used for embedding vectors. Defaults to [224,224,3].
        """
        assert small_group_strategy in ["concat","ignore"]
        self.CNN = CNN
        self.allow_dataset_mixing = allow_dataset_mixing
        self.split_start_and_stop = split_start_and_stop
        self.small_group_strategy = small_group_strategy
        self.filter_positives = filter_positives
        self.clip_dim = clip_dim
        self.positive_candidate_prob = positive_candidate_prob
        self.normalize = normalize
        self.padding_idx = padding_idx
        self.cnn_reshape = cnn_reshape

        self.dataset_types = [0,0,1,2]
        self.collage_types = ["font","letter","same_prompt","1k"]

        self.dataset_lengths = [0,0,0]
        self.groups = []
        self.group_types = []
        self.group_lengths = []
        self.image_names = []
        self.collage_type_prob = np.array(collage_type_prob)
        self.collage_type_prob = self.collage_type_prob/(self.collage_type_prob.sum())
        self.collage_sizes = collage_sizes
        max_collage_size = max(collage_sizes)
        self.candidate_sizes = candidate_sizes
        max_candidate_size = max(candidate_sizes)
        self.min_group_length = min_group_length
        if self.min_group_length is None:
            self.min_group_length = max_collage_size//2
            
        self.min_group_size = int(max_collage_size+self.positive_candidate_prob*max_candidate_size)
        self.use_dataset = [False,False,False]
        self.use_groups = []
        self.group_names = []
        self.group_type_lengths = []
        self.clip_matrix = torch.zeros(0,clip_dim)
        self.num_groups = 0
        self.translation = 0
        for i in range(len(self.collage_types)):
            if self.collage_type_prob[i]>0:
                ii = self.dataset_types[i]
                if not self.use_dataset[ii]:
                    self.translation = sum(self.dataset_lengths)
                    self.use_dataset[ii] = True
                    clip_matrix, names = self.get_clip_and_names(self.dataset_types[i]) 
                    self.image_names.extend(names)
                    self.dataset_lengths[ii] = len(names)
                    self.clip_matrix = torch.cat((self.clip_matrix,clip_matrix),axis=0)
                groups, group_names = self.get_groups(names,i,translation=self.translation)
                self.group_names.extend(group_names)
                self.num_groups += len(groups)
                self.groups.extend(groups)
                self.group_types.extend([i for _ in range(len(groups))])
                self.group_lengths.extend([len(g) for g in groups])
                self.group_type_lengths.append(len(groups))
            else:
                self.group_type_lengths.append(0)

        self.group_lengths = np.array(self.group_lengths)
        self.group_types = np.array(self.group_types)

        if normalize:
            self.clip_matrix = torch.nn.functional.normalize(self.clip_matrix,dim=1)

    def get_clip_and_names(self,dataset_idx):
        """loads CLIP matrix and names which define groups in datasets

        Args:
            dataset_idx (int): index of the dataset

        Returns:
            clip_matrix,names: clip matrix and names of images as a list in the same order as the clip matrix
        """
        if dataset_idx==0:
            loaded = torch.load(ROOT+"test_set_data_compressed/CLIP_fonts.pth")
        elif dataset_idx==1:
            loaded = torch.load(ROOT+"test_set_data_compressed/CLIP_sim.pth")
        elif dataset_idx==2:
            loaded = torch.load(ROOT+"test_set_data_compressed/CLIP_laion.pth")
        n = len(loaded["CLIP_matrix"])
        start = max(0,np.floor(self.split_start_and_stop[0]*n).astype(int))
        stop = min(n,np.ceil(self.split_start_and_stop[1]*n).astype(int))
            
        clip_matrix = loaded["CLIP_matrix"][start:stop]
        names = loaded["image_names"][start:stop]
        return clip_matrix, names

    def get_groups(self,names,group_idx,translation):
        """returns a set of groups (collages) from a given list of names

        Args:
            names (list): list of image names
            group_idx (int): group index
            translation (int): integer translation in terms if image indices, to make sure they are unique

        Returns:
            groups,group_names: list of groups and the name of each associated group
        """
        groups = []
        if group_idx==0: #font
            group_names = np.array([n.split('\\')[0] for n in names])
        elif group_idx==1: #letter
            group_names = np.array([n.split('\\')[1] for n in names])
        elif group_idx==2: #same_prompt
            group_names = np.array([n.split('_')[0] for n in names])
        elif group_idx==3:
            group_names = np.tile(np.arange(10),(1,np.ceil(len(names)/10).astype(int)))[:len(names)].flatten()
        uq,uq_inverse = np.unique(group_names,return_inverse=True)
        groups = [[] for _ in range(len(uq))]
        for i_sample,i_group in enumerate(uq_inverse):
            groups[i_group].append(i_sample+translation)
        
        group_names = [group_names[j] for j in range(len(groups)) if len(groups[j])>=self.min_group_length]
        groups = [g for g in groups if len(g)>=self.min_group_length]
        return groups,group_names

    def get_images(self,image_idx,reshape_size=None,return_torch=False,input_for_reshape=None):
        """return images from a list of image indices

        Args:
            image_idx (Union[int,list]): list of indices to get images for
            reshape_size (list, optional): list,tuple or array to reshape images into. Defaults to None.
            return_torch (bool, optional): should a torch tensor be returned. Defaults to False.
            input_for_reshape (list, optional): list of images to reshape instead of loading images from image_idx. Defaults to None.

        Returns:
            images: images
        """
        if input_for_reshape is None:
            if isinstance(image_idx,int):
                image_idx = [image_idx]
            images = []
            for idx in image_idx:
                if idx<0:
                    continue
                elif 0<=idx and idx<self.dataset_lengths[0]:
                    img_path = ROOT+"test_set_data_compressed/Dafonts/"+self.image_names[idx]+".png"
                elif sum(self.dataset_lengths[:1])<=idx and idx<sum(self.dataset_lengths[:2]):
                    img_path = ROOT+"test_set_data_compressed/Simulacra/"+self.image_names[idx]+".jpg"
                elif sum(self.dataset_lengths[:2])<=idx and idx<sum(self.dataset_lengths[:3]):
                    im_name = self.image_names[idx][self.image_names[idx].find("\\")+2:]
                    img_path = ROOT+"test_set_data_compressed/LAION_aesthetics_6dot5/"+im_name+".jpg"
                pil_image = Image.open(img_path)
                images.append(np.array(pil_image))
        else:
            images = input_for_reshape

        if reshape_size is not None:
            if isinstance(reshape_size,tuple):
                reshape_size = list(reshape_size)
            if len(reshape_size)<3:
                reshape_size += [3]
            images = [cv2.resize(im, reshape_size[:2], interpolation=cv2.INTER_LINEAR) for im in images]
            for i in range(len(images)):                
                im_size = images[i].shape
                if len(im_size)==2:
                    images[i] = images[i][:,:,None]
                    im_size = images[i].shape
                assert reshape_size[2] in [1,3]
                if reshape_size[2]==3:
                    if im_size[2]==1:
                        images[i]=np.tile(images[i],(1,1,3))
                    elif im_size[2]==4:
                        images[i] = images[i][:3]
                    else:
                        assert im_size[2]==3
                elif reshape_size[2]==1:
                    images[i] = images[i].mean(2,keepdims=True)
        if return_torch and (reshape_size is not None):
            images = torch.stack([torch.from_numpy(im) for im in images],axis=0).permute((0,3,1,2))
        elif return_torch and (reshape_size is None):
            images = [torch.from_numpy(im) for im in images]
        return images
    
    def sample_group_idx(self,group_type):
        """Fast sampling function for getting a group idx

        Args:
            group_type (int): index of group type to sample from

        Returns:
            idx: sampled group idx
        """
        if group_type==0:
            idx = np.random.choice(self.group_type_lengths[0])
        elif group_type>0:
            idx = np.random.choice(self.group_type_lengths[group_type])+sum(self.group_type_lengths[:group_type])
        return idx
    
    def sample_negative_idx(self,group_type,group_indices):
        """Sample a negative index from a group type, but without sampling from indices in group_indices

        Args:
            group_type (int): group type index from which to sample from
            group_indices (list): list of group indices which constitute the positive groups, and therefore not sampled

        Returns:
            idx: sampled negative index
        """
        illegal_indices = sum([self.groups[id] for id in group_indices],[])
        for _ in range(5):
            if self.allow_dataset_mixing:
                idx = np.random.choice(sum(self.dataset_lengths))
            elif self.dataset_types[group_type]==0:
                idx = np.random.choice(self.dataset_lengths[0])
            else:# self.dataset_types[group_type]==1 or 2:
                k = self.dataset_types[group_type]
                idx = np.random.choice(self.dataset_lengths[k])+sum(self.dataset_lengths[:k])
            if idx not in illegal_indices:
                break
        return idx
    
    def __len__(self):
        return sum(self.dataset_lengths)

    def __getitem__(self, idx):
        max_collage_size = max(self.collage_sizes)
        max_candidate_size = max(self.candidate_sizes)

        n_collage = np.random.choice(self.collage_sizes)
        n_candidates = np.random.choice(self.candidate_sizes)

        info = {}
        group_type = np.random.choice(len(self.collage_types),p=self.collage_type_prob)

        info["group_type"] = [group_type,self.collage_types[group_type]]

        if self.small_group_strategy=="concat":
            positives = []
            group_indices = []
            for _ in range(max_collage_size+max_candidate_size):
                group_idx = self.sample_group_idx(group_type)
                group_indices.append(group_idx)
                positives.extend(self.groups[group_idx])

                if len(positives)>self.min_group_size:
                    break
        elif self.small_group_strategy=="ignore":
            group_idx = self.sample_group_idx(group_type)
            positives = self.groups[group_idx]
            group_indices = [group_idx]
        
        collage_idx = self.padding_idx*torch.ones(max_collage_size,dtype=int)
        collage_idx[:n_collage] = torch.from_numpy(np.random.choice(positives, size=n_collage)).long()

        if len(group_indices)>1:
            group_indices_actually_used = []
            for g_idx in group_indices:
                if any([(g_i in collage_idx[:n_collage]) for g_i in self.groups[g_idx]]):
                    group_indices_actually_used.append(g_idx)
            positives = sum([self.groups[id] for id in group_indices_actually_used],[])
            group_indices = group_indices_actually_used
        
        info["group_indices"] = group_indices
        info["group_names"] = [self.group_names[g_idx] for g_idx in group_indices]



        if self.filter_positives:
            positives_tmp = [n for n in positives if n not in collage_idx]
            if len(positives_tmp)>0:
                positives = positives_tmp

        label = self.padding_idx*torch.ones(max_candidate_size,dtype=torch.float32)
        candidate_idx = self.padding_idx*torch.ones(max_candidate_size,dtype=int)
        for i in range(n_candidates):
            if np.random.rand()<self.positive_candidate_prob: 
                candidate_idx[i] = np.random.choice(positives)
                label[i] = 1
            else:
                candidate_idx[i] = self.sample_negative_idx(group_type,group_indices)
                label[i] = 0
        
        if self.CNN:
            candidate = self.padding_idx*torch.ones(max_candidate_size,self.cnn_reshape[2],self.cnn_reshape[0],self.cnn_reshape[1],dtype=torch.float32)
            collage = self.padding_idx*torch.ones(max_collage_size,self.cnn_reshape[2],self.cnn_reshape[0],self.cnn_reshape[1],dtype=torch.float32)
            candidate[:n_candidates] = self.get_images(candidate_idx,reshape_size=self.cnn_reshape,return_torch=True)/255
            collage[:n_collage] = self.get_images(collage_idx,reshape_size=self.cnn_reshape,return_torch=True)/255
        else:
            candidate = self.padding_idx*torch.ones(max_candidate_size,self.clip_dim,dtype=torch.float32)
            collage = self.padding_idx*torch.ones(max_collage_size,self.clip_dim,dtype=torch.float32)
            candidate[:n_candidates] = self.clip_matrix[candidate_idx[:n_candidates]]
            collage[:n_collage] = self.clip_matrix[collage_idx[:n_collage]]

        return collage, candidate, label, collage_idx, candidate_idx, info

def custom_collate_with_info(original_batch):
    n = len(original_batch[0])
    bs = len(original_batch)
    normal_batch = []
    for i in range(n):
        list_of_items = [item[i] for item in original_batch]
        if i+1==n:
            info = list_of_items
        else:
            normal_batch.append(torch.stack(list_of_items,axis=0))

    return *normal_batch,info

class CollageDataset2(torch.utils.data.Dataset):
    def __init__(self,
                 collage_sizes = list(range(1,8)),
                 candidate_sizes = [8],
                 positive_candidate_prob = 0.5,
                 collage_type_prob = [0.25,0.25,0.5,0],
                 normalize = True,
                 padding_idx = -1,
                 clip_dim = 768,
                 small_group_strategy="concat",
                 min_group_length = None,
                 filter_positives = True,
                 split_start_and_stop = [0,1],
                 allow_dataset_mixing = False,
                 CNN = False,
                 cnn_reshape = [224,224,3]):
        """Dataset for training a CollageTransformer

        Args:
            collage_sizes (list, optional): array to sample the collage size from uniformly. Defaults to list(range(1,8)).
            candidate_sizes (list, optional): array to sample the candidate size from uniformly. Defaults to [8].
            positive_candidate_prob (float, optional): Probability of sampling positive candidates for the collage. Defaults to 0.5.
            collage_type_prob (list, optional): length 4 array or list with probabilities for different collage types in the order ["font","letter","same_prompt","1k"] for datasets [Dafont-free,Dafont-free,Simulacra,LAION-Aesthetics6.5+]. Defaults to [0.25,0.25,0.5,0].
            normalize (bool, optional): Should the clip vectors be L2-normalized. Defaults to True.
            padding_idx (int, optional): padding index used for the torch.tensor entries which are empty since the collage was smaller than the maximum size. Defaults to -1.
            clip_dim (int, optional): Dimension of clip vectors. Defaults to 768.
            small_group_strategy (["concat","ignore"], optional): How to deal with small groups (collages). "concat" combines small groups with other groups. "ignore" only uses single groups - however they can be very small which usually ends in the same images being sampled many times. Defaults to "concat".
            min_group_length (int, optional): Minimum size which groups should have to be included in the dataset. Defaults to None.
            filter_positives (bool, optional): Should positives which are already in the collage not be included in candidates. Defaults to True.
            split_start_and_stop (list, optional): dataset starting and stopping point in terms of ratio of indices to use for dataloader. Defaults to [0,1].
            allow_dataset_mixing (bool, optional): should the model show candidates from different datasets than the sampled used for the collage. Defaults to False.
            CNN (bool, optional): Is a CNN being used (instead of CLIP) to embed images to vectors. Defaults to False.
            cnn_reshape (list, optional): Reshape size for inputs of CNN if a CNN is used for embedding vectors. Defaults to [224,224,3].
        """
        assert small_group_strategy in ["concat","ignore"]
        self.CNN = CNN
        self.allow_dataset_mixing = allow_dataset_mixing
        self.split_start_and_stop = split_start_and_stop
        self.small_group_strategy = small_group_strategy
        self.filter_positives = filter_positives
        self.clip_dim = clip_dim
        self.positive_candidate_prob = positive_candidate_prob
        self.normalize = normalize
        self.padding_idx = padding_idx
        self.cnn_reshape = cnn_reshape

        self.dataset_types = [0,0,1,2]
        self.collage_types = ["font","letter","same_prompt","1k"]

        self.dataset_lengths = [0,0,0]
        self.groups = []
        self.group_types = []
        self.group_lengths = []
        self.image_names = []
        self.collage_type_prob = np.array(collage_type_prob)
        self.collage_type_prob = self.collage_type_prob/(self.collage_type_prob.sum())
        self.collage_sizes = collage_sizes
        max_collage_size = max(collage_sizes)
        self.candidate_sizes = candidate_sizes
        max_candidate_size = max(candidate_sizes)
        self.min_group_length = min_group_length
        if self.min_group_length is None:
            self.min_group_length = max_collage_size//2
            
        self.min_group_size = int(max_collage_size+self.positive_candidate_prob*max_candidate_size)
        self.use_dataset = [False,False,False]
        self.use_groups = []
        self.group_names = []
        self.group_type_lengths = []
        self.clip_matrix = torch.zeros(0,clip_dim)
        self.num_groups = 0
        self.translation = 0
        for i in range(len(self.collage_types)):
            if self.collage_type_prob[i]>0:
                ii = self.dataset_types[i]
                if not self.use_dataset[ii]:
                    self.translation = sum(self.dataset_lengths)
                    self.use_dataset[ii] = True
                    clip_matrix, names = self.get_clip_and_names(self.dataset_types[i]) 
                    self.image_names.extend(names)
                    self.dataset_lengths[ii] = len(names)
                    self.clip_matrix = torch.cat((self.clip_matrix,clip_matrix),axis=0)
                groups, group_names = self.get_groups(names,i,translation=self.translation)
                self.group_names.extend(group_names)
                self.num_groups += len(groups)
                self.groups.extend(groups)
                self.group_types.extend([i for _ in range(len(groups))])
                self.group_lengths.extend([len(g) for g in groups])
                self.group_type_lengths.append(len(groups))
            else:
                self.group_type_lengths.append(0)

        self.group_lengths = np.array(self.group_lengths)
        self.group_types = np.array(self.group_types)

        if normalize:
            self.clip_matrix = torch.nn.functional.normalize(self.clip_matrix,dim=1)

    def get_clip_and_names(self,dataset_idx):
        """loads CLIP matrix and names which define groups in datasets

        Args:
            dataset_idx (int): index of the dataset

        Returns:
            clip_matrix,names: clip matrix and names of images as a list in the same order as the clip matrix
        """
        if dataset_idx==0:
            loaded = torch.load(ROOT+"test_set_data_compressed/CLIP_fonts.pth")
        elif dataset_idx==1:
            loaded = torch.load(ROOT+"test_set_data_compressed/CLIP_sim.pth")
        elif dataset_idx==2:
            loaded = torch.load(ROOT+"test_set_data_compressed/CLIP_laion.pth")
        n = len(loaded["CLIP_matrix"])
        start = max(0,np.floor(self.split_start_and_stop[0]*n).astype(int))
        stop = min(n,np.ceil(self.split_start_and_stop[1]*n).astype(int))
            
        clip_matrix = loaded["CLIP_matrix"][start:stop]
        names = loaded["image_names"][start:stop]
        return clip_matrix, names

    def get_groups(self,names,group_idx,translation):
        """returns a set of groups (collages) from a given list of names

        Args:
            names (list): list of image names
            group_idx (int): group index
            translation (int): integer translation in terms if image indices, to make sure they are unique

        Returns:
            groups,group_names: list of groups and the name of each associated group
        """
        groups = []
        if group_idx==0: #font
            group_names = np.array([n.split('\\')[0] for n in names])
        elif group_idx==1: #letter
            group_names = np.array([n.split('\\')[1] for n in names])
        elif group_idx==2: #same_prompt
            group_names = np.array([n.split('_')[0] for n in names])
        elif group_idx==3:
            group_names = np.tile(np.arange(10),(1,np.ceil(len(names)/10).astype(int)))[:len(names)].flatten()
        uq,uq_inverse = np.unique(group_names,return_inverse=True)
        groups = [[] for _ in range(len(uq))]
        for i_sample,i_group in enumerate(uq_inverse):
            groups[i_group].append(i_sample+translation)
        
        group_names = [group_names[j] for j in range(len(groups)) if len(groups[j])>=self.min_group_length]
        groups = [g for g in groups if len(g)>=self.min_group_length]
        return groups,group_names

    def get_images(self,image_idx,reshape_size=None,return_torch=False,input_for_reshape=None):
        """return images from a list of image indices

        Args:
            image_idx (Union[int,list]): list of indices to get images for
            reshape_size (list, optional): list,tuple or array to reshape images into. Defaults to None.
            return_torch (bool, optional): should a torch tensor be returned. Defaults to False.
            input_for_reshape (list, optional): list of images to reshape instead of loading images from image_idx. Defaults to None.

        Returns:
            images: images
        """
        if input_for_reshape is None:
            if isinstance(image_idx,int):
                image_idx = [image_idx]
            images = []
            for idx in image_idx:
                if idx<0:
                    continue
                elif 0<=idx and idx<self.dataset_lengths[0]:
                    img_path = ROOT+"test_set_data_compressed/Dafonts/"+self.image_names[idx]+".png"
                elif sum(self.dataset_lengths[:1])<=idx and idx<sum(self.dataset_lengths[:2]):
                    img_path = ROOT+"test_set_data_compressed/Simulacra/"+self.image_names[idx]+".png"
                elif sum(self.dataset_lengths[:2])<=idx and idx<sum(self.dataset_lengths[:3]):
                    img_path = ROOT+"test_set_data_compressed/LAION_aesthetics_6dot5/"+self.image_names[idx]+".jpg"
                pil_image = Image.open(img_path)
                images.append(np.array(pil_image))
        else:
            images = input_for_reshape

        if reshape_size is not None:
            if isinstance(reshape_size,tuple):
                reshape_size = list(reshape_size)
            if len(reshape_size)<3:
                reshape_size += [3]
            images = [cv2.resize(im, reshape_size[:2], interpolation=cv2.INTER_LINEAR) for im in images]
            for i in range(len(images)):                
                im_size = images[i].shape
                if len(im_size)==2:
                    images[i] = images[i][:,:,None]
                    im_size = images[i].shape
                assert reshape_size[2] in [1,3]
                if reshape_size[2]==3:
                    if im_size[2]==1:
                        images[i]=np.tile(images[i],(1,1,3))
                    elif im_size[2]==4:
                        images[i] = images[i][:3]
                    else:
                        assert im_size[2]==3
                elif reshape_size[2]==1:
                    images[i] = images[i].mean(2,keepdims=True)
        if return_torch and (reshape_size is not None):
            images = torch.stack([torch.from_numpy(im) for im in images],axis=0).permute((0,3,1,2))
        elif return_torch and (reshape_size is None):
            images = [torch.from_numpy(im) for im in images]
        return images
    
    def sample_group_idx(self,group_type):
        """Fast sampling function for getting a group idx

        Args:
            group_type (int): index of group type to sample from

        Returns:
            idx: sampled group idx
        """
        if group_type==0:
            idx = np.random.choice(self.group_type_lengths[0])
        elif group_type>0:
            idx = np.random.choice(self.group_type_lengths[group_type])+sum(self.group_type_lengths[:group_type])
        return idx
    
    def sample_negative_idx(self,group_type,group_indices):
        """Sample a negative index from a group type, but without sampling from indices in group_indices

        Args:
            group_type (int): group type index from which to sample from
            group_indices (list): list of group indices which constitute the positive groups, and therefore not sampled

        Returns:
            idx: sampled negative index
        """
        illegal_indices = sum([self.groups[id] for id in group_indices],[])
        for _ in range(5):
            if self.allow_dataset_mixing:
                idx = np.random.choice(sum(self.dataset_lengths))
            elif self.dataset_types[group_type]==0:
                idx = np.random.choice(self.dataset_lengths[0])
            else:
                k = self.dataset_types[group_type]
                idx = np.random.choice(self.dataset_lengths[k])+sum(self.dataset_lengths[:k])
            if idx not in illegal_indices:
                break
        return idx
    
    def __len__(self):
        return sum(self.dataset_lengths)

    def __getitem__(self, idx):
        max_collage_size = max(self.collage_sizes)
        max_candidate_size = max(self.candidate_sizes)

        n_collage = np.random.choice(self.collage_sizes)
        n_candidates = np.random.choice(self.candidate_sizes)

        info = {}
        group_type = np.random.choice(len(self.collage_types),p=self.collage_type_prob)

        info["group_type"] = [group_type,self.collage_types[group_type]]

        if self.small_group_strategy=="concat":
            positives = []
            group_indices = []
            for _ in range(max_collage_size+max_candidate_size):
                group_idx = self.sample_group_idx(group_type)
                group_indices.append(group_idx)
                positives.extend(self.groups[group_idx])

                if len(positives)>self.min_group_size:
                    break
        elif self.small_group_strategy=="ignore":
            group_idx = self.sample_group_idx(group_type)
            positives = self.groups[group_idx]
            group_indices = [group_idx]
        
        collage_idx = self.padding_idx*torch.ones(max_collage_size,dtype=int)
        collage_idx[:n_collage] = torch.from_numpy(np.random.choice(positives, size=n_collage)).long()

        if len(group_indices)>1:
            group_indices_actually_used = []
            for g_idx in group_indices:
                if any([(g_i in collage_idx[:n_collage]) for g_i in self.groups[g_idx]]):
                    group_indices_actually_used.append(g_idx)
            positives = sum([self.groups[id] for id in group_indices_actually_used],[])
            group_indices = group_indices_actually_used
        
        info["group_indices"] = group_indices
        info["group_names"] = [self.group_names[g_idx] for g_idx in group_indices]



        if self.filter_positives:
            positives_tmp = [n for n in positives if n not in collage_idx]
            if len(positives_tmp)>0:
                positives = positives_tmp

        label = self.padding_idx*torch.ones(max_candidate_size,dtype=torch.float32)
        candidate_idx = self.padding_idx*torch.ones(max_candidate_size,dtype=int)
        for i in range(n_candidates):
            if np.random.rand()<self.positive_candidate_prob: 
                candidate_idx[i] = np.random.choice(positives)
                label[i] = 1
            else:
                candidate_idx[i] = self.sample_negative_idx(group_type,group_indices)
                label[i] = 0
        
        if self.CNN:
            candidate = self.padding_idx*torch.ones(max_candidate_size,self.cnn_reshape[2],self.cnn_reshape[0],self.cnn_reshape[1],dtype=torch.float32)
            collage = self.padding_idx*torch.ones(max_collage_size,self.cnn_reshape[2],self.cnn_reshape[0],self.cnn_reshape[1],dtype=torch.float32)
            candidate[:n_candidates] = self.get_images(candidate_idx,reshape_size=self.cnn_reshape,return_torch=True)/255
            collage[:n_collage] = self.get_images(collage_idx,reshape_size=self.cnn_reshape,return_torch=True)/255
        else:
            candidate = self.padding_idx*torch.ones(max_candidate_size,self.clip_dim,dtype=torch.float32)
            collage = self.padding_idx*torch.ones(max_collage_size,self.clip_dim,dtype=torch.float32)
            candidate[:n_candidates] = self.clip_matrix[candidate_idx[:n_candidates]]
            collage[:n_collage] = self.clip_matrix[collage_idx[:n_collage]]

        return collage, candidate, label, collage_idx, candidate_idx, info

def custom_collate_with_info(original_batch):
    """Custom collate function to enables a list of objects as the last tuple element of the batch

    Args:
        original_batch (list): batch from a torch dataloader to collate

    Returns:
        tuple with the "normal" batch in terms of torch tensors as the first k-1 elements and the last k'th element as a list
    """
    n = len(original_batch[0])
    normal_batch = []
    for i in range(n):
        list_of_items = [item[i] for item in original_batch]
        if i+1==n:
            info = list_of_items
        else:
            normal_batch.append(torch.stack(list_of_items,axis=0))
    return *normal_batch,info



def plot_loss_dict_multi(loss_dict_list,
                   names=None,
                   plot_name="acc_t",
                   vali_every=1000,
                   figsize=(5,8),
                   sigma_loss_t=10,
                   sigma_acc_t=10,
                   alpha=1.0,
                   colors=None,
                   vertical_subplots=True):
    """Plot loss graphs for multiple models

    Args:
        loss_dict_list (list): list of loss dictionaries
        names (list, optional): name of models corresponding to their loss dicts. Defaults to None.
        plot_name (Union[str,list], optional): Either "acc_t", "loss_t" or a list with multiple of these, which is what will be plotted. (still plots validation). Defaults to "acc_t".
        vali_every (int, optional): How often validation was done during training. Defaults to 1000.
        figsize (tuple, optional): size of generated figure. Defaults to (5,8).
        sigma_loss_t (int, optional): Standard deviation for gaussian filtering of the training loss if it is plotted. Defaults to 10.
        sigma_acc_t (int, optional): Standard deviation for gaussian filtering of the training accuracy if it is plotted. Defaults to 10.
        alpha (float, optional): alpha value for training graphs. Defaults to 1.0.
        colors (list, optional): Colors for the graphs, same length as the loss_dict_list. Defaults to None.
        vertical_subplots (bool, optional): Should multiple plots be above and below eachother (True) or side by side (False). Defaults to True.
    """
    n_models = len(loss_dict_list)
    if colors is None:
        colors = ["C"+str(i) for i in range(n_models)]
    plot_names = [plot_name] if isinstance(plot_name,str) else plot_name
    acc_bool = any([plot_name.find("acc")>=0 for plot_name in plot_names])
    loss_bool = any([plot_name.find("loss")>=0 for plot_name in plot_names])
    n_plots = int(acc_bool)+int(loss_bool)
    figsize = (figsize[0]*n_plots,figsize[1])
    plt.figure(figsize=figsize)
    for plot_name in plot_names:
        loss_bool_i = plot_name.find("loss")>=0
        vali_bool_i = plot_name.find("_v")>=0
        if n_plots>1:
            plt.subplot(n_plots,1,1 if loss_bool_i else 2) if vertical_subplots else plt.subplot(1,n_plots,1 if loss_bool_i else 2)
        for i in range(n_models):
            if vali_bool_i:
                n_iter = len(loss_dict_list[i][plot_name.replace("_v","_t")])
                x = np.arange(0,n_iter,vali_every)
                y = loss_dict_list[i][plot_name]
            else:
                n_iter = len(loss_dict_list[i][plot_name])
                x = np.arange(n_iter)
                y = loss_dict_list[i][plot_name]
                y = nd.gaussian_filter(y,sigma=sigma_loss_t if loss_bool_i else sigma_acc_t)
            plt.plot(x,y,
                     "-o" if vali_bool_i else "-",
                     alpha=1.0 if vali_bool_i else alpha,
                     color=colors[i])
        plt.xlabel("Ite")
    
    legend_elements = [Line2D([0], [0], color=colors[i], lw=2, label=names[i]) for i in range(n_models)]
    legend_elements.append(Line2D([0], [0], color="k", marker='o', lw=2, label="Validation"))
    legend_elements.append(Line2D([0], [0], color="k", lw=2, alpha=alpha, label="Training"))
    
    if n_plots==1:
        plt.ylabel("Loss" if i==1 else "Accuracy")
        plt.legend(handles=legend_elements,loc="upper left")
    else:
        for i in range(1,n_plots+1):
            plt.subplot(n_plots,1,i) if vertical_subplots else plt.subplot(1,n_plots,i)
            plt.legend(handles=legend_elements)
            plt.ylabel("Loss" if i==1 else "Accuracy")