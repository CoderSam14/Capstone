"""Center-heatmap and reference-frame motion losses for variable-N tracklets."""
import torch
import torch.nn.functional as F
from model import VariableTrackletConvLSTM


def make_center_target(coords, height, width, input_h, input_w, device, sigma=1.5):
    target=torch.zeros((height,width),device=device,dtype=torch.float32)
    center_indices=[]
    for xy in coords:
        x,y=float(xy[0]),float(xy[1])
        gx=x*(width-1)/max(input_w-1,1)
        gy=y*(height-1)/max(input_h-1,1)
        ix=min(width-1,max(0,int(round(gx))))
        iy=min(height-1,max(0,int(round(gy))))
        center_indices.append((ix,iy))
        radius=int(max(3,round(4*sigma)))
        x0,x1=max(0,ix-radius),min(width,ix+radius+1)
        y0,y1=max(0,iy-radius),min(height,iy+radius+1)
        yy,xx=torch.meshgrid(torch.arange(y0,y1,device=device,dtype=torch.float32),
                             torch.arange(x0,x1,device=device,dtype=torch.float32),indexing="ij")
        gaussian=torch.exp(-((xx-gx)**2+(yy-gy)**2)/(2*sigma*sigma))
        gaussian[iy-y0,ix-x0]=1.0
        target[y0:y1,x0:x1]=torch.maximum(target[y0:y1,x0:x1],gaussian)
    return target,center_indices


def center_focal_loss(logits,target,num_objects):
    p=torch.sigmoid(logits.float()).clamp(1e-6,1-1e-6)
    positive=target.eq(1.0)
    negative=target.lt(1.0)
    pos=-(torch.log(p)*(1-p).pow(2)*positive).sum()/max(int(num_objects),1)
    neg_weight=(1-target).pow(4)
    neg=-(torch.log(1-p)*p.pow(2)*neg_weight*negative).sum()/max(int(num_objects),1)
    return pos+neg


def compute_loss(output,sample):
    logits=output["center_logits"][0,0]
    motion=output["motion_normalized"][0]
    frames=sample["frames"]
    _,_,h,w=frames.shape
    coords=sample["coords"].to(logits.device)
    velocities=sample["velocities"].to(logits.device)
    n=int(coords.shape[0])
    ref=coords.shape[1]//2 if n else frames.shape[0]//2
    centers=coords[:,ref,:] if n else coords.new_empty((0,2))
    target,indices=make_center_target(centers,logits.shape[-2],logits.shape[-1],h,w,logits.device)
    center_loss=center_focal_loss(logits,target,n)
    if n:
        pred_vel=torch.stack([motion[:,iy,ix] for ix,iy in indices],dim=0)
        target_vel=velocities/VariableTrackletConvLSTM.velocity_scale
        velocity_loss=F.smooth_l1_loss(pred_vel.float(),target_vel.float(),beta=0.1)
        offsets=output["offsets"][0]
        target_offsets=[]
        for obj,(ix,iy) in enumerate(indices):
            x=float(centers[obj,0])*(logits.shape[-1]-1)/max(w-1,1)
            y=float(centers[obj,1])*(logits.shape[-2]-1)/max(h-1,1)
            target_offsets.append([x-ix,y-iy])
        target_offsets=torch.tensor(target_offsets,device=logits.device,dtype=torch.float32).clamp(-0.5,0.5)
        pred_offsets=torch.stack([offsets[:,iy,ix] for ix,iy in indices],dim=0)
        offset_loss=F.smooth_l1_loss(pred_offsets.float(),target_offsets,beta=0.1)
    else:
        velocity_loss=motion.float().sum()*0.0
        offset_loss=output["offsets"].float().sum()*0.0
    total=center_loss+2.0*velocity_loss+offset_loss
    stats={"total":float(total.detach()),"center":float(center_loss.detach()),
           "velocity":float(velocity_loss.detach()),"offset":float(offset_loss.detach()),"objects":n}
    return total,stats
