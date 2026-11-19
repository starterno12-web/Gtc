# -*- coding: utf-8 -*-
"""PyTorch twin of Android Nlm. NLM2 layout is byte-for-byte compatible."""
import json, struct
import numpy as np

MAGIC=b"NLM2"

class GRULM:
    def __init__(self,e=64,h=384,chars=None):
        self.e=e; self.h=h; self.chars=list(chars or ["\n","�"])
        self.v=len(self.chars)
        self.P=[np.zeros((self.v,e),np.float32),np.zeros((e,3*h),np.float32),np.zeros((h,3*h),np.float32),np.zeros(3*h,np.float32),
                np.zeros((h,3*h),np.float32),np.zeros((h,3*h),np.float32),np.zeros(3*h,np.float32),np.zeros((h,self.v),np.float32),np.zeros(self.v,np.float32)]
        self._init()
    def _init(self):
        rng=np.random.default_rng(20261119)
        for i,p in enumerate(self.P):
            if p.ndim==2: p[:]=rng.normal(0,0.025 if i in (0,7) else 0.035,p.shape).astype(np.float32)
    def encode(self,s,grow=False):
        ids={c:i for i,c in enumerate(self.chars)}
        return np.asarray([ids.get(c,1) for c in s],dtype=np.int64)
    def bpc(self,ids):
        if len(ids)<2:return 99.
        import torch
        with torch.no_grad():
            loss=self._loss(torch.tensor(ids[:-1]),torch.tensor(ids[1:]))
        return float(loss.item()/np.log(2))
    def _loss(self,x,y):
        import torch
        p=[torch.as_tensor(a) for a in self.P]; h=self.h
        s1=torch.zeros(h);s2=torch.zeros(h); losses=[]
        for t in range(len(x)):
            ix=int(x[t]); gx=p[3]+p[0][ix]@p[1]
            def cell(gx,s,wh):
                a=gx[:2*h]+s@wh[:,:2*h]; r=torch.sigmoid(a[:h]); z=torch.sigmoid(a[h:])
                n=torch.tanh(gx[2*h:]+(r*s)@wh[:,2*h:])
                return (1-z)*s+z*n
            s1=cell(gx,s1,p[2]); s2=cell(p[6]+s1@p[4],s2,p[5])
            logits=p[8]+s2@p[7]
            losses.append(torch.nn.functional.cross_entropy(logits.view(1,-1),y[t].view(1)))
        return torch.stack(losses).mean()
    def to_bytes(self):
        ch=json.dumps(self.chars,ensure_ascii=False,separators=(",",":")).encode()
        head=MAGIC+struct.pack("<iiiI",self.e,self.h,self.v,len(ch))+ch
        return head+b"".join(np.asarray(a,dtype="<f4").tobytes(order="C") for a in self.P)
    @classmethod
    def from_bytes(cls,b):
        if b[:4]!=MAGIC: raise ValueError("not NLM2")
        e,h,v,n=struct.unpack_from("<iiiI",b,4); off=20
        chars=json.loads(b[off:off+n].decode()); off+=n
        m=cls(e,h,chars); sizes=[(v,e),(e,3*h),(h,3*h),(3*h,),(h,3*h),(h,3*h),(3*h,),(h,v),(v,)]
        for i,shape in enumerate(sizes):
            count=int(np.prod(shape)); end=off+count*4
            if end>len(b): raise ValueError("truncated NLM2 weights")
            m.P[i]=np.frombuffer(b[off:end],dtype="<f4").copy().reshape(shape); off=end
        if off!=len(b): raise ValueError("unexpected trailing bytes")
        return m
    def save(self,path):
        open(path,"wb").write(self.to_bytes())
