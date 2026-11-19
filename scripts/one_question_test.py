#!/usr/bin/env python3
"""One-shot, raw character-generation test for the repository's NLM2 GRU."""
import json, struct, urllib.request, pathlib, numpy as np

MODEL_URL = "https://raw.githubusercontent.com/starterno12-web/Gtc/main/nlm2.bin"
OUT = pathlib.Path("one_question_answer.txt")
QUESTION = "س: اكتب برمجة صفحة ويب تعرض لوحة ألوان متنوعة على خلفية بيضاء، وضع كلمة شغال وسط الصفحة بخط واضح. أعطني الناتج كنص برمجي فقط.\nج:"

def load_model():
    raw = urllib.request.urlopen(MODEL_URL, timeout=90).read()
    if raw[:4] != b"NLM2":
        raise ValueError("Model header isn't NLM2")
    e, h, v, n = struct.unpack_from("<iiiI", raw, 4)
    off = 20
    chars = json.loads(raw[off:off+n].decode("utf-8")); off += n
    shapes = [(v,e),(e,3*h),(h,3*h),(3*h,),(h,3*h),(h,3*h),(3*h,),(h,v),(v,)]
    weights=[]
    for shape in shapes:
        count=int(np.prod(shape)); end=off+count*4
        weights.append(np.frombuffer(raw[off:end],dtype="<f4").copy().reshape(shape)); off=end
    if off != len(raw): raise ValueError("Unexpected trailing model bytes")
    return chars, weights, h

def sigmoid(x):
    x=np.clip(x,-35,35)
    return 1/(1+np.exp(-x))

def cell(gx,s,wh):
    a=gx[:2*H]+s@wh[:,:2*H]
    r=sigmoid(a[:H]); z=sigmoid(a[H:])
    n=np.tanh(gx[2*H:]+(r*s)@wh[:,2*H:])
    return (1-z)*s+z*n

chars,P,H=load_model(); ids={c:i for i,c in enumerate(chars)}
unk=ids.get("�",1)
s1=np.zeros(H,dtype=np.float32); s2=np.zeros(H,dtype=np.float32)
# Prime state from prompt characters, then generate exactly one completion.
for ch in QUESTION:
    ix=ids.get(ch,unk)
    gx=P[3]+P[0][ix]@P[1]
    s1=cell(gx,s1,P[2]); s2=cell(P[6]+s1@P[4],s2,P[5])
logits=P[8]+s2@P[7]
rng=np.random.default_rng(2026); generated=[]
for _ in range(1100):
    logits=P[8]+s2@P[7]
    # Mild temperature and top-k sampling; no canned answer or prompt lookup.
    z=logits.astype(np.float64)/0.85
    top=np.argpartition(z,-35)[-35:]
    probs=np.exp(z[top]-np.max(z[top])); probs/=probs.sum()
    ix=int(rng.choice(top,p=probs))
    ch=chars[ix]; generated.append(ch)
    gx=P[3]+P[0][ix]@P[1]
    s1=cell(gx,s1,P[2]); s2=cell(P[6]+s1@P[4],s2,P[5])
answer="".join(generated).strip()
OUT.write_text("السؤال المرسل إلى النموذج:\n"+QUESTION+"\n\nالناتج الخام من النموذج (توليد حرفي، لا تحرير):\n"+answer+"\n",encoding="utf-8")
print("Saved one_question_answer.txt; generated characters:",len(generated))
