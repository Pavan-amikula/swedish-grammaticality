"""CPU training: a frozen head, then paired fine-tuning of six ModernBERT blocks.

No test split is read. All model choices use a fixed validation objective;
temperature calibration uses a separate set of source-correction families.
"""
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from scipy.stats import spearmanr
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed
from transformers import get_linear_schedule_with_warmup

BASE = "AI-Sweden-Models/ModernBERT-base"
REVISION = "0bf738233da3c1bd32bc88cdbeb2c2c01dbac0eb"
SEED = 42
MAX_LENGTH = 128
PROTOCOL = {"seed":SEED,"max_length":MAX_LENGTH,"frozen_head_max_epochs":12,
            "frozen_head_patience":3,"fine_tune_epochs":2,"fine_tune_blocks":6,
            "encoder_learning_rate":3e-5,"head_learning_rate":3e-4,
            "weight_decay":0.01,"train_batch_size":16,"inference_batch_size":32,
            "warmup_fraction":0.06,"objective":"0.75*human_validation_auc+0.25*ud_validation_auc",
            "paired_rank_loss_weight":0.5,"fine_tune_dropout":0.0,
            "temperature_grid":[0.5,0.75,1.0,1.5,2.0,3.0,5.0]}


def read(name):
    return [json.loads(l) for l in Path(f"data/grammar/{name}.jsonl").read_text(encoding="utf-8").splitlines()]


def metrics(rows, scores):
    y = [int(r["score"]) for r in rows]
    return {"n":len(rows),"roc_auc":float(roc_auc_score(y,scores)),
            "spearman":float(spearmanr(y,scores).statistic),
            "accuracy":float(accuracy_score(y,np.asarray(scores)>=0.5)),
            "brier":float(brier_score_loss(y,scores)),
            "log_loss":float(log_loss(y,np.clip(scores,1e-7,1-1e-7),labels=[0,1]))}


def tensor_batches(encoded, tokenizer, batch_size, seed=None):
    # Length buckets reduce CPU padding; a seeded permutation prevents label order.
    indices = sorted(range(len(encoded)),key=lambda i:len(encoded[i]["input_ids"]))
    rng = np.random.default_rng(seed)
    batches = []
    for start in range(0,len(indices),256):
        bucket = indices[start:start+256]
        if seed is not None:
            rng.shuffle(bucket)
        batches.extend(bucket[i:i+batch_size] for i in range(0,len(bucket),batch_size))
    if seed is not None:
        rng.shuffle(batches)
    for batch in batches:
        yield batch,tokenizer.pad([encoded[i] for i in batch],padding=True,return_tensors="pt")


@torch.no_grad()
def pooled_features(encoded, model, tokenizer, label):
    output = torch.empty((len(encoded),model.config.hidden_size))
    model.eval()
    for n,(indices,batch) in enumerate(tensor_batches(encoded,tokenizer,32)):
        hidden = model.model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1)
        output[indices] = (hidden*mask).sum(1)/mask.sum(1)
        if n%50 == 0:
            print(f"Cached {label}: {n*32}/{len(encoded)}",flush=True)
    return output


def head_logits(x,model):
    return model.classifier(model.drop(model.head(x)))


def paired_batches(rows,encoded,tokenizer,batch_size,seed=None):
    groups = {}
    for i,row in enumerate(rows):
        groups.setdefault(row["group"],{})[int(row["score"])] = i
    assert all(set(g)=={0,1} for g in groups.values())
    pairs = sorted(([g[1],g[0]] for g in groups.values()),
                   key=lambda pair:max(len(encoded[i]["input_ids"]) for i in pair))
    rng = np.random.default_rng(seed)
    batches = []
    for start in range(0,len(pairs),128):
        bucket = pairs[start:start+128]
        if seed is not None:
            rng.shuffle(bucket)
        for at in range(0,len(bucket),batch_size//2):
            batches.append([i for pair in bucket[at:at+batch_size//2] for i in pair])
    if seed is not None:
        rng.shuffle(batches)
    for indices in batches:
        yield indices,tokenizer.pad([encoded[i] for i in indices],padding=True,return_tensors="pt")


@torch.no_grad()
def evaluate(model, data, tokenizer, encoded=None, cached=None):
    model.eval()
    if cached is not None:
        logits = torch.cat([head_logits(cached[i:i+256],model) for i in range(0,len(cached),256)])
    else:
        logits = torch.empty((len(data),2))
        for indices,batch in tensor_batches(encoded,tokenizer,32):
            logits[indices] = model(**batch).logits
    scores = logits.softmax(-1)[:,1].numpy()
    return metrics(data,scores),logits


def validate(model, rows, tokenizer, encoded, cached=None):
    result = {}
    for name in ["validation","bridge_validation"]:
        result[name],_ = evaluate(model,rows[name],tokenizer,encoded[name],None if cached is None else cached[name])
    result["objective"] = 0.75*result["validation"]["roc_auc"]+0.25*result["bridge_validation"]["roc_auc"]
    return result


def snapshot(model):
    return {name:p.detach().clone() for name,p in model.named_parameters() if p.requires_grad}


def restore(model,state):
    with torch.no_grad():
        for name,p in model.named_parameters():
            if name in state:
                p.copy_(state[name])


def main():
    started = time.perf_counter()
    Path("results").mkdir(exist_ok=True)
    Path("results/grammar_protocol.json").write_text(json.dumps(PROTOCOL,indent=2),encoding="utf-8")
    set_seed(SEED)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    rows = {s:read(s) for s in ["train","bridge_train","validation","bridge_validation","calibration"]}
    rows["train"] += rows.pop("bridge_train")
    tokenizer = AutoTokenizer.from_pretrained(BASE,revision=REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(BASE,revision=REVISION,num_labels=2,
                id2label={0:"incorrect",1:"correct"},label2id={"incorrect":0,"correct":1},
                classifier_pooling="mean",attn_implementation="eager").eval()
    model.config.scoring_max_length = MAX_LENGTH
    model.config.scoring_attention_implementation = "eager"
    model.config.problem_type = "single_label_classification"
    model.config.training_provenance = {"base":BASE,"revision":REVISION,"protocol":PROTOCOL,
                                        "data":"data/grammar/provenance.json"}
    encoded = {}
    for name,data in rows.items():
        batch = tokenizer([r["text"] for r in data],truncation=True,max_length=MAX_LENGTH)
        encoded[name] = [{k:v[i] for k,v in batch.items()} for i in range(len(data))]
    # Stage 1: cache a frozen encoder once; fit only pooler/classifier.
    for p in model.parameters():
        p.requires_grad_(False)
    for module in [model.head,model.classifier]:
        for p in module.parameters():
            p.requires_grad_(True)
    cached = {name:pooled_features(encoded[name],model,tokenizer,name)
              for name in ["train","validation","bridge_validation"]}
    y = torch.tensor([int(r["score"]) for r in rows["train"]])
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4,weight_decay=0.01)
    trials, best, patience = [], None, 0
    for epoch in range(1,13):
        generator = torch.Generator().manual_seed(SEED+epoch)
        indices = torch.randperm(len(y),generator=generator)
        model.drop.train()
        for start in range(0,len(y),64):
            ix = indices[start:start+64]
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head_logits(cached["train"][ix],model),y[ix])
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
            optimizer.step()
        result = {"stage":"frozen_head","epoch":epoch,**validate(model,rows,tokenizer,encoded,cached)}
        trials.append(result)
        print(json.dumps(result),flush=True)
        if best is None or result["objective"]>best[0]["objective"]:
            best = result,snapshot(model)
            patience = 0
        else:
            patience += 1
        if patience>=3:
            break
    restore(model,best[1])
    frozen_result = best[0]
    model.eval().save_pretrained("candidates/modernbert_frozen_head")
    tokenizer.save_pretrained("candidates/modernbert_frozen_head")
    del cached
    # Stage 2: continue from the selected head, adapting six contextual blocks.
    for layer in model.model.layers[-6:]:
        for p in layer.parameters():
            p.requires_grad_(True)
    head = [p for name,p in model.named_parameters() if p.requires_grad and
            (name.startswith("classifier") or name.startswith("head"))]
    encoder = [p for name,p in model.named_parameters() if p.requires_grad and
               not (name.startswith("classifier") or name.startswith("head"))]
    optimizer = torch.optim.AdamW([{"params":encoder,"lr":3e-5},{"params":head,"lr":3e-4}],weight_decay=0.01)
    steps_per_epoch = sum(1 for _ in paired_batches(rows["train"],encoded["train"],tokenizer,16))
    scheduler = get_linear_schedule_with_warmup(optimizer,int(0.06*2*steps_per_epoch),2*steps_per_epoch)
    best_fine = None
    for epoch in range(1,3):
        epoch_started = time.perf_counter()
        model.train()
        # Paired views should differ by the annotated edit, not random dropout.
        for module in model.modules():
            if isinstance(module,torch.nn.Dropout):
                module.eval()
        model.model.embeddings.eval()
        for layer in model.model.layers[:-6]:
            layer.eval()
        total_loss = 0.0
        for step,(indices,batch) in enumerate(paired_batches(rows["train"],encoded["train"],tokenizer,16,SEED+epoch)):
            optimizer.zero_grad(set_to_none=True)
            logits = model(**batch).logits
            classification = F.cross_entropy(logits,y[indices])
            evidence = logits[:,1]-logits[:,0]
            ranking = F.softplus(-(evidence[0::2]-evidence[1::2])).mean()
            loss = 0.5*classification+0.5*ranking
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.detach())
            if step%50 == 0:
                print(f"Fine-tune epoch {epoch}: {step}/{steps_per_epoch}, loss={total_loss/(step+1):.4f}, seconds={time.perf_counter()-epoch_started:.1f}",flush=True)
        result = {"stage":"fine_tune_last_6","epoch":epoch,"mean_loss":total_loss/steps_per_epoch,
                  "epoch_seconds":time.perf_counter()-epoch_started,**validate(model,rows,tokenizer,encoded)}
        trials.append(result)
        print(json.dumps(result),flush=True)
        if best_fine is None or result["objective"]>best_fine[0]["objective"]:
            best_fine = result,snapshot(model)
    restore(model,best_fine[1])
    fine_result = best_fine[0]
    if frozen_result["objective"]>fine_result["objective"]:
        model = AutoModelForSequenceClassification.from_pretrained("candidates/modernbert_frozen_head",attn_implementation="eager").eval()
        selected = frozen_result
    else:
        selected = fine_result
    # Temperature is fitted on reserved families, never on validation/test.
    uncalibrated, logits = evaluate(model,rows["calibration"],tokenizer,encoded["calibration"])
    target = torch.tensor([int(r["score"]) for r in rows["calibration"]])
    temperatures = [{"temperature":t,"nll":float(F.cross_entropy(logits/t,target))} for t in PROTOCOL["temperature_grid"]]
    temperature = min(temperatures,key=lambda r:r["nll"])["temperature"]
    with torch.no_grad():
        model.classifier.weight.div_(temperature)
        model.classifier.bias.div_(temperature)
    model.config.scoring_temperature = temperature
    model.config.training_provenance["selected"] = selected
    model.eval().save_pretrained("grammar_model")
    tokenizer.save_pretrained("grammar_model")
    result = {"base":BASE,"revision":REVISION,"rows":{n:len(r) for n,r in rows.items()},
              "trials":trials,"selected":selected,"temperature_trials":temperatures,
              "temperature":temperature,"uncalibrated_calibration":uncalibrated,
              "parameters":sum(p.numel() for p in model.parameters()),
              "trainable_parameters_stage2":sum(p.numel() for p in encoder+head),
              "training_seconds":time.perf_counter()-started}
    Path("results/grammar_training.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="trials"}),flush=True)


if __name__ == "__main__":
    main()
