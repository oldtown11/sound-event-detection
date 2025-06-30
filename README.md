# Global Enhanced Frame Prompt Tuning for Sound Event Detection

The code is being organized and will be open-sourced as soon as possible. Currently, only the innovative parts have been open-sourced. However, I will provide detailed instructions on how to use them. Feel free to DM me if you have any questions.

## Code Modifications

### 1. Detect Function Update
Modified the `detect` function in `sed_trainer_pretrained.py` to return audio:

```python
def detect(self, mel_feats, audio, model, embeddings=None, **kwargs):
    if embeddings is None:
        return model(self.scaler(self.take_log(mel_feats)), **kwargs)
    else:
        return model(self.scaler(self.take_log(mel_feats)), audio=audio, embeddings=embeddings, **kwargs
```

### 2. CRNN Model
- Integrated BEATS into the forward propagation pipeline
- Implemented prompt parameter passing functionality

### 3. BEATS Backbone Adjustment
Updated the `backbone.py` file in the BEATS module

---

## Implementation Notes

### Hardware Requirements
• Dual NVIDIA RTX 3090 GPUs (48GB combined VRAM)

### Known Issues & Solutions
- **MACs calculation errors**: Simply comment out the problematic code section
- **sed_teacher implementation**:
  - Requires custom implementation (avoid `deepcopy`)
  - Fully supports dual-GPU operation

---

## Dual-GPU Configuration
Add this to your `train_pretrained.py`:

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  
sed_student = CRNN(**config["net"])
sed_student = torch.nn.DataParallel(sed_student, device_ids=[0,1]).cuda()
sed_teacher = CRNN(**config["net"])
sed_teacher = torch.nn.DataParallel(sed_teacher, device_ids=[0,1]).cuda()
```

Basic usage example:
```python
trainer = Trainer(
    student_model=sed_student,
    teacher_model=sed_teacher,
    # Additional parameters...
)
```

---

## Citation
```bibtex
@INPROCEEDINGS{10889807,
  author={Yu, Shiyu and Gao, Lijian and Mao, Qirong},
  booktitle={ICASSP 2025}, 
  title={Global Enhanced Frame Prompt Tuning for Sound Event Detection}, 
  year={2025},
  pages={1-5},
  doi={10.1109/ICASSP49660.2025.10889807}}
```

---
