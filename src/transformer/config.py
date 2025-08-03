import yaml
from enum import Enum 
from pydantic import BaseModel 
class TokenizationStrategy(Enum):
    JOINT = 'joint' 
    SEPARATE = "separate"
class ModelConfig(BaseModel):
    src_vocab_size: int
    tgt_vocab_size: int
    d_model: int
    num_heads: int
    d_ff: int
    num_encoder_layers: int
    num_decoder_layers: int
    dropout: float
    src_max_len: int
    tgt_max_len: int

class TrainingConfig(BaseModel):
    seed: int 
    batch_size: int
    epochs: int
    lr_factor: float
    num_workers: int
    quick_val_size: int
    quick_eval_every: int 
    full_eval_every: int
    warmup_steps: int 
    weight_decay: float 
    adam_eps: float
    adam_beta1: float 
    adam_beta2: float
    label_smoothing: float 
    max_grad_norm: float 

class ExperimentConfig(BaseModel):
    base_dir: str = "experiments" 
    checkpoint_dir: str = "checkpoints" 
    save_every_steps: int = 1 
    keep_last_n: int = 3 
    log_every: int = 100 
    log_dir: str = "logs"

class DataConfig(BaseModel):
    dataset_name: str
    subset: str
    lang_src: str
    lang_tgt: str
    tokenization_strategy: TokenizationStrategy
    validation_fraction: float

class AppConfig(BaseModel):
    model: ModelConfig
    training: TrainingConfig
    experiment: ExperimentConfig
    data: DataConfig

def load_config(path: str) -> AppConfig:
    """Loads the YAML config and returns a typed AppConfig object."""
    with open(path, 'r') as f:
        yaml_data = yaml.safe_load(f)
    
    return AppConfig(**yaml_data) 