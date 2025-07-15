import yaml
from enum import Enum 
from dataclasses import dataclass 

class TokenizationStrategy(Enum):
    JOINT = 'joint' 
    SEPARATE = "separate"
@dataclass
class ModelConfig:
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

@dataclass
class TrainingConfig:
    batch_size: int
    epochs: int
    lr: float
    num_workers: int
    quick_eval_every: int 
    full_eval_every: int

@dataclass 
class ExperimentConfig:
    base_dir: str = "experiments" 
    checkpoint_dir: str = "checkpoints" 
    save_every_epoch: int = 1 
    keep_last_n: int = 3 
    log_every: int = 100 
    log_dir: str = "logs"

@dataclass
class DataConfig:
    dataset_name: str
    subset: str
    lang_src: str
    lang_tgt: str
    tokenization_strategy: TokenizationStrategy

@dataclass
class AppConfig:
    model: ModelConfig
    training: TrainingConfig
    experiment: ExperimentConfig
    data: DataConfig

def load_config(path: str) -> AppConfig:
    """Loads the YAML config and returns a typed AppConfig object."""
    with open(path, 'r') as f:
        yaml_data = yaml.safe_load(f)
    
    # create config objects from the dictionary
    model_cfg = ModelConfig(**yaml_data['model'])
    training_cfg = TrainingConfig(**yaml_data['training'])
    experiment_cfg = ExperimentConfig(**yaml_data['experiment']) 
    data_dict = yaml_data['data'] 
    data_dict['tokenization_strategy'] = TokenizationStrategy(data_dict['tokenization_strategy']) 
    data_cfg = DataConfig(**data_dict)

    return AppConfig(model=model_cfg, training=training_cfg,experiment=experiment_cfg, data=data_cfg)