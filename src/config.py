import yaml
from dataclasses import dataclass 

@dataclass
class ModelConfig:
    vocab_size: int
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

@dataclass
class DataConfig:
    dataset_name: str
    subset: str
    lang_src: str
    lang_tgt: str
    tokenization_strategy: str

@dataclass
class AppConfig:
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig

def load_config(path: str) -> AppConfig:
    """Loads the YAML config and returns a typed AppConfig object."""
    with open(path, 'r') as f:
        yaml_data = yaml.safe_load(f)
    
    # create config objects from the dictionary
    model_cfg = ModelConfig(**yaml_data['model'])
    training_cfg = TrainingConfig(**yaml_data['training'])
    data_cfg = DataConfig(**yaml_data['data'])

    return AppConfig(model=model_cfg, training=training_cfg, data=data_cfg)