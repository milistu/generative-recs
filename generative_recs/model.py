from transformers import T5Config, T5ForConditionalGeneration


def create_model(
    vocab_size: int = 3026,
    pad_token_id: int = 3024,
    beam_size: int = 20,
    num_levels: int = 4,
    d_model: int = 384,
    d_kv: int = 64,
    d_ff: int = 1024,
    num_layers: int = 4,
    num_heads: int = 6,
    dropout_rate: float = 0.1,
) -> T5ForConditionalGeneration:
    """Create a T5 model for TIGER generative retrieval.

    Initializes T5ForConditionalGeneration from scratch (random weights)
    with a custom vocabulary for semantic ID prediction.

    Args:
        vocab_size: Total vocabulary size (semantic + user + special tokens).
        pad_token_id: Padding token ID. Also used as decoder_start_token_id
            (T5 uses pad as BOS for the decoder via _shift_right).
        beam_size: Number of beams for beam search.
        num_levels: Number of Semantic ID tokens, including the collision token.
        d_model: Model hidden dimension.
        d_kv: Key/value dimension per attention head.
        d_ff: Feed-forward intermediate dimension.
        num_layers: Number of encoder and decoder layers.
        num_heads: Number of attention heads.
        dropout_rate: Dropout probability.
    Returns:
        T5ForConditionalGeneration model with random weights.
    """
    config = T5Config(
        vocab_size=vocab_size,
        d_model=d_model,
        d_kv=d_kv,
        d_ff=d_ff,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        decoder_start_token_id=pad_token_id,
        pad_token_id=pad_token_id,
        eos_token_id=pad_token_id + 1,
    )

    model = T5ForConditionalGeneration(config)

    model.generation_config.num_beams = beam_size
    model.generation_config.num_return_sequences = beam_size
    model.generation_config.do_sample = False
    model.generation_config.early_stopping = False
    model.generation_config.max_length = 1 + num_levels
    model.generation_config.max_new_tokens = None

    return model
