from ..llama_configs import LLAMA_CONFIGS


class T3Config:
    DEFAULT_MULTILINGUAL_TEXT_VOCAB = 2455

    def __init__(self, text_tokens_dict_size=704, *, multilingual: bool = False):
        self.start_text_token = 255
        self.stop_text_token = 0
        self.text_tokens_dict_size = text_tokens_dict_size
        self.max_text_tokens = 2048
        self._is_multilingual = multilingual

        self.start_speech_token = 6561
        self.stop_speech_token = 6562
        self.speech_tokens_dict_size = 8194
        self.max_speech_tokens = 4096

        self.llama_config_name = "Llama_520M"
        self.input_pos_emb = "learned"
        self.speech_cond_prompt_len = 150

        self.encoder_type = "voice_encoder"
        self.speaker_embed_size = 256
        self.use_perceiver_resampler = True
        self.emotion_adv = True

    @property
    def n_channels(self):
        return LLAMA_CONFIGS[self.llama_config_name]["hidden_size"]
    
    @property
    def is_multilingual(self):
        return self._is_multilingual

    @classmethod
    def english_only(cls):
        """Create configuration for English-only TTS model."""
        return cls(text_tokens_dict_size=704, multilingual=False)
    
    @classmethod 
    def multilingual(cls, text_tokens_dict_size: int | None = None):
        """Create configuration for multilingual TTS model."""
        if text_tokens_dict_size is None:
            text_tokens_dict_size = cls.DEFAULT_MULTILINGUAL_TEXT_VOCAB
        return cls(text_tokens_dict_size=text_tokens_dict_size, multilingual=True)
