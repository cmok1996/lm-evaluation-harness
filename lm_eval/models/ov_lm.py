import logging
import time

import requests
from requests.exceptions import RequestException
from tqdm import tqdm

from transformers import AutoTokenizer

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

import openvino_genai

import asyncio
import copy

logger = logging.getLogger(__name__)

@register_model("openvino_genai")
class OVLM(LM):
    def __init__(self, model_path = None,  **kwargs):
        super().__init__()
        assert model_path, "must pass `model_path`"
        self.model_path = model_path
        self.model_id = kwargs.get('model_id', None)
        self.device = kwargs.get('device', 'CPU')
        pipeline_config = kwargs.get('pipeline_config', dict())
        self.model = openvino_genai.LLMPipeline(model_path, self.device.upper(), pipeline_config)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        except:
            self.tokenizer = self.model.get_tokenizer()
        
        if 'instruct' in self.model_id.lower():
            self.is_instruct = True
        else:
            self.is_instruct = False

        self.generation_mapping = {'max_gen_toks': 'max_new_tokens', 
                                   'temperature': 'temperature', 
                                   'do_sample': 'do_sample', 
                                   'until': 'stop_strings', 
                                   'logprobs': 'logprobs'}
        
    def process_prompt(self, prompt, is_tokenize):
        if self.is_instruct:
            messages = [{'role': 'user', 'content': prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt=True)
        else:
            formatted_prompt = copy.deepcopy(prompt)

        if is_tokenize:
            prompt_encoded = self.tokenizer(formatted_prompt, return_tensors = 'pt')
            return prompt_encoded
        else:
            return formatted_prompt

    # def ov_completions(self, prompt, streamer, generation_config, is_return_dict = False):
    #     if self.is_instruct:
    #         messages = [{'role': 'user', 'content': prompt}]
    #         formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt=True)
    #     else:
    #         formatted_prompt = copy.deepcopy(prompt)

    #     if is_return_dict:
    #         prompt_encoded = self.tokenizer(formatted_prompt, return_tensors = 'pt')
    #         out = self.model.generate(**prompt_encoded, streamer = streamer, **generation_config)
    #     else:
    #         out = self.model.generate(formatted_prompt, streamer = streamer, **generation_config)
        
    #     return out
    
    def generate_until(self, requests):
        if not requests:
            return []
    
        res = []
        for request in tqdm([req.args for req in requests]):
            inp = request[0]
            request_args = request[1]
            generation_config = {self.generation_mapping.get(k,k): v for k,v in request_args.items()}
            generation_config['max_length'] = 4096
            # generation_config['top_k'] = 0
            # generation_config['top_p'] = 1
            generation_config['do_sample'] = False
            
            # until = request_args.get("until", ["</s>"])
            # temperature = request_args.get('temperature')
            # max_tokens = request_args.get('max_gen_toks')

            streamer = None #lambda x: print(x, end='', flush=True)

            # prompt input as string
            processed_prompt = self.process_prompt(prompt = inp, is_tokenize = False)
            response = self.model.generate(processed_prompt, streamer = streamer, **generation_config)
            if isinstance(response, str):
                res.append(response)
            else:
                logger.error(
                            f"Invalid response for generate_until. Response: {response}"
                        )
                
        return res

    def loglikelihood(self, requests):
        raise NotImplementedError(
            "loglikelihood_rolling not yet supported for MY_GGUF models"
        )  
    
    def loglikelihood_rolling(self, requests):
        raise NotImplementedError(
            "loglikelihood_rolling not yet supported for MY_GGUF models"
        )
                
            


          