import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSeq2SeqLM

import engine
from engine import generation
from engine import stopping
import time
import resource
import gc
import math

from typing import Optional, Tuple, Union
# from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from torch.nn import CrossEntropyLoss
import torch
import warnings
import time
import numpy as np
import os
from tqdm import tqdm

import engine
from helpers import utils, datasets

large_text = """Monkeys: Nature's Pranksters, Social Geniuses, and Ecological Wonders

Introduction

Monkeys, the enchanting inhabitants of our planet's diverse ecosystems, have long been subjects of fascination and study for scientists, wildlife enthusiasts, and the curious at heart. Their intriguing behaviors, remarkable adaptations, and complex social structures invite us into a world that is both astonishing and deeply entwined with our understanding of life on Earth. In this essay, we will embark on a journey through the realms of monkeys, exploring their evolutionary history, the rich tapestry of their classifications, the ecological significance they hold, the intricate dynamics of their societies, their modes of communication, and the critical importance of their conservation.

Evolutionary History

The tale of monkeys begins in the annals of evolutionary history, stretching back millions of years to a time when Earth was teeming with diverse life forms. Monkeys belong to the grand order of Primates, a lineage that emerged around 60 million years ago. This group shared common traits that set them apart from other mammals: agile grasping hands and feet, forward-facing eyes for depth perception, and increasingly enlarged brains, which laid the foundation for their remarkable cognitive abilities.

Around 35 million years ago, a pivotal moment occurred in the primate story - the division of the evolutionary tree. This division led to the emergence of two significant lineages: New World monkeys (Platyrrhini) and Old World monkeys (Catarrhini). The geographical separation between these groups set the stage for a diversity of adaptations and characteristics unique to each.

Classification and Diversity

Monkeys are a diverse family, representing over 260 species and encompassing a remarkable range of shapes, sizes, and behaviors. These fascinating creatures are categorized into two major families: Cebidae, encompassing New World monkeys, and Cercopithecidae, home to the Old World monkeys. The distinction between these two families extends beyond mere taxonomy; it shapes their ecology, behavior, and evolutionary pathways.

New World Monkeys (Family: Cebidae)

Within the New World monkeys, we encounter an astonishing array of life. These monkeys have mastered the art of adaptation to their specific environments in the Americas. Some swing effortlessly through the treetops, while others are known for their resonant vocalizations that echo through the forests. The capuchin monkeys exhibit remarkable intelligence, utilizing tools to extract food from hard-to-reach places. In contrast, tamarins and marmosets have evolved unique cooperative breeding systems that govern their social structures.

Old World Monkeys (Family: Cercopithecidae)

The Old World monkeys, found in Africa, Asia, and parts of Gibraltar, have charted their own evolutionary course. Here, we encounter the imposing baboons, known for their dog-like faces and robust physiques. Macaques, on the other hand, are exemplars of adaptability, thriving in diverse habitats, from snow-clad mountains to lush tropical forests. Mandrills, with their striking facial colors and imposing canines, claim the title of the world's largest monkeys, inhabiting the lush rainforests of Central Africa.

Ecological Significance

Beyond their aesthetic appeal and charismatic behaviors, monkeys play a pivotal role in the delicate balance of the ecosystems they call home. They are often referred to as "keystone species" because their presence or absence can trigger cascading effects throughout their habitats.

One of the most significant contributions of monkeys to their ecosystems is seed dispersal. As avid fruit consumers, they play a vital role in the regeneration of plant populations by spreading seeds as they move through forests. This not only helps maintain forest diversity but also contributes to the health and stability of these ecosystems.

Moreover, monkeys' selective feeding habits, such as leaf consumption by howler monkeys and colobus monkeys, shape the structure of forests by influencing the composition and density of vegetation. Their predation on insects, small mammals, and bird eggs helps regulate prey populations, ensuring the intricate balance of food webs within their habitats.

Monkeys, as indicator species, offer valuable insights into the health of their ecosystems. Their population trends can serve as early warnings of environmental disturbances, including habitat loss, deforestation, and climate change. As such, preserving monkey populations is not merely an act of conservation but also a means of safeguarding the well-being of entire ecosystems.

Social Structures and Behavior

Monkeys are renowned for their complex social structures and behaviors, which vary widely among species and are shaped by a multitude of ecological and environmental factors. These social dynamics are essential for their survival and reproduction, shedding light on the intricacies of primate evolution and behavior.

In the world of monkeys, social life revolves around the concept of the "troop." These groups, varying greatly in size and composition, are the core units of monkey society. Within these troops, a dominance hierarchy often emerges, dictating access to valuable resources such as food and mates. Dominant individuals assert their authority through various displays of dominance, including vocalizations and physical posturing.

The intricacies of monkey societies extend to mating strategies, parenting, and cooperation. Some species, like capuchin monkeys, display impressive problem-solving abilities and tool use, revealing the depths of their intelligence. Others, such as tamarins and marmosets, engage in cooperative breeding systems, with non-breeding individuals helping to care for the offspring of dominant breeding pairs.

Communication and Language

Monkeys communicate through a diverse array of vocalizations, gestures, and behaviors, revealing a rich tapestry of interaction within their social groups. Vocalizations range from the booming howls of howler monkeys to the more subtle chirps and calls of capuchins. Each species has developed its unique repertoire of sounds to convey information about food, danger, and social dynamics.

Beyond vocalizations, monkeys employ body language and facial expressions to convey emotions, assert dominance, or establish social bonds. The intricate dance of gestures and postures within monkey societies is a testament to their highly evolved communication systems.

Conservation Challenges

Despite their resilience and adaptability, monkeys face a host of challenges in the modern world. Habitat loss due to deforestation, logging, and urban expansion is a severe threat to their survival. The fragmentation of their habitats isolates populations, limiting genetic diversity and making them more vulnerable to diseases and other environmental pressures.

Illegal wildlife trade also poses a grave danger to monkeys. Poaching for the pet trade and traditional medicine markets, as well as the loss of individuals to captivity, disrupts natural populations and can have devastating consequences.

Moreover, monkeys often come into conflict with humans due to habitat encroachment. This can lead to retaliation killings, further endangering their populations. Climate change, with its unpredictable effects on ecosystems, adds another layer of uncertainty to their future.

Conclusion

Monkeys, these captivating creatures of the wild, offer us a glimpse into the wonders of the natural world. Their evolutionary journey, intricate social structures, communication methods, and ecological significance remind us of the interconnectedness of all life on Earth. As we strive to conserve these remarkable beings and their habitats, we not only safeguard the diversity of our planet but also enrich our understanding of the delicate tapestry of life that surrounds us. Monkeys are not merely subjects of fascination; they are ambassadors of the wild, beckoning us to protect and preserve the ecosystems they call home. In doing so, we honor our shared heritage and ensure a brighter future for all living creatures.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn

from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.modeling_outputs import CausalLMOutputWithPast, Seq2SeqLMOutput
from transformers import (
    MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
    MODEL_FOR_VISION_2_SEQ_MAPPING,
)
from transformers.utils import ExplicitEnum, ModelOutput, is_accelerate_available, logging
from transformers import DisjunctiveConstraint, PhrasalConstraint
from transformers import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
from transformers import GenerationConfig
from transformers import (
    EncoderNoRepeatNGramLogitsProcessor,
    EncoderRepetitionPenaltyLogitsProcessor,
    EpsilonLogitsWarper,
    EtaLogitsWarper,
    ExponentialDecayLengthPenalty,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    ForceTokensLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitNormalization,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    MinNewTokensLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    SequenceBiasLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalLogitsWarper,
    UnbatchedClassifierFreeGuidanceLogitsProcessor,
)
from transformers.generation import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)

from transformers.generation.utils import SampleOutput, SampleEncoderDecoderOutput, SampleDecoderOnlyOutput



def sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer = None,
        **model_kwargs,
    ) -> Union[SampleOutput, torch.LongTensor]:
       
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        this_peer_finished = False  # used by synced_gpus only
        # auto-regressive generation
        i = 0
        while True:
            i += 1
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            # if 'past_key_values' in model_inputs.keys() and model_inputs['past_key_values'] is not None:
                # s = model_inputs['past_key_values'][0].shape
                # assert all([x.shape == s for x in model_inputs['past_key_values']]), 'Missed'
                # tot = torch.stack(model_inputs['past_key_values'])
                # mem = tot.nelement()*tot.element_size() / 1024**3
                # print(f'Shape: {tot.shape}      memory: {mem:.3f} GiB')
                
                # tot = [torch.stack(x) for x in model_inputs['past_key_values']]
                # tot = torch.stack(tot)
                # mem = tot.nelement()*tot.element_size() / 1024**3
                # print(f'Shape: {tot.shape}      memory: {mem:.3f} GiB')
                
            # print(model_inputs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            # return input_ids

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            # stop if we exceed the maximum length
            if stopping_criteria(input_ids, scores):
                this_peer_finished = True

            if this_peer_finished and not synced_gpus:
                break

            # if i == 3:
            #     return input_ids

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return SampleEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return SampleDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return input_ids
        

       


def mem_usage(past_key_values):

    if isinstance(past_key_values, torch.Tensor):
        return past_key_values.nelement() * past_key_values.element_size()
    elif isinstance(past_key_values[0], torch.Tensor):
        return sum([x.nelement() * x.element_size() for x in past_key_values])
    else:
        return sum([mem_usage(x) for x in past_key_values])






       

model = engine.HFModel('star-chat-beta')
# model = engine.HFModel('llama2-13B')
model.extra_eos_tokens = []
model.model.__class__.sample = sample

ids = model.tokenizer.encode(large_text)
print(len(ids))
prompt = model.tokenizer.decode(ids[0:600], skip_special_tokens=True)
new_prompt = model.tokenizer.decode(ids[0:800], skip_special_tokens=True)




use_cache = True
output_attentions = False
output_hidden_states = False


with torch.no_grad():
    prompt_ids = model.tokenizer.encode(prompt, return_tensors='pt').cuda()
    new_prompt_ids = model.tokenizer.encode(new_prompt, return_tensors='pt').cuda()
    bar = torch.tensor([[6483]]).cuda()
    concat_ids = torch.cat([prompt_ids, bar], dim=-1).cuda()
    concat_new_ids = torch.cat([new_prompt_ids, bar], dim=-1).cuda()

    torch.cuda.reset_peak_memory_stats(0)
    actual_peak = torch.cuda.max_memory_allocated(0) / 1024**3
    output1 = model.model(prompt_ids, use_cache=True)
    mem = torch.cuda.max_memory_allocated(0) / 1024**3 - actual_peak
    print(f'Memory to compute small past key values: {mem}')

    torch.cuda.reset_peak_memory_stats(0)
    actual_peak = torch.cuda.max_memory_allocated(0) / 1024**3
    output2 = model.model(new_prompt_ids, use_cache=True)
    mem = torch.cuda.max_memory_allocated(0) / 1024**3 - actual_peak
    print(f'Memory to compute large past key values: {mem}')

    past_keys1 = output1.past_key_values
    past_keys2 = output2.past_key_values
    print(f'Small past key values: {mem_usage(past_keys1) / 1024**3} GiB')
    print(f'Large past key values: {mem_usage(past_keys2) / 1024**3} GiB')


with torch.no_grad():
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats(0)
    actual_peak = torch.cuda.max_memory_allocated(0) / 1024**3
    # foo = model(prompt, batch_size=1, max_new_tokens=5, min_new_tokens=5, seed=12, post_process_output=False,
    #             use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
    # foo = model.model(ids, use_cache=use_cache)
    inputs = model.model.prepare_inputs_for_generation(concat_ids, use_cache=use_cache, past_key_values=past_keys1)
    foo = model.model(**inputs)
    mem = torch.cuda.max_memory_allocated(0) / 1024**3 - actual_peak
    dt0 = time.time() - t0

    print(f'Mem first time: {mem} GiB')
    print(f'Time first time: {dt0:.2f} s')


with torch.no_grad():
    t1 = time.time()
    torch.cuda.reset_peak_memory_stats(0)
    actual_peak2 = torch.cuda.max_memory_allocated(0) / 1024**3
    # foo2 = model(prompt, batch_size=1, max_new_tokens=5, min_new_tokens=5, seed=12, post_process_output=False,
    #             use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
    # foo2 = model.model(ids, use_cache=use_cache)
    inputs = model.model.prepare_inputs_for_generation(concat_ids, use_cache=use_cache, past_key_values=past_keys1)
    foo2 = model.model(**inputs)
    mem2 = torch.cuda.max_memory_allocated(0) / 1024**3 - actual_peak2
    dt1 = time.time() - t1

print(f'Mem second time: {mem2} GiB')
print(f'Time second time: {dt1:.2f} s')


with torch.no_grad():
    t2 = time.time()
    torch.cuda.reset_peak_memory_stats(0)
    actual_peak4 = torch.cuda.max_memory_allocated(0) / 1024**3
    # foo4 = model(prompt, batch_size=1, max_new_tokens=500, min_new_tokens=500, seed=12, post_process_output=False,
    #             use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
    # foo4 = model.model(ids, use_cache=use_cache)
    inputs = model.model.prepare_inputs_for_generation(concat_ids, use_cache=use_cache, past_key_values=past_keys1)
    foo4 = model.model(**inputs)
    mem4 = torch.cuda.max_memory_allocated(0) / 1024**3 - actual_peak4
    dt2 = time.time() - t2

print(f'Mem with large max new tokens: {mem4} GiB')
print(f'Time with large max new tokens: {dt2:.2f} s')


with torch.no_grad():
    t3 = time.time()
    torch.cuda.reset_peak_memory_stats(0)
    actual_peak5 = torch.cuda.max_memory_allocated(0) / 1024**3
    # foo5 = model(new_prompt, batch_size=1, max_new_tokens=2, min_new_tokens=1, seed=12, post_process_output=False,
    #             use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
    # foo5 = model.model(ids, use_cache=use_cache)
    inputs = model.model.prepare_inputs_for_generation(concat_new_ids, use_cache=use_cache, past_key_values=past_keys2)
    foo5 = model.model(**inputs)
    mem5 = torch.cuda.max_memory_allocated(0) / 1024**3 - actual_peak5
    dt3 = time.time() - t3

print(f'Mem with new prompt: {mem5} GiB')
print(f'Time with new prompt: {dt3:.2f} s')