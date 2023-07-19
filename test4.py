from typing import Optional, Tuple, Union
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from torch.nn import CrossEntropyLoss
import torch

import engine


def new_forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Only compute LOGITS FOR LAST TOKEN!!!!!!!!!!!!!
        # Greatly help to save memory!!!!!!!!!!!!!!
        lm_logits = self.lm_head(hidden_states[:, -1:, :])

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )


prompt = """Monkeys are captivating creatures that have long intrigued humans with their playful antics, social structures, and remarkable adaptations. 
"""
max_tokens = 512
batch_size = 200

model = engine.HFModel('bloom-560M', gpu_rank=0, device_map='balanced_low_0')
model.forward = new_forward
print(model.device_map)
print(model.input_device)
input_size = model.tokenizer.encode(prompt, return_tensors='pt').shape[1]
print(f'Input sequence size: {input_size}')

for i in range(torch.cuda.device_count()):
    print(f'Before generation gpu {i}: {(torch.cuda.max_memory_allocated(i) / 1024**3):.5f} GB')

for i in range(torch.cuda.device_count()):
    torch.cuda.reset_peak_memory_stats(device=i)

# out1 = model(prompt, num_return_sequences=200, max_new_tokens=max_tokens, seed=1)
input_ids = model.tokenizer.encode(prompt, return_tensors='pt')
large_input, _ = model.model._expand_inputs_for_generation(expand_size=200, input_ids=input_ids)
out1 = model.model(large_input)

for i in range(torch.cuda.device_count()):
    print(f'After generation with trick gpu {i}: {(torch.cuda.max_memory_allocated(i) / 1024**3):.5f} GB')

del model
model = engine.HFModel('bloom-560M', gpu_rank=0, device_map='balanced_low_0')

for i in range(torch.cuda.device_count()):
    torch.cuda.reset_peak_memory_stats(device=i)

# out2 = model(prompt, num_return_sequences=200, max_new_tokens=max_tokens, seed=1)
input_ids = model.tokenizer.encode(prompt, return_tensors='pt')
large_input, _ = model.model._expand_inputs_for_generation(expand_size=200, input_ids=input_ids)
out2 = model.model(large_input)

for i in range(torch.cuda.device_count()):
    print(f'After generation without trick gpu {i}: {(torch.cuda.max_memory_allocated(i) / 1024**3):.5f} GB')

# print(f'The outputs are similar: {out1 == out2}')
print(f'The outputs are similar: {torch.all(out1 == out2)}')