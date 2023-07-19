from typing import Optional, Tuple, Union
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from torch.nn import CrossEntropyLoss
import torch
import warnings

import engine


def new_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **deprecated_arguments,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` could have been `torch.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Only compute LOGITS FOR LAST TOKEN!!!!!!!!!!!!!
        # Greatly help to save memory!!!!!!!!!!!!!!
        print('We made it this far!')
        lm_logits = self.lm_head(hidden_states[:, -1:, :])

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            batch_size, seq_length, vocab_size = shift_logits.shape
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(batch_size * seq_length, vocab_size), shift_labels.view(batch_size * seq_length)
            )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


prompt = """Monkeys are captivating creatures that have long intrigued humans with their playful antics, social structures, and remarkable adaptations.

One of the defining features of monkeys is their incredible diversity. There are over 260 known species of monkeys, each with its own distinct traits and adaptations. They come in a wide range of sizes, from the tiny pygmy marmoset, which can fit in the palm of your hand, to the large and powerful mandrill, known for its strikingly colorful face. This diversity allows monkeys to occupy various ecological niches and adapt to different habitats and diets.
"""
max_tokens = 50
batch_size = 200

model = engine.HFModel('bloom-560M', gpu_rank=0, device_map='balanced_low_0')

# Hijack the forward method with our new better optimized one
original_forward = model.model.__class__.forward
model.model.__class__.forward = new_forward

print(model.device_map)
print(model.input_device)
# input_size = model.tokenizer.encode(prompt, return_tensors='pt').shape[1]
# print(f'Input sequence size: {input_size}')

for i in range(torch.cuda.device_count()):
    print(f'Before generation gpu {i}: {(torch.cuda.max_memory_allocated(i) / 1024**3):.5f} GB')

for i in range(torch.cuda.device_count()):
    torch.cuda.reset_peak_memory_stats(device=i)

out1 = model(prompt, num_return_sequences=200, max_new_tokens=max_tokens, seed=1)
# input_ids = model.tokenizer.encode(prompt, return_tensors='pt')
# large_input, _ = model.model._expand_inputs_for_generation(expand_size=200, input_ids=input_ids)
# out1 = model.model(large_input)

for i in range(torch.cuda.device_count()):
    print(f'After generation with trick gpu {i}: {(torch.cuda.max_memory_allocated(i) / 1024**3):.5f} GB')

del model

# Put back original method
model = engine.HFModel('bloom-560M', gpu_rank=0, device_map='balanced_low_0')
model.model.__class__.forward = original_forward

for i in range(torch.cuda.device_count()):
    torch.cuda.reset_peak_memory_stats(device=i)

out2 = model(prompt, num_return_sequences=200, max_new_tokens=max_tokens, seed=1)
# input_ids = model.tokenizer.encode(prompt, return_tensors='pt')
# large_input, _ = model.model._expand_inputs_for_generation(expand_size=200, input_ids=input_ids)
# out2 = model.model(large_input)

for i in range(torch.cuda.device_count()):
    print(f'After generation without trick gpu {i}: {(torch.cuda.max_memory_allocated(i) / 1024**3):.5f} GB')

print(f'The outputs are similar: {out1 == out2}')
# print(f'The outputs are similar: {torch.all(out1 == out2)}')