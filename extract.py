import torch


def extract_minigpt4_attentions(model, image, prompt="Describe the image."):
    image_attn_holder = []
    qformer_attn_holder = []

    def hook_image_self_attn(module, input, output):
        image_attn_holder.append(output[1].detach().cpu())

    model.visual_encoder.blocks[-1].attn.register_forward_hook(hook_image_self_attn)
    def hook_qformer_cross_attn(module, input, output):
        qformer_attn_holder.append(output[0].detach().cpu())

    model.Qformer.bert.encoder.layer[-2].crossattention.self.register_forward_hook(hook_qformer_cross_attn)

    model.eval()
    with torch.inference_mode() and torch.autocast("cuda"):

        image_embeds = model.visual_encoder(image.cuda())  # [B, T, D]
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)  # [B, T]


        query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)  # [B, Q, D]

        _ = model.Qformer.bert(
            query_embeds=query_tokens.cuda(),
            encoder_hidden_states=image_embeds.cuda(),
            encoder_attention_mask=image_atts.cuda(),
            use_cache=True,
            return_dict=True,
            output_attentions=True,
        )


    image_attn = image_attn_holder[0] if image_attn_holder else None
    qformer_attn = qformer_attn_holder[0] if qformer_attn_holder else None

    image_attn = image_attn.mean(dim=(0, 1))  # [T, T]
    qformer_attn = qformer_attn.mean(dim=(0, 1))  # [Q, K]

    vis_ref = qformer_attn @ image_attn  # [Q, T]
    vis_ref = vis_ref.sum(dim=-1)  
    return vis_ref



def extract_llava_attentions(model, image,questions):
    with torch.no_grad():
        image = image['pixel_values'][0]
        vision_out = model.get_vision_tower().vision_tower(
            image.cuda(), output_attentions=True, return_dict=True
        )
        attn_vis = torch.concat(vision_out.attentions)
        attn_vis = attn_vis[:, :, 1:, 1:]
        vis_ref = attn_vis[-1,:,:,:].mean(dim=0).sum(dim=0)
    return vis_ref


def extract_shikra_attentions(model, image, prompt="Describe the image."):
    with torch.no_grad():
        image = image['pixel_values'][0]
        clip_last_attn = model.model.vision_tower[0]
        vision_out = clip_last_attn(
                image.cuda(), output_attentions=True, return_dict=True
            )
    attn_vis = torch.concat(vision_out.attentions) 
    attn_vis = attn_vis[:, :, :, :]
    vis_ref = attn_vis[-1,:,:,:].mean(dim=0).sum(dim=0)
    return vis_ref

def extract_vision(model_name, model, image, questions):
    if model_name == "minigpt4":
        return extract_minigpt4_attentions(model,image,questions)
    elif model_name=="llava-1.5":
        return extract_llava_attentions(model, image,questions)
    elif model_name=="shikra":
        return extract_shikra_attentions(model, image,questions)
    else:
        raise Exception("Please check model.")