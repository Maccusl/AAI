import torch

# def extract_minigpt4_attentions(model, image, prompt="Describe the image."):
#     """
#     提取 MiniGPT-4 中：
#     1. ViT 最后一层 self-attention
#     2. Q-Former 最后一层 cross-attention

#     参数:
#         model: MiniGPT-4 实例
#         image: 输入图像张量（预处理后）
#         prompt: 文本 prompt（默认 "Describe the image."）

#     返回:
#         image_attn: [B, Heads, T, T]，视觉编码器 self-attention
#         qformer_attn: [B, Heads, Q, K]，Qformer cross-attention
#     """
#     image_attn_holder = []
#     qformer_attn_holder = []

#     # 1. 注册视觉编码器 self-attention hook
#     def hook_image_self_attn(module, input, output):
#         attn = output[1]
#         image_attn_holder.append(attn.detach().cpu())

#     handle1 = model.visual_encoder.blocks[-1].attn.register_forward_hook(hook_image_self_attn)

#     # 2. 注册 Qformer cross-attention hook
#     def hook_qformer_cross_attn(module, input, output):
#         attn = output[0]
#         qformer_attn_holder.append(attn.detach().cpu())

#     handle2 = model.Qformer.bert.encoder.layer[-2].crossattention.self.register_forward_hook(hook_qformer_cross_attn)

#     # 3. 构造输入并前向（使用 model.forward() 或 generate）
#     model.eval()
#     with torch.no_grad():
#         _ = model.generate(
#             images = image, 
#             texts = prompt
#         )

#     # 4. 移除 hook
#     handle1.remove()
#     handle2.remove()

#     # 5. 返回两个注意力
#     image_attn = image_attn_holder[0] if image_attn_holder else None
#     qformer_attn = qformer_attn_holder[0] if qformer_attn_holder else None
#     image_attn = image_attn.mean(dim=(0,1))
#     qformer_attn = qformer_attn.mean(dim=(0,1))
#     vis_ref = qformer_attn@image_attn
#     vis_ref = vis_ref.sum(dim=-1)
    
#     return vis_ref

def extract_minigpt4_attentions(model, image, prompt="Describe the image."):
    image_attn_holder = []
    qformer_attn_holder = []

    # 注册视觉 self-attn hook
    def hook_image_self_attn(module, input, output):
        image_attn_holder.append(output[1].detach().cpu())

    model.visual_encoder.blocks[-1].attn.register_forward_hook(hook_image_self_attn)

    # 注册 Qformer cross-attn hook
    def hook_qformer_cross_attn(module, input, output):
        qformer_attn_holder.append(output[0].detach().cpu())

    model.Qformer.bert.encoder.layer[-2].crossattention.self.register_forward_hook(hook_qformer_cross_attn)

    model.eval()
    with torch.inference_mode() and torch.autocast("cuda"):
        # 1. 视觉编码
        image_embeds = model.visual_encoder(image.cuda())  # [B, T, D]
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)  # [B, T]

        # 2. Qformer 输入准备
        query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)  # [B, Q, D]

        # 3. Qformer 前向
        _ = model.Qformer.bert(
            query_embeds=query_tokens.cuda(),
            encoder_hidden_states=image_embeds.cuda(),
            encoder_attention_mask=image_atts.cuda(),
            use_cache=True,
            return_dict=True,
            output_attentions=True,
        )

    # 处理输出
    image_attn = image_attn_holder[0] if image_attn_holder else None
    qformer_attn = qformer_attn_holder[0] if qformer_attn_holder else None

    image_attn = image_attn.mean(dim=(0, 1))  # [T, T]
    qformer_attn = qformer_attn.mean(dim=(0, 1))  # [Q, K]

    vis_ref = qformer_attn @ image_attn  # [Q, T]
    vis_ref = vis_ref.sum(dim=-1)  # [T]，每个 image token 的最终重要性
    return vis_ref



def extract_llava_attentions(model, image,questions):
    with torch.no_grad():
        image = image['pixel_values'][0]
        vision_out = model.get_vision_tower().vision_tower(
            image.cuda(), output_attentions=True, return_dict=True
        )
        attn_vis = torch.concat(vision_out.attentions) # -> (layers, H, T, T)
        attn_vis = attn_vis[:, :, 1:, 1:]
        vis_ref = attn_vis[-1,:,:,:].mean(dim=0).sum(dim=0)
        # attn_vis = attn_vis.mean(dim=0).mean(dim=0).sum(dim=0)
    return vis_ref


def extract_shikra_attentions(model, image, prompt="Describe the image."):
    with torch.no_grad():
        image = image['pixel_values'][0]
        clip_last_attn = model.model.vision_tower[0]
        vision_out = clip_last_attn(
                image.cuda(), output_attentions=True, return_dict=True
            )
    attn_vis = torch.concat(vision_out.attentions) # -> (layers, H, T, T)
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