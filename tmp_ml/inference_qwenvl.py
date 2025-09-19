import torch
import torch.nn as nn
import time
import json
import os 

from diffusers import FluxPipeline, AutoencoderKL
from diffusers.image_processor import VaeImageProcessor


from transformers import T5EncoderModel, T5TokenizerFast, CLIPTokenizer, CLIPTextModel, T5Config
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoProcessor, Qwen2_5_VLForConditionalGeneration

from qwen_vl_utils import process_vision_info
from proj import create_proj3_qwen3b, create_proj3_qwen7b
from PIL import Image
import argparse
# from qwen_vl_utils import process_vision_info

import librosa
import soundfile as sf
from decord import VideoReader, cpu, gpu




parser = argparse.ArgumentParser("Inference", add_help=True)
# parser.add_argument('--minicpm_path', type=str, default="openbmb/MiniCPM-o-2_6")
parser.add_argument('--qwen_size', type=str, default='7b', choices=['3b', '7b'], help="Model size: 1b or 4b")
parser.add_argument('--qwen3b_path', type=str, default="/mnt/data/group/models/Qwen2.5-VL-3B-Instruct")
# parser.add_argument('--flux_path', type=str,  default="shuttleai/shuttle-3-diffusion")
parser.add_argument('--flux_path', type=str,  default="/leonardo_scratch/fast/EUHPC_R04_192/fmohamma/fast_weights/FLUX.1-schnell")
parser.add_argument('--use_answer', type=bool,  default=False)
parser.add_argument('--num_steps', type=int, default=20)
parser.add_argument('--num_gen_imgs', type=int, default=1)
parser.add_argument('--multi_gpu', action='store_true', help='Enable multi-GPU support')
args = parser.parse_args()

# 多GPU设备配置
if args.multi_gpu and torch.cuda.device_count() > 1:
    print(f"🔥 检测到 {torch.cuda.device_count()} 张GPU卡")
    device_qwen = "cuda:0"      # Qwen模型放在第一张卡
    device_flux = "cuda:1"      # FLUX相关模型放在第二张卡
    print(f"📍 设备分配:")
    print(f"  - Qwen模型: {device_qwen}")
    print(f"  - FLUX/VAE模型: {device_flux}")
else:
    print(f"🔥 使用单GPU模式")
    device_qwen = "cuda:0"
    device_flux = "cuda:0"
    print(f"📍 设备: {device_qwen}")

dtype = torch.bfloat16

outputs = "./outputs_qwen3b"

if args.qwen_size == "7b":
    outputs = "./outputs_qwen7b"
    qwen_path = '/leonardo_work/EUHPC_R04_192/fmohamma/TRIG/data/Qwen2.5-VL-7B-Instruct'
    qwen_proj_path = '/leonardo_work/EUHPC_R04_192/fmohamma/TRIG/data/X2I/X2I-Qwen2.5VL-7B.bin'
if args.qwen_size == "3b":
    outputs = "./outputs_qwen3b"
    qwen_path = '/mnt/data/group/models/Qwen2.5-VL-3B-Instruct'
    qwen_proj_path = '/mnt/data/group/xuguo/X2I_qwen/checkpoints/qwen3b/diffusion_pytorch_model.bin'


num_steps = args.num_steps
flux_path = args.flux_path
use_answer = args.use_answer
num_gen_imgs = args.num_gen_imgs


print("=" * 70)
print("🚀 开始加载 Qwen-VL X2I 模型")
print("=" * 70)

print(f"📋 模型配置:")
print(f"  - Qwen设备: {device_qwen}")
print(f"  - FLUX设备: {device_flux}")
print(f"  - 数据类型: {dtype}")
print(f"  - Qwen模型大小: {args.qwen_size}")
print(f"  - Qwen模型路径: {qwen_path}")
print(f"  - Flux模型路径: {flux_path}")
print(f"  - 投影层权重: {qwen_proj_path}")
print(f"  - 推理步数: {num_steps}")

# 设置GPU内存
if args.multi_gpu:
    torch.cuda.set_device(device_qwen)
    print(f"💾 GPU内存状态:")
    for i in range(torch.cuda.device_count()):
        print(f"  - GPU {i}: {torch.cuda.get_device_properties(i).name}")
        print(f"    总内存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f}GB")
else:
    torch.cuda.set_device(device_qwen)

# 1. 加载Qwen编码器
print(f"\n🧠 [1/7] 加载Qwen2.5-VL-{args.qwen_size.upper()}编码器...")
start_time = time.time()
qwen_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    qwen_path, 
    dtype=torch.bfloat16,
    local_files_only=True
).eval().to(device=device_qwen)
print(f"  ✅ Qwen编码器加载完成 ({time.time() - start_time:.2f}s)")
print(f"  📍 已放置在设备: {device_qwen}")

# 2. 加载Qwen处理器
print(f"\n🔤 [2/7] 加载Qwen处理器...")
start_time = time.time()
qwen_processor = AutoProcessor.from_pretrained(qwen_path, local_files_only=True)
print(f"  ✅ Qwen处理器加载完成 ({time.time() - start_time:.2f}s)")

# 3. 加载CLIP分词器和模型
print(f"\n📝 [3/7] 加载CLIP分词器和文本编码器...")
start_time = time.time()
clip_tokenizer = CLIPTokenizer.from_pretrained(flux_path, subfolder="tokenizer", dtype=dtype)
clip_model = CLIPTextModel.from_pretrained(flux_path, subfolder="text_encoder", dtype=dtype).to(device_flux).eval()
print(f"  ✅ CLIP组件加载完成 ({time.time() - start_time:.2f}s)")
print(f"  📍 已放置在设备: {device_flux}")

# 4. 加载T5分词器和模型
print(f"\n📚 [4/7] 加载T5分词器和文本编码器...")
start_time = time.time()
t5_tokenizer = T5TokenizerFast.from_pretrained(flux_path, subfolder="tokenizer_2", dtype=dtype)
t5_model = T5EncoderModel.from_pretrained(flux_path, subfolder="text_encoder_2", dtype=dtype).to(device_flux).eval()
print(f"  ✅ T5组件加载完成 ({time.time() - start_time:.2f}s)")
print(f"  📍 已放置在设备: {device_flux}")

# 5. 加载FLUX Pipeline
print(f"\n🌊 [5/7] 加载FLUX Pipeline...")
start_time = time.time()
pipeline = FluxPipeline.from_pretrained(
    flux_path, 
    text_encoder=None, 
    text_encoder_2=None,
    tokenizer=None, 
    tokenizer_2=None, 
    vae=None, 
    torch_dtype=dtype
).to(device_flux)
print(f"  ✅ FLUX Pipeline加载完成 ({time.time() - start_time:.2f}s)")
print(f"  📍 已放置在设备: {device_flux}")

# 6. 加载VAE
print(f"\n🖼️  [6/7] 加载VAE...")
start_time = time.time()
vae = AutoencoderKL.from_pretrained(flux_path, subfolder="vae", torch_dtype=dtype).to(device_flux)
print(f"  ✅ VAE加载完成 ({time.time() - start_time:.2f}s)")
print(f"  📍 已放置在设备: {device_flux}")

def get_proj(proj_path):
    print(f"\n⚖️  [7/7] 加载投影层...")
    start_time = time.time()
    print(f"  📁 投影层权重路径: {proj_path}")

    if args.qwen_size == "3b":
        proj = create_proj3_qwen3b(in_channels=37, use_t5=False, use_scale=False, use_cnn=True)
        print(f"  🔧 创建3B投影层架构")
    if args.qwen_size == "7b":
        proj = create_proj3_qwen7b(in_channels=29, use_t5=False, use_scale=False, use_cnn=True)
        print(f"  🔧 创建7B投影层架构")

    state_dict = torch.load(proj_path, map_location="cpu")
    print(f"  📊 加载权重包含 {len(state_dict)} 个参数")
    
    state_dict_new = {}
    for k,v in state_dict.items():
        k_new = k.replace("module.","")
        state_dict_new[k_new] = v

    proj.load_state_dict(state_dict_new)
    proj.to(device=device_qwen, dtype=dtype)
    proj.eval()
    print(f"  ✅ 投影层加载完成 ({time.time() - start_time:.2f}s)")
    print(f"  📍 已放置在设备: {device_qwen}")
    return proj

qwen_proj = get_proj(qwen_proj_path)

print("\n" + "=" * 70)
print("🎉 所有模型组件加载完成！")

# 显示GPU内存使用情况
if args.multi_gpu:
    print(f"\n💾 当前GPU内存使用情况:")
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  - GPU {i}: 已分配 {allocated:.1f}GB / 已预留 {reserved:.1f}GB / 总计 {total:.1f}GB")
else:
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\n💾 GPU内存使用: 已分配 {allocated:.1f}GB / 已预留 {reserved:.1f}GB / 总计 {total:.1f}GB")

print("=" * 70)
def get_t5_input_embeds(text_prompt=None):
    text_input_ids = clip_tokenizer(
        text_prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_overflowing_tokens=False,
        return_length=False,
        return_tensors="pt",
    ).input_ids
    pooled_prompt_embeds = clip_model(text_input_ids.to(device_flux), output_hidden_states=False).pooler_output.to(dtype=torch.bfloat16, device=device_flux)
    text_input_ids = t5_tokenizer(
        text_prompt,
        # padding="max_length",
        # max_length=512,
        # truncation=True,
        return_overflowing_tokens=False,
        return_length=False,
        return_tensors="pt",
    ).input_ids
    print(f"get_t5_input_embeds input_ids: {text_input_ids.shape}")
    prompt_embeds = t5_model(text_input_ids.to(device_flux), output_hidden_states=False)[0].to(dtype=torch.bfloat16, device=device_flux)
    return pooled_prompt_embeds, prompt_embeds

def get_text_embeddings(output_hidden_state):
    if args.qwen_size == "3b":
        text_embeddings = torch.cat(output_hidden_state["hidden_states"][0]).unsqueeze(0)
    if args.qwen_size == "7b":
        if use_answer:
            text_embeddings = []
            for hidden_states in  output_hidden_state["hidden_states"][1:]:
                text_embeddings.append(torch.cat(hidden_states))
            text_embeddings = torch.cat(text_embeddings,dim=1).unsqueeze(0)
        else:
            text_embeddings = torch.cat(output_hidden_state["hidden_states"][0]).unsqueeze(0)
    return text_embeddings



def get_qwen_inputs_embeds(videos=None, images=None, text_prompt=None, proj=qwen_proj):
    message = [{"role": "user", "content": []}]
    image_list = []
    if images is not None and len(images) > 0:
        for image in images:
            image_input = Image.open(image).convert('RGB').resize(size=(128, 128))
            message[0]["content"].append({"type": "image", "image": image_input})
            image_list.append(image_input)
    
    if videos is not None and len(videos) > 0:
        assert len(videos) == 1
        video = videos[0]
        message[0]["content"].append({
                    "type": "video",
                    "video": video,
                    "max_pixels": 128 * 128,
                    "fps": 1.0,
                })
        _, video_inputs = process_vision_info(message)

    if videos is None:
        video_inputs = None
             

    if text_prompt is not None:
        message[0]["content"].append({"type": "text", "text": text_prompt})
    
    prompt = qwen_processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

    inputs = qwen_processor(
    text=[prompt],
    images=None if len(image_list) == 0 else image_list,
    videos= video_inputs,
    padding="max_length",
    max_length=512, 
    truncation=True, 
    return_tensors="pt",
    
    ).to(device_qwen)

    output_hidden_state = qwen_encoder.generate(**inputs, max_new_tokens=128,output_hidden_states=True,return_dict_in_generate=True)

    text_embeddings = get_text_embeddings(output_hidden_state)
    pooled_prompt_embeds, prompt_embeds = proj(text_embeddings)
    
    # 如果是多GPU模式，需要将嵌入移到FLUX设备上
    if args.multi_gpu and device_qwen != device_flux:
        pooled_prompt_embeds = pooled_prompt_embeds.to(device_flux)
        prompt_embeds = prompt_embeds.to(device_flux)
    
    return pooled_prompt_embeds, prompt_embeds


@torch.no_grad()
def generate(pooled_prompt_embeds, prompt_embeds, outputs, filename, seed=None, height=1024, width=1024):
    print(f"\n  🎨 生成图像: {filename}")
    gen_start_time = time.time()
    os.makedirs(outputs, exist_ok=True)

    print(f"      📐 图像尺寸: {height}x{width}")
    print(f"      🎲 随机种子: {seed if seed is not None else '随机'}")
    print(f"      🔄 推理步数: {num_steps}")
    
    # 扩散生成
    print(f"      🌀 [1/3] FLUX扩散生成...")
    step_start = time.time()
    if seed is not None:
        latents = pipeline(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            num_inference_steps=num_steps,
            guidance_scale=3.5,
            height=height,
            width=width,
            output_type="latent",
            generator=torch.Generator(device_flux).manual_seed(seed)
        ).images
    else:
        latents = pipeline(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            num_inference_steps=num_steps,
            guidance_scale=3.5,
            height=height,
            width=width,
            output_type="latent",
        ).images
    print(f"          ✅ 扩散生成完成 ({time.time() - step_start:.2f}s)")

    # VAE解码
    print(f"      🖼️  [2/3] VAE解码...")
    step_start = time.time()
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels))
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
    vae_scale_factor = pipeline.vae_scale_factor
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2)

    latents = FluxPipeline._unpack_latents(latents, height, width, vae_scale_factor)
    latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
    image = vae.decode(latents, return_dict=False)[0]

    latents = FluxPipeline._unpack_latents(latents, height, width, vae_scale_factor)
    latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
    image = vae.decode(latents, return_dict=False)[0]
    print(f"          ✅ VAE解码完成 ({time.time() - step_start:.2f}s)")

    # 后处理和保存
    print(f"      💾 [3/3] 后处理和保存...")
    step_start = time.time()
    image = image_processor.postprocess(image, output_type="pil")
    output_path = f"{outputs}/{filename}.jpg"
    image[0].save(output_path)
    print(f"          ✅ 图像已保存: {output_path} ({time.time() - step_start:.2f}s)")
    
    print(f"      🎉 图像生成完成，总用时: {time.time() - gen_start_time:.2f}s")


def text2image(outputs=outputs):
    print(f"\n🎨 开始文本到图像生成任务")
    print("=" * 50)
    
    outputs = os.path.join(outputs, "text2image")
    os.makedirs(outputs, exist_ok=True)
    print(f"📁 输出目录: {outputs}")

    prompts = [{
        "EN": "A majestic elephant stands gracefully in a sun-drenched savannah, its textured gray skin glistening under the warm golden light of the late afternoon sun. The elephant, with large, expressive ears and a gently curved trunk, is posed mid-stride, kicking up a cloud of dust as it moves towards a shimmering waterhole surrounded by lush green acacia trees. The scene is painted in a vibrant impressionistic style, utilizing a rich palette of earthy tones, soft greens, and warm yellows that evoke a sense of tranquility and connection to nature. The camera angle is low, capturing the elephant's grandeur against the expansive sky, dotted with wispy clouds. In the foreground, a few colorful wildflowers bloom, adding splashes of color, while a distant herd of antelope grazes peacefully, enhancing the serene atmosphere of this enchanting moment in the wild.",
        "ZH": "一只雄伟的大象优雅地站在阳光普照的草原上，它那纹理分明的灰色皮肤在午后温暖的金色阳光下闪闪发光。大象有着大而富有表现力的耳朵和微微弯曲的鼻子，正处于行走的姿态中，踢起一团尘土，向被茂密的绿色金合欢树环绕的波光粼粼的水坑走去。这幅场景以一种充满活力的印象派风格绘制，使用了丰富的泥土色调、柔和的绿色和温暖的黄色，唤起了一种宁静与自然相连的感觉。拍摄角度较低，捕捉到了大象在点缀着缕缕白云的广阔天空下的壮观景象。前景中，几朵色彩斑斓的野花盛开，增添了颜色的斑驳，而远处一群羚羊正在平静地吃草，增强了这片野外迷人时刻的宁静氛围。",
        "DE": "Ein majestätischer Elefant steht anmutig in einer sonnenüberfluteten Savanne, seine texturierte graue Haut glänzt im warmen goldenen Licht der Nachmittagssonne. Der Elefant mit großen ausdrucksstarken Ohren und einem sanft gebogenen Rüssel ist mitten im Schritt, wirbelt eine Staubwolke auf, während er sich auf einen schimmernden Wasserloch zubewegt, umgeben von üppigen grünen Akazienbäumen. Die Szene ist in einem lebendigen impressionistischen Stil gemalt, mit einem reichen Farbspektrum aus Erdtönen, weichen Grüntönen und warmen Gelbtönen, die ein Gefühl von Ruhe und Verbundenheit mit der Natur vermitteln. Der Kamerawinkel ist niedrig, um die Erhabenheit des Elefanten vor dem weiten Himmel, gesprenkelt mit hauchdünnen Wolken, einzufangen. Im Vordergrund blühen einige bunte Wildblumen und setzen farbliche Akzente, während sich in der Ferne eine Herde Antilopen friedlich grasend aufhält und die ruhige Atmosphäre dieses bezaubernden Moments in der Wildnis verstärkt.",
        "FR": "Un éléphant majestueux se tient gracieusement dans une savane baignée de soleil, sa peau grise texturée étincelle sous la douce lumière dorée du soleil de l'après-midi. L'éléphant, avec ses grandes oreilles expressives et sa trompe légèrement courbée, est représenté en plein mouvement, soulevant un nuage de poussière alors qu'il se dirige vers un point d'eau scintillant entouré d'acacias verts luxuriants. La scène est peinte dans un style impressionniste dynamique, utilisant une palette riche de tons terreux, de verts doux et de jaunes chauds qui évoquent un sentiment de tranquillité et de connexion à la nature. L'angle de la caméra est bas, capturant la grandeur de l'éléphant contre le vaste ciel parsemé de légers nuages. En avant-plan, quelques fleurs sauvages colorées sont en pleine floraison, ajoutant des touches de couleur, tandis qu'un troupeau d'antilopes paît paisiblement au loin, renforçant l'atmosphère sereine de ce moment enchanteur dans la nature.",
        "JA": "荘厳な象が日差しに照らされたサバンナに優雅に立ち、そのテクスチャーのある灰色の肌は午後の暖かい黄金色の光の中で輝いています。大きな表現力豊かな耳と優しく曲がった鼻を持つ象は歩みを進める姿勢で、足元から塵の雲を巻き上げながら、緑豊かなアカシアの木々に囲まれたキラキラとした水たまりに向かって移動しています。このシーンは活気に満ちた印象派のスタイルで描かれ、落ち着いた大地の色調、柔らかな緑、そして温かい黄色の豊かなパレットを使用して、平穏さと自然とのつながりを感じさせるものです。カメラアングルは低く設定され、ほんのりとした雲が浮かぶ広大な空を背景に象の壮大さを捉えています。手前にいくつかの色彩豊かな野の花が咲き、彩りを添え、遠くでは一群のインパラが平和に草を食んでおり、野生の中のこの魅惑的な瞬間の静けさを強調しています。",
        "VI": "Một con voi uy nghi đứng thanh lịch trên thảo nguyên đầy nắng, làn da xám của nó có kết cấu sáng lên dưới ánh nắng vàng ấm áp của buổi chiều muộn. Con voi với đôi tai lớn biểu cảm và chiếc vòi cong nhẹ đang đi, tạo ra một đám bụi khi nó di chuyển về phía một hồ nước lấp lánh được bao quanh bởi những cây keo xanh tươi. Cảnh tượng này được vẽ theo phong cách ấn tượng sống động, sử dụng bảng màu đa dạng gồm các sắc độ đất, màu xanh mềm mại và màu vàng ấm áp gợi lên cảm giác yên bình và sự gắn kết với thiên nhiên. Góc máy thấp, chụp được sự hùng vĩ của con voi trước bầu trời rộng lớn điểm xuyết những đám mây mỏng. Phía trước, một vài loài hoa dại nhiều màu sắc nở rộ, thêm vào đó là hình ảnh đàn linh dương ở xa đang ăn cỏ một cách yên bình, làm tăng thêm không khí tĩnh lặng của khoảnh khắc quyến rũ giữa thiên nhiên hoang dã.",
    }]

    total_tasks = len(prompts) * len(prompts[0]) * num_gen_imgs
    current_task = 0
    
    print(f"📊 任务统计:")
    print(f"  - 提示词集合数: {len(prompts)}")
    print(f"  - 每个集合的语言数: {len(prompts[0])}")
    print(f"  - 每个提示词的生成数: {num_gen_imgs}")
    print(f"  - 总任务数: {total_tasks}")
    
    task_start_time = time.time()

    for index, prompt_dict in enumerate(prompts):
        print(f"\n📝 处理提示词集合 {index + 1}/{len(prompts)}")
        
        for key, prompt in prompt_dict.items():
            print(f"\n  🌍 语言: {key}")
            print(f"  📄 提示词: {prompt[:100]}...")
            
            for i in range(num_gen_imgs):
                current_task += 1
                print(f"\n  🔄 [{current_task}/{total_tasks}] 处理 {key} 语言，第 {i+1} 张图像")
                
                # 文本编码
                print(f"    📝 [1/2] Qwen文本编码...")
                encoding_start = time.time()
                pooled_prompt_embeds, prompt_embeds = get_qwen_inputs_embeds(text_prompt=prompt)
                print(f"        ✅ 文本编码完成 ({time.time() - encoding_start:.2f}s)")
                print(f"        📊 嵌入维度: pooled={pooled_prompt_embeds.shape}, prompt={prompt_embeds.shape}")
                
                # 图像生成
                print(f"    🎨 [2/2] 图像生成...")
                generate(pooled_prompt_embeds, prompt_embeds, outputs=outputs, filename=f"{index}_{key}_{i}")
                
                # 每5个任务显示一次内存使用情况
                if args.multi_gpu and current_task % 5 == 0:
                    print(f"    💾 内存状态检查:")
                    for gpu_id in range(torch.cuda.device_count()):
                        allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
                        print(f"      GPU {gpu_id}: {allocated:.1f}GB")
    
    print(f"\n🎉 所有文本到图像生成任务完成！")
    print(f"⏱️  总用时: {time.time() - task_start_time:.2f}s")
    print(f"📁 所有图像已保存到: {outputs}")
    
    # 最终内存状态
    if args.multi_gpu:
        print(f"\n💾 最终GPU内存使用情况:")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  - GPU {i}: 已分配 {allocated:.1f}GB / 已预留 {reserved:.1f}GB / 总计 {total:.1f}GB")
    
    print("=" * 50)



if __name__ == "__main__":
    print(f"\n📋 运行参数:")
    print(f"  - 多GPU模式: {'启用' if args.multi_gpu else '关闭'}")
    print(f"  - 生成步数: {num_steps}")
    print(f"  - 每个提示词生成数: {num_gen_imgs}")
    print(f"  - Qwen模型大小: {args.qwen_size}")
    
    if args.multi_gpu and torch.cuda.device_count() < 2:
        print(f"⚠️  警告: 启用了多GPU模式但只检测到 {torch.cuda.device_count()} 张GPU卡")
        print(f"   建议使用单GPU模式或确保有至少2张GPU可用")
    
    text2image()


    
    

    


