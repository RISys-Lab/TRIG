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
from tqdm import tqdm




parser = argparse.ArgumentParser("Inference", add_help=True)
# parser.add_argument('--minicpm_path', type=str, default="openbmb/MiniCPM-o-2_6")
parser.add_argument('--qwen_size', type=str, default='7b', choices=['3b', '7b'], help="Model size: 1b or 4b")
parser.add_argument('--flux_path', type=str,  default="/leonardo_scratch/fast/EUHPC_R04_192/fmohamma/fast_weights/FLUX.1-schnell")
parser.add_argument('--use_answer', type=bool,  default=False)
parser.add_argument('--num_steps', type=int, default=28)
parser.add_argument('--num_gen_imgs', type=int, default=1)
parser.add_argument('--multi_gpu', action='store_true', help='Enable multi-GPU support')
parser.add_argument('--input_json', type=str, default="/leonardo_work/EUHPC_R04_192/fmohamma/TRIG/dataset/TRIG-multilingual/text-to-image-multilingual.json")
parser.add_argument('--output_dir', type=str, default="/leonardo_work/EUHPC_R04_192/fmohamma/TRIG/data/output/t2i_ml/X2I")
args = parser.parse_args()

# 多GPU设备配置
if args.multi_gpu and torch.cuda.device_count() > 1:
    print(f"🔥 Detected {torch.cuda.device_count()} GPUs")
    device_qwen = "cuda:0"
    device_flux = "cuda:1"
    print(f"📍 Device assignment:")
    print(f"  - Qwen model: {device_qwen}")
    print(f"  - FLUX/VAE: {device_flux}")
else:
    print(f"🔥 Single-GPU mode")
    device_qwen = "cuda:0"
    device_flux = "cuda:0"
    print(f"📍 Device: {device_qwen}")

dtype = torch.bfloat16



outputs = "./outputs_qwen7b"
qwen_path = '/leonardo_work/EUHPC_R04_192/fmohamma/TRIG/data/Qwen2.5-VL-7B-Instruct'
qwen_proj_path = '/leonardo_work/EUHPC_R04_192/fmohamma/TRIG/data/X2I/X2I-Qwen2.5VL-7B.bin'


num_steps = args.num_steps
flux_path = args.flux_path
use_answer = args.use_answer
num_gen_imgs = args.num_gen_imgs


print("=" * 70)
print("🚀 Start loading Qwen-VL X2I models")
print("=" * 70)

print(f"📋 Configuration:")
print(f"  - Qwen device: {device_qwen}")
print(f"  - FLUX device: {device_flux}")
print(f"  - Dtype: {dtype}")
print(f"  - Qwen size: {args.qwen_size}")
print(f"  - Qwen path: {qwen_path}")
print(f"  - FLUX path: {flux_path}")
print(f"  - Projection weights: {qwen_proj_path}")
print(f"  - Inference steps: {num_steps}")

# GPU memory setup
if args.multi_gpu:
    torch.cuda.set_device(device_qwen)
    print(f"💾 GPU memory info:")
    for i in range(torch.cuda.device_count()):
        print(f"  - GPU {i}: {torch.cuda.get_device_properties(i).name}")
        print(f"    Total memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f}GB")
else:
    torch.cuda.set_device(device_qwen)

# 1. Load Qwen encoder
print(f"\n🧠 [1/7] Loading Qwen2.5-VL-{args.qwen_size.upper()} encoder...")
start_time = time.time()
qwen_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    qwen_path, 
    dtype=torch.bfloat16,
    local_files_only=True
).eval().to(device=device_qwen)
print(f"  ✅ Qwen encoder loaded ({time.time() - start_time:.2f}s)")
print(f"  📍 Placed on device: {device_qwen}")

# 2. Load Qwen processor
print(f"\n🔤 [2/7] Loading Qwen processor...")
start_time = time.time()
qwen_processor = AutoProcessor.from_pretrained(qwen_path, local_files_only=True)
print(f"  ✅ Qwen processor loaded ({time.time() - start_time:.2f}s)")

# 3. Load CLIP tokenizer & text encoder
print(f"\n📝 [3/7] Loading CLIP tokenizer & text encoder...")
start_time = time.time()
clip_tokenizer = CLIPTokenizer.from_pretrained(flux_path, subfolder="tokenizer", dtype=dtype)
clip_model = CLIPTextModel.from_pretrained(flux_path, subfolder="text_encoder", dtype=dtype).to(device_flux).eval()
print(f"  ✅ CLIP components loaded ({time.time() - start_time:.2f}s)")
print(f"  📍 Placed on device: {device_flux}")

# 4. Load T5 tokenizer & text encoder
print(f"\n📚 [4/7] Loading T5 tokenizer & text encoder...")
start_time = time.time()
t5_tokenizer = T5TokenizerFast.from_pretrained(flux_path, subfolder="tokenizer_2", dtype=dtype)
t5_model = T5EncoderModel.from_pretrained(flux_path, subfolder="text_encoder_2", dtype=dtype).to(device_flux).eval()
print(f"  ✅ T5 components loaded ({time.time() - start_time:.2f}s)")
print(f"  📍 Placed on device: {device_flux}")

# 5. Load FLUX Pipeline
print(f"\n🌊 [5/7] Loading FLUX Pipeline...")
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
print(f"  ✅ FLUX Pipeline loaded ({time.time() - start_time:.2f}s)")
print(f"  📍 Placed on device: {device_flux}")

# 6. Load VAE
print(f"\n🖼️  [6/7] Loading VAE...")
start_time = time.time()
vae = AutoencoderKL.from_pretrained(flux_path, subfolder="vae", torch_dtype=dtype).to(device_flux)
print(f"  ✅ VAE loaded ({time.time() - start_time:.2f}s)")
print(f"  📍 Placed on device: {device_flux}")

def get_proj(proj_path):
    print(f"\n⚖️  [7/7] Loading projection head...")
    start_time = time.time()
    print(f"  📁 Projection weights path: {proj_path}")

    if args.qwen_size == "3b":
        proj = create_proj3_qwen3b(in_channels=37, use_t5=False, use_scale=False, use_cnn=True)
        print(f"  🔧 Create 3B projection architecture")
    if args.qwen_size == "7b":
        proj = create_proj3_qwen7b(in_channels=29, use_t5=False, use_scale=False, use_cnn=True)
        print(f"  🔧 Create 7B projection architecture")

    state_dict = torch.load(proj_path, map_location="cpu")
    print(f"  📊 Loaded weights contain {len(state_dict)} tensors")
    
    state_dict_new = {}
    for k,v in state_dict.items():
        k_new = k.replace("module.","")
        state_dict_new[k_new] = v

    proj.load_state_dict(state_dict_new)
    proj.to(device=device_qwen, dtype=dtype)
    proj.eval()
    print(f"  ✅ Projection head loaded ({time.time() - start_time:.2f}s)")
    print(f"  📍 Placed on device: {device_qwen}")
    return proj

qwen_proj = get_proj(qwen_proj_path)

print("\n" + "=" * 70)
print("🎉 All model components loaded!")

# Show GPU memory
if args.multi_gpu:
    print(f"\n💾 Current GPU memory usage:")
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  - GPU {i}: Alloc {allocated:.1f}GB / Reserved {reserved:.1f}GB / Total {total:.1f}GB")
else:
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\n💾 GPU memory: Alloc {allocated:.1f}GB / Reserved {reserved:.1f}GB / Total {total:.1f}GB")

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
    gen_start_time = time.time()
    os.makedirs(outputs, exist_ok=True)

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

    vae_scale_factor = pipeline.vae_scale_factor
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2)

    # 兼容不同diffusers版本：latent 可能是打包的3D [B, P, C]，也可能已解包为4D [B, C, H, W]
    if isinstance(latents, (list, tuple)):
        latents = latents[0]
    if isinstance(latents, torch.Tensor):
        if latents.ndim == 3:
            latents = FluxPipeline._unpack_latents(latents, height, width, vae_scale_factor)
        elif latents.ndim == 4:
            pass
        else:
            raise ValueError(f"Unexpected latent shape: {latents.shape}")
    else:
        raise TypeError(f"Unsupported latent type: {type(latents)}")
    latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
    image = vae.decode(latents, return_dict=False)[0]
    image = image_processor.postprocess(image, output_type="pil")
    output_path = f"{outputs}/{filename}.jpg"
    image[0].save(output_path)
    return output_path


def text2image(outputs=outputs):
    print(f"\n🎨 Starting text-to-image generation")
    print("=" * 50)

    outputs = os.path.join(outputs, "text2image")
    os.makedirs(outputs, exist_ok=True)
    print(f"📁 Output directory: {outputs}")

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

    task_start_time = time.time()

    for index, prompt_dict in enumerate(prompts):
        for key, prompt in prompt_dict.items():
            for i in range(num_gen_imgs):
                current_task += 1
                pooled_prompt_embeds, prompt_embeds = get_qwen_inputs_embeds(text_prompt=prompt)
                generate(pooled_prompt_embeds, prompt_embeds, outputs=outputs, filename=f"{index}_{key}_{i}")
    
    print(f"\n🎉 Generation finished")
    print(f"⏱️  Total time: {time.time() - task_start_time:.2f}s")
    print(f"📁 Images saved to: {outputs}")


def json2image(input_json_path: str, base_outputs: str):
    start_time = time.time()
    if args.output_dir is not None:
        out_dir = args.output_dir
    else:
        out_dir = os.path.join(base_outputs, "json2image")
    os.makedirs(out_dir, exist_ok=True)

    with open(input_json_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    if not isinstance(data_list, list):
        raise ValueError("Input JSON must be a list of objects with data_id and prompt")

    generated_count = 0
    skipped_count = 0
    for item in tqdm(data_list, desc="Generating", unit="img"):
        data_id = item.get("data_id")
        prompt = item.get("prompt")
        if data_id is None or prompt is None:
            continue
        filename = str(data_id).replace("/", "_")
        target_path = os.path.join(out_dir, f"{filename}.jpg")
        if os.path.exists(target_path):
            skipped_count += 1
            continue
        pooled_prompt_embeds, prompt_embeds = get_qwen_inputs_embeds(text_prompt=prompt)
        generate(pooled_prompt_embeds, prompt_embeds, outputs=out_dir, filename=filename)
        generated_count += 1

    print(f"\n🎉 Generation finished")
    print(f"⏱️  Total time: {time.time() - start_time:.2f}s")
    print(f"📁 Images saved to: {out_dir}")
    print(f"✅ Generated: {generated_count} | ⏭️ Skipped existing: {skipped_count}")
    
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
    print(f"\n📋 Run args:")
    print(f"  - Multi-GPU: {'enabled' if args.multi_gpu else 'disabled'}")
    print(f"  - Inference steps: {num_steps}")
    print(f"  - Images per prompt: {num_gen_imgs}")
    print(f"  - Qwen size: {args.qwen_size}")

    if args.multi_gpu and torch.cuda.device_count() < 2:
        print(f"⚠️  Warning: Multi-GPU enabled but only {torch.cuda.device_count()} GPU(s) detected")
        print(f"   Suggest using single-GPU mode or ensure at least 2 GPUs are available")

    if args.input_json:
        json2image(args.input_json, outputs)
    else:
        text2image()


    
    

    


