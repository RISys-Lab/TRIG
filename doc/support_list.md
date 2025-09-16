## Supported Models
### General Models
1. [OneDiffusion](https://arxiv.org/abs/2411.16318)  
Alias: onediffusion  
Class: general_models.OneDiffusionModel
2. [OmnigGen](https://arxiv.org/abs/2409.11340)  
Alias: omnigen  
Class: general_models.OmniGenModel

### Text-to-Image Model
1.  [DALL·E 3](https://openai.com/dall-e-3/)  
Alias: dalle3  
Class: text_to_image_models.DALLE3Model
2. [Stable Diffusion 1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)  
Alias: sd15  
Class: text_to_image_models.SD15Model
2. [Stable Diffusion XL](https://arxiv.org/abs/2307.01952)  
Alias: sdxl  
Class: text_to_image_models.SDXLModel
3. [Stable Diffusion 3.5](https://huggingface.co/stabilityai/stable-diffusion-3.5-large)  
Alias: sd35   
Class: text_to_image_models.SD35Model
4. [PixArt-Σ](https://arxiv.org/abs/2403.04692)  
Alias: pixart_sigma  
Class: text_to_image_models.PixartSigmaModel
5. [Sana](https://arxiv.org/abs/2410.10629)  
Alias: sana  
Class: text_to_image_models.SanaModel
6. [FLUX](https://bfl.ai/)  
Alias: flux  
Class: text_to_image_models.FLUXModel
7. [Janus-Pro](https://arxiv.org/abs/2501.17811)  
Alias: janus  
Class: text_to_image_models.JanusProModel
8. [Janus-Flow](https://arxiv.org/abs/2411.07975)  
Alias: janus_flow  
Class: text_to_image_models.JanusFlowModel
9. [Qwen-Image](https://arxiv.org/abs/2508.02324)  
Alias: qwen_image  
Class: text_to_image_models.QwenImageModel
### TRIG Models
1. SD1.5 models after RL by DDPO  
Alias: sd15_ddpo  
Class: text_to_image_models.SD15DDPOModel
2. FLUX models after LoRA Finetuning on Knowlege and Ambiguity Dimension  
Alias: flux_ft  
Class: text_to_image_models.FLUXFTModel
### Image-editing Models
1. [InstructPix2Pix](https://arxiv.org/abs/2211.09800)  
Alias: instructp2p  
Class: image_editing_models.InstructPix2PixModel
2. [FreeDiff](https://arxiv.org/abs/2404.11895)  
Alias: freediff  
Class: image_editing_models.FreeDiffModel
3. [FlowEdit](https://arxiv.org/abs/2412.08629)  
Alias: flowedit  
Class: image_editing_models.FlowEditModel
4. [HQ-Edit](https://arxiv.org/abs/2404.09990)  
Alias: hqedit   
Class: image_editing_models.HQEditModel
5. [RF-Inversion](https://arxiv.org/abs/2410.10792)  
Alias: rfinversion   
Class: image_editing_models.RFInversionModel
6. [RF-Solver](https://arxiv.org/abs/2411.04746)  
Alias: rfsolver  
Class: image_editing_models.RFSolverModel
### Subjects Driven Models
'blipdiffusion': 'subject_driven_models.BlipDiffusionModel'
'ssrencoder': 'subject_driven_models.SSREncoderModel'
'ominicontrol': 'subject_driven_models.OminiControlModel'
'xflux': 'subject_driven_models.XFluxModel'
### TRIG DTM models
'sd35_dtm': 'text_to_image_models.SD35DTMModel'
'flux_dtm': 'text_to_image_models.FLUXDTMModel'
'sana_dtm': 'text_to_image_models.SanaDTMModel'
### DTM models with dimension
'sd35_dtm_dim': 'text_to_image_models.SD35DTMDimModel'
'sana_dtm_dim': 'text_to_image_models.SanaDTMDimModel'
'xflux_dtm_dim': 'subject_driven_models.XFluxDTMDimModel'
'hqedit_dtm_dim': 'image_editing_models.HQEditDTMDimModel'
### Multilingual models
'altdiffusion': 'text_to_image_models.AltDiffusionModel'
'mulan': 'text_to_image_models.MuLanModel'
'peadiffusion': 'text_to_image_models.PEADiffusionModel'
