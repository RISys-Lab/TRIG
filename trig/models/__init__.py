from .base import BaseModel
import importlib

AVAILABLE_Model = {
    'base': 'base.BaseModel', 
    # general models
    'onediffusion': 'general_models.OneDiffusionModel', 
    'omnigen': 'general_models.OmniGenModel', 
    # t2i models 12300 pics
    'dalle3': 'text_to_image_models.DALLE3Model',
    'sdxl': 'text_to_image_models.SDXLModel',
    'pixart_sigma': 'text_to_image_models.PixartSigmaModel', 
    'sana': 'text_to_image_models.SanaModel', 
    'sd35': 'text_to_image_models.SD35Model', 
    'flux': 'text_to_image_models.FLUXModel', 
    'janus': 'text_to_image_models.JanusProModel',
    'janus_flow': 'text_to_image_models.JanusFlowModel',
    'sd15': 'text_to_image_models.SD15Model',
    'qwen_image': 'text_to_image_models.QwenImageModel',
    'hunyuan': 'text_to_image_models.HunyuanModel',
    # SD1.5 models after RL by DDPO
    'sd15_ddpo': 'text_to_image_models.SD15DDPOModel',
    # FLUX models after LoRA FT
    'flux_ft': 'text_to_image_models.FLUXFTModel',
    # p2p models
    'instructp2p': 'image_editing_models.InstructPix2PixModel',
    'freediff': 'image_editing_models.FreeDiffModel',
    'flowedit': 'image_editing_models.FlowEditModel',
    'hqedit': 'image_editing_models.HQEditModel',
    'rfinversion': 'image_editing_models.RFInversionModel',
    'rfsolver': 'image_editing_models.RFSolverModel',
    # subjects models
    'blipdiffusion': 'subject_driven_models.BlipDiffusionModel',
    'ssrencoder': 'subject_driven_models.SSREncoderModel',
    'ominicontrol': 'subject_driven_models.OminiControlModel',
    'xflux': 'subject_driven_models.XFluxModel',
    # DTM models
    'sd35_dtm': 'text_to_image_models.SD35DTMModel',
    'flux_dtm': 'text_to_image_models.FLUXDTMModel',
    'sana_dtm': 'text_to_image_models.SanaDTMModel',
    # DTM models with dimension
    'sd35_dtm_dim': 'text_to_image_models.SD35DTMDimModel',
    'sana_dtm_dim': 'text_to_image_models.SanaDTMDimModel',
    'xflux_dtm_dim': 'subject_driven_models.XFluxDTMDimModel',
    'hqedit_dtm_dim': 'image_editing_models.HQEditDTMDimModel',
    # Multilingual models
    'altdiffusion': 'text_to_image_models.AltDiffusionModel',
    'mulan': 'text_to_image_models.MuLanModel',
    'peadiffusion': 'text_to_image_models.PEADiffusionModel',
}

def import_model(model_name):
    """
    Dynamically import a specific class based on its name.

    :param model_name: The key in AVAILABLE_Model
    :return: The imported class.
    """
    if model_name not in AVAILABLE_Model:
        raise ValueError(f"Metric {model_name} not found in AVAILABLE_Metric.")

    try:
        full_path = f"trig.models.{AVAILABLE_Model[model_name]}"
        print(full_path)
        module_path, class_name = full_path.rsplit('.', 1)

        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except Exception as e:
        raise ImportError(f"Failed to import model {model_name}. Error: {e}")