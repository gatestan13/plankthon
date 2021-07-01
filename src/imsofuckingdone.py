import timm
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


CHECKPOINT_PATH = "C:\\Users\\gates\\Desktop\\project\\checkpoint-0.pth.tar"

model = timm.create_model('vit_base_patch16_224_miil_in21k', num_classes=84)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])
model.eval()

config = resolve_data_config({}, model=model)
transform = create_transform(**config)





def get_image_proba(imageinbytes):
	img = Image.open('C:\\Users\\gates\\Desktop\\project\\test\\images\\86177.jpg')
	tensor = transform(img).unsqueeze(0)
	with torch.no_grad():
		out = model(tensor)
	probabilites = torch.nn.functional.softmax(out[0], dim=0)
	print(probabilites.shape)


