import torch

def removekey(d, listofkeys):
    r = dict(d)
    for key in listofkeys:
        if key in r:
            print('key: {} is removed'.format(key))
            r.pop(key)
        else:
            print('key: {} not found'.format(key))
    return r

# Paths
model_path = "/home/kim/mini_cond_sgg_env/checkpoints/pretrained_faster_rcnn/model_final.pth"
save_path = "/home/kim/mini_cond_sgg_env/checkpoints/pretrained_faster_rcnn/trimmed_model.pth"

# Load the model
model = torch.load(model_path)

# Print the keys in the state dictionary
print("Keys in the model's state dictionary:")
for key in model['model'].keys():
    print(key)

# Specify the layers to be removed
layers_to_remove = [
    "module.roi_heads.box.predictor.cls_score.weight",
    "module.roi_heads.box.predictor.cls_score.bias",
    "module.roi_heads.box.predictor.bbox_pred.weight",
    "module.roi_heads.box.predictor.bbox_pred.bias"
]

# Remove the specified layers
model['model'] = removekey(model['model'], layers_to_remove)

# Save the modified model
torch.save(model, save_path)
print('Modified model saved to {}.'.format(save_path))
