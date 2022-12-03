import lime
from lime import lime_image
import numpy as np
import torchvision
from torchvision import transforms
import torch
from torch import nn
import shap
from PIL import Image
import cv2
import timm
import torch.nn.functional as F
from tqdm import tqdm
from omnixai.explainers.vision import LimeImage
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.color import label2rgb
from matplotlib import pyplot as plt
def nhwc_to_nchw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        x = x if x.shape[1] == 3 else x.permute(0, 3, 1, 2)
    elif x.dim() == 3:
        x = x if x.shape[0] == 3 else x.permute(2, 0, 1)
    return x

def nchw_to_nhwc(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        x = x if x.shape[3] == 3 else x.permute(0, 2, 3, 1)
    elif x.dim() == 3:
        x = x if x.shape[2] == 3 else x.permute(1, 2, 0)
    return x

def make_predict_func(model, device ,  phase='shap' ):
    if phase == 'shap':
        def predict(img):
            img = nhwc_to_nchw(torch.Tensor(img))
            output = model(img)
            return output
    elif phase == 'lime':

        def predict(img_raw):
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            normalize = transforms.Normalize(mean = mean , std = std )
            img_tensor = normalize(torch.FloatTensor(img_raw).permute((0,3,1,2))/255.).to(device)
            model.eval()
            outputs = model(img_tensor)
            probas = F.softmax(outputs , dim =1 ).detach().cpu().numpy()
            return probas

    return predict




def get_shap(args , img_tensor, class_names , model ,  filename ,device  = torch.device('cpu') , FACE_SIZE = 260  )  :


    # define predict func
    predict  = make_predict_func(model = model , device = device , phase = 'shap' )

    masker_blur = shap.maskers.Image("blur(128,128)", (FACE_SIZE , FACE_SIZE, 3))
    explainer = shap.Explainer(predict, masker_blur, output_names=class_names)
    shap_values = explainer(img_tensor, max_evals=args.n_evals, batch_size=args.batch_size,
                            outputs=shap.Explanation.argsort.flip[:args.topk])

    _ , inv_transform = get_transform()
    ### post processing of output

    shap_values.data = inv_transform(shap_values.data).cpu().numpy()[0]
    shap_values.values = shap_values.values.transpose(0, 2, 3, 1, 4)
    shap_values.values = [val for val in np.moveaxis(shap_values.values[0], -1, 0)]

    shap.image_plot(shap_values=shap_values.values,
                    pixel_values=shap_values.data,
                    labels=shap_values.output_names,
                    true_labels= None,
                    show = False)
    if args.save_image:

        name = "./xai_result/" + filename + "_SHAP" + ".jpg"
        plt.savefig(name)
        plt.close()

    return shap_values
def get_lime( args , img, model , pred ,  class_names ,filename , device  = torch.device('cpu'), FACE_SIZE = 260  )  :

    predict = make_predict_func(model=model, device =device ,  phase='lime')

    lime_explainer = lime_image.LimeImageExplainer()
    segmenter = SegmentationAlgorithm('slic',n_segments=100, compactnes=1, sigma=1)

    exp = lime_explainer.explain_instance(img, predict, top_labels= 1, num_samples=args.n_samples, batch_size = 1, segmentation_fn= segmenter )
    ## plotting values
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8))
    ax = [ax1, ax2, ax3, ax4]
    for i in ax:
        i.grid(False)

    temp, mask = exp.get_image_and_mask(label=pred, positive_only = True, num_features= 8 , hide_rest = False)
    ax1.imshow(label2rgb(mask, temp, bg_label=0), interpolation='nearest')
    ax1.set_title('Positive Regions for {}'.format(class_names[pred]))

    temp, mask = exp.get_image_and_mask(pred,
                                        positive_only=False,  # 설명 모델이 결과값을 가장 잘 설명하는 이미지 영역만 출력
                                        num_features=8,  # 분할 영역의 크기
                                        hide_rest=False)  # 이미지를 분류하는 데 도움이 되는 서브모듈 외의 모듈도 출력

    ax2.imshow(label2rgb(4 - mask, temp, bg_label=0), interpolation='nearest')  # 역변환
    ax2.set_title('Positive/Negative Regions for {}'.format(class_names[pred]))

    temp, mask = exp.get_image_and_mask(label=pred, positive_only=False, num_features=8, hide_rest=False)
    ax3.imshow(temp, interpolation='nearest')
    ax3.set_title('Show output image only')
    ax4.imshow(mask, interpolation='nearest')  # 정수형 array
    ax4.set_title('Show mask only')

    if args.save_image:
        name = "./xai_result/" + filename + "_LIME" + ".jpg"
        plt.savefig(name)
        plt.close()
    return

def get_transform() :
    IMG_SIZE = 260
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    test_transforms = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,
                                 std=std)
        ]
    )
    inv_transform = transforms.Compose([
        torchvision.transforms.Lambda(nhwc_to_nchw),
        torchvision.transforms.Normalize(
            mean=(-1 * np.array(mean) / np.array(std)).tolist(),
            std=(1 / np.array(std)).tolist()
        ),
        torchvision.transforms.Lambda(nchw_to_nhwc),
    ])
    return test_transforms , inv_transform


