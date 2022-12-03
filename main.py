
'''
this is cell for command of main
'''

import argparse

import matplotlib.pyplot as plt
import torch.backends.mps
from detect_face import *
from emotion_classification import *
from xai import *
from detect_face import *
import time

_class_names = ['Neutral', 'Happiness', 'Sadness',
                   'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt']
_class_names = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
_FACE_SIZE = 260
def main(args) :

    ## setting device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif args.device == 'mps' and torch.backends.mps.is_available() :
        device = torch.device('mps')
    else :
        device = torch.device('cpu')

    # define model for face detector and emotion_classifier
    detector , cfg = get_face_detector(args)
    emo_classifier = load_emomodel(args).eval()
    detector.to(device)
    emo_classifier.to(device)
    transform, inv_transform = get_transform()

    if args.save_image:
        if not os.path.exists("./xai_result/"):
            os.makedirs("./xai_result/")


    ## read image files in imagefolder
    for image_num , image_files in enumerate(os.listdir(args.image_path)):

        if image_num == 0 :
            print('directory has {} images'.format(len(os.listdir(args.image_path))))
        filename = image_files.split('.')[-2]
        print('file {} processing...'.format(filename))

        ## read file
        image_file = os.path.join(args.image_path, image_files)
        image_raw = cv2.imread(image_file, cv2.IMREAD_COLOR)
        image_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)

        # get face_detection result and cropped image_raw list
        dets , _  = detect_faces(args, detector = detector , image_raw= image_raw ,cfg = cfg, device= device)
        cropped_image_list = get_cropped_img_raw(args, image_raw, dets, filename)

        print(" file {} has {} faces".format(filename, len(cropped_image_list)))


        ## get faces xai result on each iamge file
        for i , face_raw in enumerate(cropped_image_list) :
            try:
                face_raw = cv2.resize(face_raw ,(_FACE_SIZE , _FACE_SIZE))
            except :
                continue
            filename = filename + str(i+1)

            # get result predicted by network.
            face_tensor = transform(Image.fromarray(face_raw)).unsqueeze_(0).to(device)

            pred = int(torch.argmax(emo_classifier(face_tensor)))
            ## Visualize SHAP
            print(" doing shap...")
            shap_value = get_shap(args ,face_tensor , class_names  = _class_names , model = emo_classifier ,device = device, filename= filename, FACE_SIZE= _FACE_SIZE)

            ## Visualize LIME
            print(" doing lime...")
            lime_value = get_lime(args , face_raw, model = emo_classifier, pred= pred , filename= filename ,FACE_SIZE= _FACE_SIZE, class_names =_class_names)

            ## 여기 결과들을 찍는게 좋을 듯..!
    print('end of analysis check directory')
    return

if __name__ == '__main__' :


    parser =  argparse.ArgumentParser()
    parser.add_argument("--image_path" ,  default= 'generated_photos' , help = 'directory to your image_data')

    ## for SHAP
    parser.add_argument('--n_evals', default=1000, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--topk', default=4, type=int)

    ## for LIME
    parser.add_argument('--n_samples',default = 1000 ,type = int)

    ## arguments for FaceDetector_RetinaFace
    parser.add_argument('-m', '--trained_model', default='./weights/Resnet50_Final.pth',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
    parser.add_argument('--origin_size', default=True, type=str, help='Whether use origin image size to evaluate')
    parser.add_argument('--save_folder', default='./widerface_evaluate/widerface_txt/', type=str,
                        help='Dir to save txt results')
    parser.add_argument('--cpu', action="store_true", default=True, help='Use cpu inference')

    #parser.add_argument('--dataset_folder', default='./data/widerface/val/images/', type=str, help='dataset path')
    parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
    parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
    parser.add_argument('--vis_threshold', default=0.5, type=float, help='visualization_threshold')
    parser.add_argument('--face_detector', default="Resnet")
    parser.add_argument('--device', default='cpu')

    args = parser.parse_args()

    main(args)


