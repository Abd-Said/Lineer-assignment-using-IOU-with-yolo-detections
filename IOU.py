import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

import math
import numpy as np

def min_zero_row(zero_mat, mark_zero):
    

    min_row = [99999, -1]

    for row_num in range(zero_mat.shape[0]): 
        if np.sum(zero_mat[row_num] == True) > 0 and min_row[0] > np.sum(zero_mat[row_num] == True):
            min_row = [np.sum(zero_mat[row_num] == True), row_num]

    # Marked the specific row and column as False
    zero_index = np.where(zero_mat[min_row[1]] == True)[0][0]
    mark_zero.append((min_row[1], zero_index))
    zero_mat[min_row[1], :] = False
    zero_mat[:, zero_index] = False

def mark_matrix(mat):

    '''
    Finding the returning possible solutions for LAP problem.
    '''

    #Transform the matrix to boolean matrix(0 = True, others = False)
    cur_mat = mat
    zero_bool_mat = (cur_mat == 0)
    zero_bool_mat_copy = zero_bool_mat.copy()

    #Recording possible answer positions by marked_zero
    marked_zero = []
    while (True in zero_bool_mat_copy):
        min_zero_row(zero_bool_mat_copy, marked_zero)
    
    #Recording the row and column positions seperately.
    marked_zero_row = []
    marked_zero_col = []
    for i in range(len(marked_zero)):
        marked_zero_row.append(marked_zero[i][0])
        marked_zero_col.append(marked_zero[i][1])

    #Step 2-2-1
    non_marked_row = list(set(range(cur_mat.shape[0])) - set(marked_zero_row))
    
    marked_cols = []
    check_switch = True
    while check_switch:
        check_switch = False
        for i in range(len(non_marked_row)):
            row_array = zero_bool_mat[non_marked_row[i], :]
            for j in range(row_array.shape[0]):
                #Step 2-2-2
                if row_array[j] == True and j not in marked_cols:
                    #Step 2-2-3
                    marked_cols.append(j)
                    check_switch = True

        for row_num, col_num in marked_zero:
            #Step 2-2-4
            if row_num not in non_marked_row and col_num in marked_cols:
                #Step 2-2-5
                non_marked_row.append(row_num)
                check_switch = True
    #Step 2-2-6
    marked_rows = list(set(range(mat.shape[0])) - set(non_marked_row))

    return(marked_zero, marked_rows, marked_cols)

def adjust_matrix(mat, cover_rows, cover_cols):
    cur_mat = mat
    non_zero_element = []

    #Step 4-1
    for row in range(len(cur_mat)):
        if row not in cover_rows:
            for i in range(len(cur_mat[row])):
                if i not in cover_cols:
                    non_zero_element.append(cur_mat[row][i])
    min_num = min(non_zero_element)

    #Step 4-2
    for row in range(len(cur_mat)):
        if row not in cover_rows:
            for i in range(len(cur_mat[row])):
                if i not in cover_cols:
                    cur_mat[row, i] = cur_mat[row, i] - min_num
    #Step 4-3
    for row in range(len(cover_rows)):  
        for col in range(len(cover_cols)):
            cur_mat[cover_rows[row], cover_cols[col]] = cur_mat[cover_rows[row], cover_cols[col]] + min_num
    return cur_mat

def hungarian_algorithm(mat): 
    dim = mat.shape[0]
    cur_mat = mat

    #Step 1 - Every column and every row subtract its internal minimum
    for row_num in range(mat.shape[0]): 
        cur_mat[row_num] = cur_mat[row_num] - np.min(cur_mat[row_num])
    
    for col_num in range(mat.shape[1]): 
        cur_mat[:,col_num] = cur_mat[:,col_num] - np.min(cur_mat[:,col_num])
    zero_count = 0
    while zero_count < dim:
        #Step 2 & 3
        ans_pos, marked_rows, marked_cols = mark_matrix(cur_mat)
        zero_count = len(marked_rows) + len(marked_cols)

        if zero_count < dim:
            cur_mat = adjust_matrix(cur_mat, marked_rows, marked_cols)

    return ans_pos 

def ans_calculation(mat, pos):
    total = 0
    ans_mat = np.zeros((mat.shape[0], mat.shape[1]))
    for i in range(len(pos)):
        total += mat[pos[i][0], pos[i][1]]
        ans_mat[pos[i][0], pos[i][1]] = mat[pos[i][0], pos[i][1]]
    return total, ans_mat
    



def iou_d2t(trackers, detections):
    # Box format: (x_min, y_min, x_max, y_max)
    mainmat = []
    for track in trackers:
        yanmat = []
        for detects in detections:
            
            # Intersection rectangle coordinates
            inter_xmin = max(track[0], detects[0])
            inter_ymin = max(track[1], detects[1])
            inter_xmax = min(track[2], detects[2])
            inter_ymax = min(track[3], detects[3])

            # Calculate intersection area
            inter_width = max(0, inter_xmax - inter_xmin)
            inter_height = max(0, inter_ymax - inter_ymin)
            intersection_area = inter_width * inter_height

            # Calculate the area of both boxes
            area_track = (track[2] - track[0]) * (track[3] - track[1])
            area_detects = (detects[2] - detects[0]) * (detects[3] - detects[1])

            # Calculate Union area
            union_area = area_track + area_detects - intersection_area

            # Calculate IOU
            iou = intersection_area / union_area if union_area > 0 else 0
            iou = 1-iou
            yanmat.append(iou)
        mainmat.append(yanmat)

    return mainmat

def iou_t2d(trackers, detections):
    # Box format: (x_min, y_min, x_max, y_max)
    mainmat = []
    for detects in detections:
        yanmat = []
        for track in trackers:
            
            # Intersection rectangle coordinates
            inter_xmin = max(track[0], detects[0])
            inter_ymin = max(track[1], detects[1])
            inter_xmax = min(track[2], detects[2])
            inter_ymax = min(track[3], detects[3])

            # Calculate intersection area
            inter_width = max(0, inter_xmax - inter_xmin)
            inter_height = max(0, inter_ymax - inter_ymin)
            intersection_area = inter_width * inter_height

            # Calculate the area of both boxes
            area_track = (track[2] - track[0]) * (track[3] - track[1])
            area_detects = (detects[2] - detects[0]) * (detects[3] - detects[1])

            # Calculate Union area
            union_area = area_track + area_detects - intersection_area

            # Calculate IOU
            iou = intersection_area / union_area if union_area > 0 else 0
            iou = 1-iou
            yanmat.append(iou)
        mainmat.append(yanmat)

    return mainmat




def detect():

    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:  
        view_img = check_imshow()
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()


    first_detection = False
    first_tracking = True
    first_id = True
    first_id_success = False
    id_opened = True
    first_t = False
    tracker_input = []
    detect_input = []
    tracker_list = []
    tracked_list = []
    boxes = []
    id = 0

   
    
  


    for path, img, im0s, vid_cap in dataset:

        
        dongu = 0
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)#non_max_supression*
        t3 = time_synchronized()
        
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                tracker_input = []
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                i = 0
                for *xyxy, conf, cls in reversed(det):
                

                    if conf>=0.25:
                        detection_tracker = []
                        #detections.append(xyxy)
                        label = f'{names[int(cls)]} {conf:.2f}'
                        #plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                        x = []
                        
                        for x in xyxy:
                            detection_tracker.append(int(x.item()))
                        tracker_input.append(detection_tracker)
                        
                        first_tracking = True
                    
                        i +=1

            if len(tracker_input) > 0: 
                first_detection = True
                detect_input = []
                detect_input = tracker_input.copy()
                

            
            if first_t:
            
                if len(detect_input) > len(boxes):
                    iou_score = iou_d2t(boxes, detect_input)
                    i=0
                    k = []
                    new_dets = []
                    while i < len(detect_input): 
                        k.append(1)
                        i += 1
                    i=0
                    m=0
                    
                    n = len(boxes) 
                    while m < n:
                        if (iou_score[m]==k):
                            del boxes[i]
                            del tracker_list[i]
                            m += 1
                        else:
                            i += 1
                            m += 1
                
                    iou_score = np.array(iou_score)
                    iou_score = np.transpose(iou_score.copy())
                    iou_score = iou_score.tolist()

                    i=0
                    k = []

                    while i < len(iou_score[0]): 
                        k.append(1)
                        i += 1

                    i = 0
                    n = len(boxes)
                    m = 0

                    while m < n:
                        if (iou_score[m]==k):
                            new_dets.append(detect_input[i])
                            del detect_input[i]
                            m += 1
                        else:
                            i += 1
                            m += 1
                    

                    if len(detect_input) >= len(boxes):

                        boxes_main = []
                        boxes_main = boxes.copy()
                        detect_input_main =[]
                        detect_input_main = detect_input.copy()
                        t2d = -1
                        iou = iou_d2t(boxes, detect_input)
                        iou = np.array(iou.copy())
                        iou_score = hungarian_algorithm(iou)
                        
                        
                        g=0
                        s=0
                        j=0
                        for i in range(len(detect_input_main)):
                            s=0
                            for k in range(len(iou_score)):
                                j = iou_score[k][1]
                                if (j==i):
                                    g += 1
                                    break
                                    
                                else:
                                    s += 1
                                    if s==(len(iou_score)):
                                        
                                        new_dets.append(detect_input_main[i])
                                        del detect_input[g]


                    else:
                        boxes_main = []
                        boxes_main = boxes.copy()
                        detect_input_main =[]
                        detect_input_main = detect_input.copy()

                        iou = iou_t2d(boxes, detect_input)
                        t2d = 1
                        iou = np.array(iou.copy()) 
                        iou_score = hungarian_algorithm(iou)
                        s=0
                        j=0
                        g=0
                        for i in range(len(detect_input_main)):
                            
                            s=0
                        
                            for k in range(len(iou_score)):
                                
                                j = iou_score[k][0]
                            
                                if (j==i):
                                    g += 1
                                    break
                                
                                else:
                                    s += 1
                                    if s==(len(iou_score)):
                                        new_dets.append(detect_input_main[i])
                                        del detect_input[g]


                else:
                    
                    iou_score = iou_t2d(boxes, detect_input)
                    

                    i=0
                    k = []
                    new_dets = []
                    while i < len(boxes): 
                        k.append(1)
                        i += 1
                    

                    i=0
                    m=0
                    n = len(detect_input) 
                    while m < n:
                        if (iou_score[m]==k):
                            new_dets.append(detect_input[i])
                            del detect_input[i]
                            m += 1
                        else:
                            i += 1
                            m += 1
                    
                    iou_score = np.array(iou_score)
                    iou_score = np.transpose(iou_score.copy())
                    iou_score = iou_score.tolist()

                    i=0
                    k = []
                    while i < len(iou_score[0]): 
                        k.append(1)
                        i += 1

                    i=0
                    n = len(boxes)
                    m=0
                    while m < n:
                        if (iou_score[m]==k):
                            del tracker_list[i]
                            del boxes[i]
                            m += 1
                        else:
                            i += 1
                            m += 1

                    if len(detect_input) >= len(boxes):

                        boxes_main = []
                        boxes_main = boxes.copy()
                        detect_input_main =[]
                        detect_input_main = detect_input.copy()

                        iou = iou_d2t(boxes, detect_input)
                        t2d = -1
                        iou = np.array(iou.copy())                      
                        iou_score = hungarian_algorithm(iou)
                    
                        s=0
                        j=0
                        g=0
                        for i in range(len(detect_input_main)):
                        
                            s=0
                            for k in range(len(iou_score)):
                                
                                j = iou_score[k][1]
                            
                                if (j==i):
                                    g += 1
                                    break
                                    
                                
                                else:
                                    s += 1
                                    if s==(len(iou_score)):
                                        new_dets.append(detect_input_main[i])
                                        del detect_input[g]
                        

                    else:

                        boxes_main = []
                        boxes_main = boxes.copy()
                        detect_input_main =[]
                        detect_input_main = detect_input.copy()

                        iou = iou_t2d(boxes, detect_input)
                        iou = np.array(iou.copy())
                        iou_score = hungarian_algorithm(iou)
                        k=0
                        j=0
                        s = 0
                        
                        g=0
                        s=0
                        j=0
                        for i in range(len(detect_input_main)):
                            s=0
                            for k in range(len(iou_score)):  
                                j = iou_score[k][0]
                                if (j==i):
                                    g += 1
                                    break
                                else:
                                    s += 1
                                    if s==(len(iou_score)):
                                        new_dets.append(detect_input_main[i])
                                        del detect_input[g]

                        t2d = 1
                tracked_list = tracker_list.copy()
                tracker_list = []
                if (len(boxes)>= len(detect_input)) and (len(detect_input) > 0):
                
                    iou = iou_t2d(boxes , detect_input)
                    iou = np.array(iou.copy())

                    iou_score = hungarian_algorithm(iou)
                    boxes = []
                    for iou in range(len(iou_score)):
                        
                        tr = iou_score[iou][1]
                        dt = iou_score[iou][0]
                        detected = []
                        detected = detect_input[dt]
                        tracker_ids = tracked_list[tr][0]
                        tracker_list.append((tracker_ids , detected))
                    for detected in new_dets:
                        id+=1
                        tracker_ids = id
                        tracker_list.append((tracker_ids , detected))
                    new_dets = []

                boxes_main = []
                detect_input_main = []
                tracked_list = []
                boxes = []
                for tracky in tracker_list:
                    x,y,w,h = tracky[1]
                    boxes.append((x,y,w,h))
                    cv2.rectangle(im0, (x, y), (w, h), (0, 255, 0), 3)
                    cv2.putText(im0, str(tracky[0]), ( tracky[1][0], tracky[1][1]), cv2.FONT_HERSHEY_SIMPLEX , 0.5 , (255,0,0), 2)


            else:
                
                if webcam:
                    im0 =  im0s[0].copy()
                else: 
                    im0 = im0s
                

                tracker_list = []
                tracked_list = []
                for detected in detect_input:
                    x,y,w,h = detected
                    boxes.append((x,y,w,h))
                    cv2.rectangle(im0, (detected[0], detected[1]), (detected[2], detected[3]), (0, 255, 0), 3)
                    id+=1
    
                    tracker_list.append((id,(x,y,w,h)))
                
                    first_id_success = True
                    id_opened = False 
                    first_tracking = False       
                    first_t = True
                first_id = False
            

            # Stream results
            if view_img:
                cv2.imshow("detect", im0)
                cv2.waitKey(1)  # 1 millisecond
              
                

            



    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))


    with torch.no_grad():
            if opt.update:  # update all models (to fix SourceChangeWarning)
                for opt.weights in ['yolov7.pt']:
                        detect()
                        strip_optimizer(opt.weights)
            else:
                detect()
