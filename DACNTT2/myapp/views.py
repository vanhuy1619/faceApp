import os
import cv2
from imageUploader.settings import MEDIA_ROOT
from django.shortcuts import render
from .forms import ImageForm
from .models import Image
from .modules.inference import FaceMainipulation


def create_folder(folder_dir):
    os.makedirs(folder_dir, exist_ok=True)
    return folder_dir


def home(request):
    
    if request.method == "POST":
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            
            image_name = str(form.files.get("photo", ""))
            if image_name.endswith(("jpg", "png")):
                # init debug folder
                heatmap_dir = create_folder(os.path.join(MEDIA_ROOT, "heatmap"))
                visualize_dir = create_folder(os.path.join(MEDIA_ROOT, "visualize"))
                # init function face manipulation
                face_manipulation = FaceMainipulation()
                face_manipulation.load_model()
                # inference
                image_path = os.path.join(MEDIA_ROOT, "myimage", image_name)
                heatmap, vis_image = face_manipulation.segment_modified(image_path)
                face_manipulation.free_model()
                # logging inference
                cv2.imwrite(os.path.join(heatmap_dir, image_name), heatmap)
                cv2.imwrite(os.path.join(visualize_dir, image_name), vis_image)
            
    form = ImageForm()
    images = Image.objects.all()
    return render(request, 'myapp/home.html', {'images': images, 'form': form})


def segment_face(request):
    if request.method == "POST":
        data = request.body.decode('utf-8') 
        print('###', data)
            
        # image_name = str(form.files.get("photo", ""))
        # if image_name.endswith(("jpg", "png")):
        #     # init debug folder
        #     heatmap_dir = create_folder(os.path.join(MEDIA_ROOT, "heatmap"))
        #     visualize_dir = create_folder(os.path.join(MEDIA_ROOT, "visualize"))
        #     # init function face manipulation
        #     face_manipulation = FaceMainipulation()
        #     face_manipulation.load_model()
        #     # inference
        #     image_path = os.path.join(MEDIA_ROOT, "myimage", image_name)
        #     heatmap, vis_image = face_manipulation.segment_modified(image_path)
        #     face_manipulation.free_model()
        #     # logging inference
        #     cv2.imwrite(os.path.join(heatmap_dir, image_name), heatmap)
        #     cv2.imwrite(os.path.join(visualize_dir, image_name), vis_image)

        